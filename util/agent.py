import re

import cv2
import numpy as np

from .datasets import DIAGNOSIS_NAMES
from .gate import working_gray


LABEL_NAMES = {0: 'non-cancer', 1: 'cancer'}

TASK = (
    'You are assisting with skin lesion triage from a clinical photograph. '
    'Decide whether the lesion is cancer (basal cell carcinoma, squamous cell carcinoma or melanoma) '
    'or non-cancer (actinic keratosis, nevus or seborrheic keratosis).'
)

ANSWER_FORMAT = 'Final Answer: {"label": "cancer" or "non-cancer", "confidence": a number between 0 and 1}'

SINGLE_SHOT_PROMPT = TASK + '\nLook at the image and reply with exactly one line in this format:\n' + ANSWER_FORMAT

AGENT_PROMPT = (
    TASK + '\nYou can call the following tools before answering:\n{tools}\n'
    'Work step by step. In every reply write one line starting with "Thought:" followed by exactly one of:\n'
    '- a line "Action: <tool name>" to call one tool, then stop and wait for the "Observation:" that will be returned to you;\n'
    '- a line ' + ANSWER_FORMAT + ' when you are ready to decide.'
)

TOOL_DESCRIPTIONS = {
    'segment_lesion': 'segment_lesion: segments the lesion and reports its size and position in the image.',
    'image_statistics': 'image_statistics: reports brightness, contrast, colour and sharpness statistics of the image.',
    'retrieve_similar_cases': 'retrieve_similar_cases: retrieves the {k} most similar reference cases and their confirmed diagnoses.',
}

FINAL_RE = re.compile(r'(?:final\s*answer\s*:?|\banswer\s*:)', re.IGNORECASE)
ACTION_RE = re.compile(r'action\s*:[ \t]*([^\n]*)', re.IGNORECASE)
BARE_ANSWERS = {
    'cancer': 1, 'malignant': 1, 'melanoma': 1, 'basal cell carcinoma': 1, 'squamous cell carcinoma': 1,
    'non-cancer': 0, 'noncancer': 0, 'non cancer': 0, 'not cancer': 0, 'benign': 0,
    'actinic keratosis': 0, 'nevus': 0, 'seborrheic keratosis': 0,
}
LABEL_FIELD_RE = re.compile(r'["\']?label["\']?\s*[:=]\s*["\']?\s*(non[\s_-]?cancer|not\s+cancer|cancer|benign|malignant)', re.IGNORECASE)
CONFIDENCE_RE = re.compile(r'["\']?confidence["\']?\s*[:=]\s*["\']?\s*([0-9]*\.?[0-9]+)\s*(%?)', re.IGNORECASE)


def normalize_label(token):
    token = token.lower()
    if re.match(r'non[\s_-]?cancer|not\s+cancer|benign', token):
        return 0
    if re.match(r'cancer|malignant', token):
        return 1
    return None


def parse_label(text):
    match = LABEL_FIELD_RE.search(text)
    if match:
        return normalize_label(match.group(1))
    match = re.search(r'non[\s_-]?cancer|not\s+cancer|benign|cancer|malignant', text, re.IGNORECASE)
    return normalize_label(match.group(0)) if match else None


def parse_confidence(text):
    match = CONFIDENCE_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2) == '%' or 1.0 < value <= 100.0:
        value = value / 100.0
    return min(max(value, 0.0), 1.0) if value <= 1.0 else None


def parse_bare(text):
    key = re.sub(r'\s+', ' ', re.sub(r'[^a-z\- ]', ' ', text.strip().lower())).strip()
    return BARE_ANSWERS.get(key)


def parse_final(text):
    match = FINAL_RE.search(text)
    if match:
        tail = text[match.end():]
        return parse_label(tail), parse_confidence(tail)
    match = LABEL_FIELD_RE.search(text)
    if match:
        return normalize_label(match.group(1)), parse_confidence(text)
    return parse_bare(text), None


def match_tool(raw, tools):
    key = re.sub(r'[^a-z]', '', raw.strip().strip('`"\'').split('(')[0].lower())
    for tool in tools:
        if key == tool.replace('_', ''):
            return tool, True
    return key, False


def parse_step(text, tools):
    final = FINAL_RE.search(text)
    action = ACTION_RE.search(text)
    if final and (action is None or final.start() < action.start()):
        label, confidence = parse_final(text)
        return {'type': 'final', 'label': label, 'confidence': confidence}
    if action:
        name, valid = match_tool(action.group(1), tools)
        return {'type': 'tool', 'name': name, 'valid': valid}
    if LABEL_FIELD_RE.search(text) or parse_bare(text) is not None:
        label, confidence = parse_final(text)
        return {'type': 'final', 'label': label, 'confidence': confidence}
    return {'type': 'none'}


class ToolBox:
    def __init__(self, record, use_retrieval, k):
        self.record = record
        self.k = k
        self.names = ['segment_lesion', 'image_statistics'] + (['retrieve_similar_cases'] if use_retrieval else [])

    def describe(self):
        return '\n'.join('- ' + TOOL_DESCRIPTIONS[n].format(k=self.k) for n in self.names)

    def __call__(self, name):
        if name == 'segment_lesion':
            s = self.record['segment']
            x0, y0, x1, y1 = s['bbox']
            return (f'The lesion mask covers {100.0 * s["area_fraction"]:.1f}% of the {s["width"]}x{s["height"]} image; '
                    f'bounding box x={x0}-{x1}, y={y0}-{y1}; segmentation confidence {s["score"]:.2f}.')
        if name == 'image_statistics':
            s = self.record['stats']
            return (f'Mean brightness {s["brightness"]:.1f}/255, RMS contrast {s["contrast"]:.1f}, '
                    f'mean RGB ({s["mean_r"]:.0f}, {s["mean_g"]:.0f}, {s["mean_b"]:.0f}), '
                    f'mean saturation {s["saturation"]:.2f}, sharpness (Laplacian variance) {s["sharpness"]:.1f}.')
        if name == 'retrieve_similar_cases':
            r = self.record['retrieval']
            query = 'the segmented lesion crop' if r['order'] == 'crop' else 'the full image'
            lines = [f'Query: {query}. Most similar reference cases:']
            for i, n in enumerate(r['neighbours'], 1):
                lines.append(f'{i}. {DIAGNOSIS_NAMES.get(n["diagnosis"], n["diagnosis"])} ({LABEL_NAMES[n["label"]]}), similarity {n["similarity"]:.3f}')
            n_cancer = sum(n['label'] for n in r['neighbours'])
            lines.append(f'Summary: {n_cancer} of {len(r["neighbours"])} retrieved cases are cancer.')
            return '\n'.join(lines)
        return f'Unknown tool "{name}". Available tools: {", ".join(self.names)}.'


def image_statistics(image, work_side):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float64)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    hsv = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2HSV)
    sharp = cv2.Laplacian(working_gray(image, work_side).astype(np.float64) * 255.0, cv2.CV_64F).var()
    return {
        'brightness': float(gray.mean()), 'contrast': float(gray.std()),
        'mean_r': float(rgb[..., 0].mean()), 'mean_g': float(rgb[..., 1].mean()), 'mean_b': float(rgb[..., 2].mean()),
        'saturation': float(hsv[..., 1].mean() / 255.0), 'sharpness': float(sharp),
    }


def run_single_shot(reasoner, image, seed):
    text = reasoner.generate([('user', SINGLE_SHOT_PROMPT)], image, seed)
    label, confidence = parse_final(text)
    return {'label': label, 'confidence': confidence, 'attempts': 0, 'valid_calls': 0, 'tools': [], 'transcript': [text]}


def run_react(reasoner, image, toolbox, max_steps, seed):
    turns = [('user', AGENT_PROMPT.replace('{tools}', toolbox.describe()))]
    result = {'label': None, 'confidence': None, 'attempts': 0, 'valid_calls': 0, 'tools': [], 'transcript': []}
    for step in range(max_steps):
        text = reasoner.generate(turns, image, seed + step, stop_strings=['Observation:'])
        text = text.split('Observation:')[0].strip()
        result['transcript'].append(text)
        parsed = parse_step(text, toolbox.names)
        turns.append(('assistant', text or '...'))
        if parsed['type'] == 'final' and parsed['label'] is not None:
            result['label'], result['confidence'] = parsed['label'], parsed['confidence']
            return result
        result['attempts'] += 1
        if parsed['type'] == 'tool' and parsed['valid']:
            result['valid_calls'] += 1
            result['tools'].append(parsed['name'])
            observation = toolbox(parsed['name'])
        elif parsed['type'] == 'tool':
            observation = toolbox(parsed['name'])
        elif parsed['type'] == 'final':
            observation = 'The final answer could not be parsed. Use exactly this format: ' + ANSWER_FORMAT
        else:
            observation = 'No valid Action or Final Answer was found. Call one tool with "Action: <tool name>" or answer with ' + ANSWER_FORMAT
        turns.append(('user', 'Observation: ' + observation))
    turns[-1] = (turns[-1][0], turns[-1][1] + '\nThe step limit has been reached. Give your final answer now in exactly this format:\n' + ANSWER_FORMAT)
    text = reasoner.generate(turns, image, seed + max_steps)
    result['transcript'].append(text)
    result['label'], result['confidence'] = parse_final(text)
    return result
