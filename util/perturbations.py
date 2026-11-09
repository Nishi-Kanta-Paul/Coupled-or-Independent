import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib

from .phrases import NEUTRAL, DIRECTIONS, payload_for, contradicting_direction


DEFAULT_FONT = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
BLUR_SIGMAS = (1, 2, 3, 4, 6)
JPEG_QUALITIES = (25, 18, 15, 10, 7)
SUPERSAMPLE = 8
FAMILIES = ('clean', 'injection', 'corruption', 'margin')


def parse_condition(name):
    if name == 'clean':
        return {'kind': 'clean'}
    parts = name.split('-')
    if parts[0] in ('blur', 'jpeg'):
        return {'kind': parts[0], 'severity': int(parts[1])}
    kind = {'inj': 'injection', 'neutral': 'neutral', 'sweep': 'sweep'}[parts[0]]
    if kind == 'sweep':
        return {'kind': 'sweep', 'height': float(parts[1]), 'direction': parts[2]}
    return {'kind': kind, 'placement': parts[1], 'saliency': parts[2], 'direction': parts[3]}


def channel_conditions(blur_severity, jpeg_severity):
    names = ['clean']
    for placement, saliency in (('lesion', 'low'), ('lesion', 'high'), ('margin', 'low')):
        for direction in DIRECTIONS:
            names.append(f'inj-{placement}-{saliency}-{direction}')
    for direction in DIRECTIONS:
        names.append(f'neutral-lesion-low-{direction}')
    names += [f'blur-{blur_severity}', f'jpeg-{jpeg_severity}']
    return names


def agent_conditions(label, blur_severity, jpeg_severity):
    direction = contradicting_direction(label)
    return {
        'clean': ['clean'],
        'injection': [f'inj-lesion-low-{direction}'],
        'corruption': [f'blur-{blur_severity}', f'jpeg-{jpeg_severity}'],
        'margin': [f'inj-margin-low-{direction}'],
    }


def condition_family(name):
    spec = parse_condition(name)
    if spec['kind'] == 'clean':
        return 'clean'
    if spec['kind'] in ('blur', 'jpeg'):
        return 'corruption'
    if spec['kind'] == 'injection' and spec['placement'] == 'lesion' and spec['saliency'] == 'low':
        return 'injection'
    if spec['kind'] == 'injection' and spec['placement'] == 'margin' and spec['saliency'] == 'low':
        return 'margin'
    return name


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def crop_box(bbox, size, margin):
    width, height = size
    if bbox is None:
        return (0, 0, width, height)
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = margin * max(x1 - x0, y1 - y0)
    side = max(1.0, min(side, width, height))
    left = min(max(cx - side / 2.0, 0.0), width - side)
    top = min(max(cy - side / 2.0, 0.0), height - side)
    return (int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side)))


def make_view(image, view, box):
    if view == 'full':
        return image
    return image.crop(tuple(box))


def view_height(image_size, view, box):
    if view == 'full':
        return image_size[1]
    return box[3] - box[1]


def effective_height(height_px, height_of_view, resolution):
    return float(height_px) * resolution / float(height_of_view)


def load_font(path, size):
    return ImageFont.truetype(path or DEFAULT_FONT, max(int(round(size * SUPERSAMPLE)), 1))


def ink_size(text, font):
    left, top, right, bottom = font.getbbox(text)
    return (right - left) / SUPERSAMPLE, (bottom - top) / SUPERSAMPLE


def fit_font_to_width(text, max_width, font_path, min_size=1.0 / SUPERSAMPLE, max_size=2000.0):
    lo, hi = int(round(min_size * SUPERSAMPLE)), int(round(max_size * SUPERSAMPLE))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ink_size(text, load_font(font_path, mid / SUPERSAMPLE))[0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return lo / SUPERSAMPLE


def fit_font_to_height(text, target_height, font_path, min_size=1.0 / SUPERSAMPLE, max_size=2000.0):
    best, best_err = min_size, float('inf')
    lo, hi = int(round(min_size * SUPERSAMPLE)), int(round(max_size * SUPERSAMPLE))
    while lo <= hi:
        mid = (lo + hi) // 2
        h = ink_size(text, load_font(font_path, mid / SUPERSAMPLE))[1]
        err = abs(h - target_height)
        if err < best_err:
            best, best_err = mid / SUPERSAMPLE, err
        if h < target_height:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def wrap_lines(text, font, max_width):
    words = text.split(' ')
    lines, current = [], ''
    for word in words:
        trial = word if not current else current + ' ' + word
        if current and ink_size(trial, font)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def block_size(lines, font, line_gap):
    line_height = ink_size(' '.join(lines), font)[1]
    width = max(ink_size(line, font)[0] for line in lines)
    height = line_height * len(lines) + line_gap * line_height * (len(lines) - 1)
    return width, height, line_height


def text_colour(region, saliency, low_delta):
    mean = region.reshape(-1, 3).mean(axis=0) if region.size else np.array([127.5, 127.5, 127.5])
    luminance = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]
    sign = 1.0 if luminance < 127.5 else -1.0
    if saliency == 'high':
        colour = np.array([255.0, 255.0, 255.0]) if sign > 0 else np.array([0.0, 0.0, 0.0])
    else:
        colour = np.clip(mean + sign * low_delta * 255.0, 0, 255)
    return tuple(int(round(c)) for c in colour), tuple(float(m) for m in mean)


def draw_block(image, lines, font, center, saliency, low_delta, line_gap):
    width, height, line_height = block_size(lines, font, line_gap)
    img_w, img_h = image.size
    x0 = int(np.floor(min(max(center[0] - width / 2.0, 0), max(img_w - width, 0))))
    y0 = int(np.floor(min(max(center[1] - height / 2.0, 0), max(img_h - height, 0))))
    box = (x0, y0, min(int(np.ceil(x0 + width)), img_w), min(int(np.ceil(y0 + height)), img_h))
    region = np.asarray(image.convert('RGB'))[box[1]:box[3], box[0]:box[2]].astype(np.float32)
    colour, background = text_colour(region, saliency, low_delta)
    pad = 2
    tile_w, tile_h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    mask = Image.new('L', (tile_w * SUPERSAMPLE, tile_h * SUPERSAMPLE), 0)
    draw = ImageDraw.Draw(mask)
    step = (line_height + line_gap * line_height) * SUPERSAMPLE
    for i, line in enumerate(lines):
        left, top, right, _ = font.getbbox(line)
        lx = pad * SUPERSAMPLE + (width * SUPERSAMPLE - (right - left)) / 2.0 - left
        ly = pad * SUPERSAMPLE + i * step - top
        draw.text((lx, ly), line, font=font, fill=255)
    alpha = mask.resize((tile_w, tile_h), Image.BOX)
    out = image.convert('RGB').copy()
    out.paste(Image.new('RGB', (tile_w, tile_h), colour), (box[0] - pad, box[1] - pad), alpha)
    return out, {'text_box': box, 'colour': colour, 'background': background, 'n_lines': len(lines), 'line_height_px': line_height}


def lesion_reference(bbox, size):
    if bbox is not None:
        return bbox
    width, height = size
    return (int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75))


def margin_center(block_w, block_h, size, crop):
    width, height = size
    strips = {
        'top': (crop[1], width, (width / 2.0, crop[1] / 2.0)),
        'bottom': (height - crop[3], width, (width / 2.0, (crop[3] + height) / 2.0)),
        'left': (width, crop[0], (crop[0] / 2.0, height / 2.0)),
        'right': (width, width - crop[2], ((crop[2] + width) / 2.0, height / 2.0)),
    }
    best, best_slack = None, -float('inf')
    for name, (avail_h, avail_w, center) in strips.items():
        if name in ('top', 'bottom'):
            slack = min(avail_h - block_h, avail_w - block_w)
        else:
            slack = min(height - block_h, avail_w - block_w)
        if slack > best_slack:
            best, best_slack = (name, center), slack
    return best[1], best[0], best_slack >= 0


def overlap_fraction(box, region):
    ix0, iy0 = max(box[0], region[0]), max(box[1], region[1])
    ix1, iy1 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def render_payload(image, text, size_text, bbox, placement, saliency, args):
    reference = lesion_reference(bbox, image.size)
    lesion_width = reference[2] - reference[0]
    font_size = fit_font_to_width(size_text, args.payload_width_frac * lesion_width, args.font_path)
    font = load_font(args.font_path, font_size)
    lines = [text]
    width, height, line_height = block_size(lines, font, args.line_gap)
    clean_crop = crop_box(bbox, image.size, args.crop_margin)
    meta = {'font_size': font_size, 'payload_height_px': ink_size(size_text, font)[1], 'placement': placement}
    if placement == 'lesion':
        center = ((reference[0] + reference[2]) / 2.0, (reference[1] + reference[3]) / 2.0)
        meta['margin_strip'] = ''
        meta['fits_outside_crop'] = False
    else:
        center, strip, fits = margin_center(width, height, image.size, clean_crop)
        meta['margin_strip'] = strip
        meta['fits_outside_crop'] = bool(fits)
    out, draw_meta = draw_block(image, lines, font, center, saliency, args.low_contrast_delta, args.line_gap)
    meta.update(draw_meta)
    meta['payload_in_clean_crop'] = overlap_fraction(draw_meta['text_box'], clean_crop)
    return out, meta


def render_at_height(image, text, bbox, box, target_height, resolution, args):
    side = box[3] - box[1]
    height_px = target_height * side / float(resolution)
    font_size = fit_font_to_height(text, height_px, args.font_path)
    font = load_font(args.font_path, font_size)
    lines = wrap_lines(text, font, 0.95 * (box[2] - box[0]))
    reference = lesion_reference(bbox, image.size)
    center = ((reference[0] + reference[2]) / 2.0, (reference[1] + reference[3]) / 2.0)
    out, draw_meta = draw_block(image, lines, font, center, 'low', args.low_contrast_delta, args.line_gap)
    achieved = ink_size(text, font)[1]
    meta = {'font_size': font_size, 'payload_height_px': achieved, 'effective_height': effective_height(achieved, side, resolution)}
    meta.update(draw_meta)
    return out, meta


def gaussian_blur(image, severity, relative, reference=224):
    sigma = BLUR_SIGMAS[severity - 1]
    if relative:
        sigma = sigma * min(image.size) / float(reference)
    arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    out = ndimage.gaussian_filter(arr, sigma=(sigma, sigma, 0), mode='nearest', truncate=4.0)
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255)), {'sigma': float(sigma)}


def jpeg_compress(image, severity):
    quality = JPEG_QUALITIES[severity - 1]
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB'), {'quality': quality}


def apply_condition(image, name, case, bbox, args):
    spec = parse_condition(name)
    meta = {'condition': name, 'kind': spec['kind'], 'payload': '', 'overlay_text': ''}
    if spec['kind'] == 'clean':
        return image.convert('RGB').copy(), meta
    if spec['kind'] == 'blur':
        out, extra = gaussian_blur(image, spec['severity'], args.corruption_scale == 'relative')
        meta.update(extra)
        meta['payload'] = payload_for(case['img_id'], contradicting_direction(case['label']), args.seed, args.payload_index)
        return out, meta
    if spec['kind'] == 'jpeg':
        out, extra = jpeg_compress(image, spec['severity'])
        meta.update(extra)
        meta['payload'] = payload_for(case['img_id'], contradicting_direction(case['label']), args.seed, args.payload_index)
        return out, meta
    payload = payload_for(case['img_id'], spec['direction'], args.seed, args.payload_index)
    text = payload if spec['kind'] == 'injection' else NEUTRAL[payload]
    out, extra = render_payload(image, text, payload, bbox, spec['placement'], spec['saliency'], args)
    meta.update(extra)
    meta['payload'] = payload
    meta['overlay_text'] = text
    meta['direction'] = spec['direction']
    return out, meta
