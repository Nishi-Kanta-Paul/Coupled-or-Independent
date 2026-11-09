import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .utils import (log, ensure_dir, save_json, load_json, append_jsonl, read_jsonl, save_mask, load_mask, case_key,
                    model_tag, stable_hash, resolve_device, resolve_dtype, free_memory)
from .datasets import load_pad, load_hiba, patient_split, dataset_summary
from .phrases import CONTROLS, DIRECTIONS, DIRECTION_TARGET, phrase_pool, payload_for, contradicting_direction, check_phrase_pools
from .perturbations import (channel_conditions, agent_conditions, apply_condition, condition_family, mask_bbox, crop_box,
                            make_view, effective_height, render_at_height, overlap_fraction)
from .retrieval import RetrievalIndex, majority_label, class_fraction, precision_at_k
from .gate import gate_statistics, fit_standardizer, q_values, select_threshold
from .agent import ToolBox, run_single_shot, run_react, image_statistics
from .metrics import modal_answer, verbal_confidence, capture_margin, directional_alignment, modality_gap


VIEWS = ('full', 'crop')
ARMS = ('single', 'full', 'crop', 'full-noret', 'crop-noret')
QUERY_ROLES = ('test', 'external')


class Workspace:
    def __init__(self, root):
        self.root = ensure_dir(root)
        self.splits = self.root / 'splits.csv'
        self.cases = self.root / 'cases.csv'
        self.summary = self.root / 'dataset_summary.json'
        self.masks = self.root / 'masks'
        self.segments = self.root / 'segments.csv'
        self.embeddings = ensure_dir(self.root / 'embeddings')
        self.channel = ensure_dir(self.root / 'channel')
        self.tools = self.root / 'tool_outputs.jsonl'
        self.gate = ensure_dir(self.root / 'gate')
        self.agent = ensure_dir(self.root / 'agent')
        self.metrics = ensure_dir(self.root / 'metrics')
        self.report = ensure_dir(self.root / 'report')

    def mask_path(self, site, img_id, condition):
        return self.masks / site / img_id / f'{condition}.npz'


def open_image(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def read_csv_safe(path):
    try:
        return pd.read_csv(path, dtype={'img_id': str, 'patient_id': str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load_cases(ws):
    return pd.read_csv(ws.cases, dtype={'img_id': str, 'patient_id': str, 'lesion_id': str})


def role_cases(cases, roles):
    return cases[cases['role'].isin(roles)].drop_duplicates(['site', 'img_id']).reset_index(drop=True)


def needed_conditions(cases, args):
    needed = {}
    for row in cases.to_dict('records'):
        key = (row['site'], row['img_id'])
        conds = ['clean'] if row['role'] in ('reference', 'validation') else channel_conditions(args.blur_severity, args.jpeg_severity)
        if key in needed:
            case, existing, roles = needed[key]
            needed[key] = (case, existing + [c for c in conds if c not in existing], roles | {row['role']})
        else:
            needed[key] = (row, list(conds), {row['role']})
    return needed


def load_segments(ws):
    frame = pd.read_csv(ws.segments, dtype={'img_id': str, 'patient_id': str})
    return {(r['site'], r['img_id'], r['condition']): r for r in frame.to_dict('records')}


def row_bbox(row):
    if row['bbox_x0'] < 0:
        return None
    return tuple(int(row[k]) for k in ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'))


def row_crop(row):
    return tuple(int(row[k]) for k in ('crop_x0', 'crop_y0', 'crop_x1', 'crop_y1'))


def perturbed_image(image, condition, case, segments, args):
    if condition == 'clean':
        return image
    clean = segments[(case['site'], case['img_id'], 'clean')]
    return apply_condition(image, condition, case, row_bbox(clean), args)[0]


def stage_split(args, ws):
    check_phrase_pools()
    pad = load_pad(args.data_path)
    split = patient_split(pad, args.split_fractions, args.seed, not args.no_stratify)
    split.to_csv(ws.splits, index=False)
    summary = {'PAD-UFES-20': dataset_summary(split)}
    if args.smoke_image:
        one = split[split['img_id'] == args.smoke_image]
        if one.empty:
            raise ValueError(f'Image {args.smoke_image} is not in the PAD-UFES-20 metadata')
        cases = pd.concat([one.assign(role=role) for role in ('reference', 'validation', 'test')], ignore_index=True)
        log(f'Executability check on a single image {args.smoke_image}: it serves as reference pool, validation set and test set')
    else:
        cases = split.assign(role=split['split'])
    if args.hiba_path and not args.smoke_image:
        hiba = load_hiba(args.hiba_path)
        summary['HIBA'] = dataset_summary(hiba)
        cases = pd.concat([cases, hiba.assign(role='external', split='external')], ignore_index=True)
    cases.to_csv(ws.cases, index=False)
    save_json(summary, ws.summary)
    for name, part in summary.items():
        log(f'{name}: {part["n_images"]} images, {part["n_patients"]} patients, cancer prevalence {part["cancer_prevalence"]:.3f}')
        for split_name, counts in part.get('splits', {}).items():
            log(f'  {split_name}: {counts}')


def segment_row(case, condition, image, mask, score, meta, clean_mask, args):
    width, height = image.size
    bbox = mask_bbox(mask)
    box = crop_box(bbox, (width, height), args.crop_margin)
    side = box[3] - box[1]
    row = {
        'site': case['site'], 'img_id': case['img_id'], 'patient_id': case['patient_id'], 'label': int(case['label']),
        'condition': condition, 'family': condition_family(condition), 'width': width, 'height': height,
        'score': score, 'area_fraction': float(mask.mean()),
        'bbox_x0': bbox[0] if bbox else -1, 'bbox_y0': bbox[1] if bbox else -1,
        'bbox_x1': bbox[2] if bbox else -1, 'bbox_y1': bbox[3] if bbox else -1,
        'crop_x0': box[0], 'crop_y0': box[1], 'crop_x1': box[2], 'crop_y1': box[3],
        'payload': meta.get('payload', ''), 'overlay_text': meta.get('overlay_text', ''),
        'font_size': meta.get('font_size', np.nan), 'payload_height_px': meta.get('payload_height_px', np.nan),
        'colour': str(meta.get('colour', '')), 'background': str(meta.get('background', '')),
        'margin_strip': meta.get('margin_strip', ''), 'fits_outside_crop': meta.get('fits_outside_crop', np.nan),
        'blur_sigma': meta.get('sigma', np.nan), 'jpeg_quality': meta.get('quality', np.nan),
    }
    if 'text_box' in meta:
        tb = meta['text_box']
        row['text_box'] = ','.join(str(int(v)) for v in tb)
        row['payload_in_crop'] = overlap_fraction(tb, box)
        row['eff_height_full'] = effective_height(meta['payload_height_px'], height, args.encoder_resolution)
        row['eff_height_crop'] = effective_height(meta['payload_height_px'], side, args.encoder_resolution)
    if clean_mask is not None and clean_mask.shape == mask.shape:
        union = np.logical_or(clean_mask, mask).sum()
        row['iou_clean'] = float(np.logical_and(clean_mask, mask).sum() / union) if union else 1.0
    return row


def stage_segment(args, ws):
    from models import LesionSegmenter
    cases = load_cases(ws)
    rows = {} if args.overwrite or not ws.segments.exists() else load_segments(ws)
    needed = needed_conditions(cases, args)
    pending = [(k, v) for k, v in needed.items() if any((k[0], k[1], c) not in rows for c in v[1])]
    if not pending:
        log('segment: all masks present')
        return
    segmenter = LesionSegmenter(args.sam_model, resolve_device(args.device))
    for step, ((site, img_id), (case, conds, _)) in enumerate(tqdm(pending, desc='segment')):
        image = open_image(case['path'])
        if (site, img_id, 'clean') not in rows:
            mask, score = segmenter(image)
            save_mask(mask, ws.mask_path(site, img_id, 'clean'))
            rows[(site, img_id, 'clean')] = segment_row(case, 'clean', image, mask, score, {}, None, args)
        clean_mask = load_mask(ws.mask_path(site, img_id, 'clean'))
        clean_bbox = row_bbox(rows[(site, img_id, 'clean')])
        for cond in conds:
            if (site, img_id, cond) in rows:
                continue
            perturbed, meta = apply_condition(image, cond, case, clean_bbox, args)
            mask, score = segmenter(perturbed)
            save_mask(mask, ws.mask_path(site, img_id, cond))
            rows[(site, img_id, cond)] = segment_row(case, cond, perturbed, mask, score, meta, clean_mask, args)
        if step % 50 == 0:
            pd.DataFrame(list(rows.values())).to_csv(ws.segments, index=False)
    pd.DataFrame(list(rows.values())).to_csv(ws.segments, index=False)
    del segmenter
    free_memory()


def stage_encode(args, ws):
    from models import build_encoder
    cases = load_cases(ws)
    segments = load_segments(ws)
    needed = needed_conditions(cases, args)
    device = resolve_device(args.device)
    for name in args.channel_encoders:
        out = ws.embeddings / f'{name}.npz'
        if out.exists() and not args.overwrite:
            log(f'encode: {name} embeddings present')
            continue
        encoder = build_encoder(name, device)
        keys, vectors, sweep_rows = [], [], []
        for (site, img_id), (case, conds, roles) in tqdm(list(needed.items()), desc=f'encode {name}'):
            image = open_image(case['path'])
            clean = segments[(site, img_id, 'clean')]
            batch, batch_keys = [], []
            for cond in conds:
                row = segments[(site, img_id, cond)]
                perturbed = perturbed_image(image, cond, case, segments, args)
                for view in VIEWS:
                    batch.append(make_view(perturbed, view, row_crop(row)))
                    batch_keys.append(case_key(site, img_id, cond, view))
            if 'test' in roles:
                direction = contradicting_direction(case['label'])
                payload = payload_for(img_id, direction, args.seed, args.payload_index)
                for height in args.render_heights:
                    rendered, meta = render_at_height(image, payload, row_bbox(clean), row_crop(clean), height, encoder.resolution, args)
                    batch.append(make_view(rendered, 'crop', row_crop(clean)))
                    batch_keys.append(case_key(site, img_id, f'sweep-{height:g}-{direction}', 'crop'))
                    sweep_rows.append({'site': site, 'img_id': img_id, 'target_height': height, 'achieved_height': meta['effective_height'],
                                       'font_size': meta['font_size'], 'n_lines': meta['n_lines']})
            vectors.append(encoder.encode_images(batch, args.encode_batch_size))
            keys.extend(batch_keys)
        texts = phrase_pool()
        np.savez(out, keys=np.array(keys), vectors=np.concatenate(vectors).astype(np.float32), texts=np.array(texts),
                 text_vectors=encoder.encode_texts(texts).astype(np.float32), resolution=np.array(encoder.resolution))
        pd.DataFrame(sweep_rows).to_csv(ws.embeddings / f'{name}_sweep.csv', index=False)
        del encoder
        free_memory()


class EmbeddingStore:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.lookup = {str(k): i for i, k in enumerate(data['keys'])}
        self.vectors = data['vectors']
        self.texts = [str(t) for t in data['texts']]
        self.text_lookup = {t: i for i, t in enumerate(self.texts)}
        self.text_vectors = data['text_vectors']
        self.resolution = int(data['resolution'])

    def get(self, site, img_id, condition, view):
        return self.vectors[self.lookup[case_key(site, img_id, condition, view)]]

    def text(self, phrase):
        return self.text_vectors[self.text_lookup[phrase]]


def build_index(store, cases, view):
    ref = role_cases(cases, ['reference'])
    embeddings = np.stack([store.get(r['site'], r['img_id'], 'clean', view) for r in ref.to_dict('records')])
    return RetrievalIndex(embeddings, ref['label'], ref['diagnosis'], ref['img_id'])


def gap_vector(store, cases, view):
    ref = role_cases(cases, ['reference'])
    images = np.stack([store.get(r['site'], r['img_id'], 'clean', view) for r in ref.to_dict('records')])
    return modality_gap(images, store.text_vectors)


def capture_conditions(label, args):
    d = contradicting_direction(label)
    return {
        'injection': f'inj-lesion-low-{d}', 'injection-high': f'inj-lesion-high-{d}', 'margin': f'inj-margin-low-{d}',
        'neutral': f'neutral-lesion-low-{d}', 'blur': f'blur-{args.blur_severity}', 'jpeg': f'jpeg-{args.jpeg_severity}',
    }


def stage_channel(args, ws):
    cases = load_cases(ws)
    queries = role_cases(cases, QUERY_ROLES)
    validation = role_cases(cases, ['validation'])
    selection, capture_rows, sweep_rows, gaps = {}, [], [], {}
    for name in args.channel_encoders:
        store = EmbeddingStore(ws.embeddings / f'{name}.npz')
        selection[name] = {}
        for view in VIEWS:
            index = build_index(store, cases, view)
            val_queries = np.stack([store.get(r['site'], r['img_id'], 'clean', view) for r in validation.to_dict('records')])
            selection[name][view] = precision_at_k(index, val_queries, validation['label'], args.knn)
            mu = gap_vector(store, cases, view)
            gaps[f'{name}|{view}'] = float(np.linalg.norm(mu))
            for case in queries.to_dict('records'):
                clean = store.get(case['site'], case['img_id'], 'clean', view)
                payload = payload_for(case['img_id'], contradicting_direction(case['label']), args.seed, args.payload_index)
                controls = [store.text(c) for c in CONTROLS[payload]]
                for family, cond in capture_conditions(case['label'], args).items():
                    delta = store.get(case['site'], case['img_id'], cond, view) - clean
                    margin, da_payload, da_control = capture_margin(delta, store.text(payload), controls, mu)
                    capture_rows.append({'encoder': name, 'view': view, 'family': family, 'condition': cond, 'site': case['site'],
                                         'img_id': case['img_id'], 'patient_id': case['patient_id'], 'label': case['label'],
                                         'delta_cap': float(margin[0]), 'da_payload': float(da_payload[0]),
                                         'da_control': float(da_control[0]), 'delta_norm': float(np.linalg.norm(delta))})
        mu_crop = gap_vector(store, cases, 'crop')
        sweep_meta = read_csv_safe(ws.embeddings / f'{name}_sweep.csv')
        achieved = {(str(r['img_id']), float(r['target_height'])): r['achieved_height'] for r in sweep_meta.to_dict('records')}
        for case in role_cases(cases, ['test']).to_dict('records'):
            direction = contradicting_direction(case['label'])
            payload = payload_for(case['img_id'], direction, args.seed, args.payload_index)
            clean = store.get(case['site'], case['img_id'], 'clean', 'crop')
            for height in args.render_heights:
                delta = store.get(case['site'], case['img_id'], f'sweep-{height:g}-{direction}', 'crop') - clean
                da_payload = float(directional_alignment(delta, store.text(payload), mu_crop)[0])
                da_controls = [float(directional_alignment(delta, store.text(c), mu_crop)[0]) for c in CONTROLS[payload]]
                sweep_rows.append({'encoder': name, 'site': case['site'], 'img_id': case['img_id'], 'patient_id': case['patient_id'],
                                   'target_height': height, 'effective_height': achieved.get((case['img_id'], float(height)), np.nan),
                                   'da_payload': da_payload, 'da_control': float(np.mean(da_controls)),
                                   'da_control_sd': float(np.std(da_controls))})
    precision = {name: float(np.mean(list(v.values()))) for name, v in selection.items()}
    selected = max(precision, key=precision.get) if args.agent_encoder == 'auto' else args.agent_encoder
    log(f'channel: validation precision@{args.knn} {precision}; agent encoder {selected}')
    save_json({'precision_by_view': selection, 'precision_mean': precision, 'selected': selected, 'gap_norms': gaps}, ws.channel / 'encoder_selection.json')
    pd.DataFrame(capture_rows).to_csv(ws.channel / 'capture.csv', index=False)
    pd.DataFrame(sweep_rows).to_csv(ws.channel / 'render_sweep.csv', index=False)
    store = EmbeddingStore(ws.embeddings / f'{selected}.npz')
    shift_rows = []
    for order in VIEWS:
        index = build_index(store, cases, order)
        for case in queries.to_dict('records'):
            clean = index.neighbours(store.get(case['site'], case['img_id'], 'clean', order), args.knn)
            for direction in DIRECTIONS:
                target = DIRECTION_TARGET[direction]
                injected = index.neighbours(store.get(case['site'], case['img_id'], f'inj-lesion-low-{direction}', order), args.knn)
                shift_rows.append({'order': order, 'direction': direction, 'site': case['site'], 'img_id': case['img_id'],
                                   'patient_id': case['patient_id'], 'label': case['label'], 'eligible': int(case['label'] != target),
                                   'target_fraction_clean': class_fraction(clean, target), 'target_fraction_injected': class_fraction(injected, target),
                                   'shift': class_fraction(injected, target) - class_fraction(clean, target)})
    pd.DataFrame(shift_rows).to_csv(ws.channel / 'retrieval_shift.csv', index=False)


FACTORIAL_SCALES = ('small', 'large')


def factorial_heights(args, seg_frame, cases):
    if args.factorial_heights:
        small, large = (float(h) for h in args.factorial_heights)
        return small, large
    test = role_cases(cases, ['test'])
    contradicting = seg_frame['condition'] == ['inj-lesion-low-' + contradicting_direction(y) for y in seg_frame['label']]
    inj = seg_frame[contradicting & (seg_frame['site'] == 'PAD') & seg_frame['img_id'].isin(test['img_id'])]
    if inj.empty:
        raise ValueError('factorial: no injected test segments; run the segment stage or pass --factorial-heights')
    return float(np.nanmedian(inj['eff_height_full'])), float(np.nanmedian(inj['eff_height_crop']))


def stage_factorial(args, ws):
    from models import build_encoder
    out = ws.channel / 'factorial.csv'
    if out.exists() and not args.overwrite:
        log('factorial: outputs present')
        return
    cases = load_cases(ws)
    segments = load_segments(ws)
    seg_frame = read_csv_safe(ws.segments)
    selected = args.factorial_encoder
    if selected == 'auto':
        selected = load_json(ws.channel / 'encoder_selection.json')['selected']
    small, large = factorial_heights(args, seg_frame, cases)
    log(f'factorial: encoder {selected}; payload height at the encoder input small={small:.2f} px, large={large:.2f} px')
    store = EmbeddingStore(ws.embeddings / f'{selected}.npz')
    indexes = {view: build_index(store, cases, view) for view in VIEWS}
    gaps = {view: gap_vector(store, cases, view) for view in VIEWS}
    encoder = build_encoder(selected, resolve_device(args.device))
    rows = []
    for case in tqdm(role_cases(cases, QUERY_ROLES).to_dict('records'), desc='factorial'):
        image = open_image(case['path'])
        clean_row = segments[(case['site'], case['img_id'], 'clean')]
        bbox, crop = row_bbox(clean_row), row_crop(clean_row)
        direction = contradicting_direction(case['label'])
        target = DIRECTION_TARGET[direction]
        payload = payload_for(case['img_id'], direction, args.seed, args.payload_index)
        controls = [store.text(c) for c in CONTROLS[payload]]
        batch, specs = [], []
        for view in VIEWS:
            box = crop if view == 'crop' else (0, 0, image.size[0], image.size[1])
            for scale, height in zip(FACTORIAL_SCALES, (small, large)):
                rendered, meta = render_at_height(image, payload, bbox, box, height, encoder.resolution, args)
                batch.append(make_view(rendered, view, crop))
                specs.append((view, scale, height, meta))
        vectors = encoder.encode_images(batch, args.encode_batch_size)
        for vector, (view, scale, height, meta) in zip(vectors, specs):
            clean_vector = store.get(case['site'], case['img_id'], 'clean', view)
            clean_nb = indexes[view].neighbours(clean_vector, args.knn)
            injected_nb = indexes[view].neighbours(vector, args.knn)
            y_clean = majority_label([n['label'] for n in clean_nb], [n['similarity'] for n in clean_nb])
            y_injected = majority_label([n['label'] for n in injected_nb], [n['similarity'] for n in injected_nb])
            margin, _, _ = capture_margin(vector - clean_vector, store.text(payload), controls, gaps[view])
            rows.append({'site': case['site'], 'img_id': case['img_id'], 'patient_id': case['patient_id'], 'label': int(case['label']),
                         'direction': direction, 'encoder': selected, 'view': view, 'scale': scale, 'target_height': height,
                         'achieved_height': meta['effective_height'], 'n_lines': meta['n_lines'],
                         'target_fraction_clean': class_fraction(clean_nb, target),
                         'target_fraction_injected': class_fraction(injected_nb, target),
                         'shift': class_fraction(injected_nb, target) - class_fraction(clean_nb, target),
                         'tool_wrong_clean': int(y_clean != case['label']), 'tool_wrong': int(y_injected != case['label']),
                         'delta_cap': float(margin[0])})
    pd.DataFrame(rows).to_csv(out, index=False)
    del encoder
    free_memory()


def stage_tools(args, ws):
    cases = load_cases(ws)
    segments = load_segments(ws)
    selected = load_json(ws.channel / 'encoder_selection.json')['selected']
    store = EmbeddingStore(ws.embeddings / f'{selected}.npz')
    indexes = {order: build_index(store, cases, order) for order in VIEWS}
    if args.overwrite and ws.tools.exists():
        ws.tools.unlink()
    done = {(r['site'], r['img_id'], r['condition'], r['order']) for r in read_jsonl(ws.tools)}
    for case in tqdm(role_cases(cases, QUERY_ROLES).to_dict('records'), desc='tools'):
        image = None
        for family, conds in agent_conditions(case['label'], args.blur_severity, args.jpeg_severity).items():
            for cond in conds:
                if all((case['site'], case['img_id'], cond, order) in done for order in VIEWS):
                    continue
                image = image or open_image(case['path'])
                perturbed = perturbed_image(image, cond, case, segments, args)
                row = segments[(case['site'], case['img_id'], cond)]
                bbox = row_bbox(row) or (0, 0, row['width'], row['height'])
                segment = {'bbox': list(bbox), 'area_fraction': row['area_fraction'], 'score': row['score'],
                           'width': int(row['width']), 'height': int(row['height']), 'crop': list(row_crop(row))}
                stats = image_statistics(perturbed, args.gate_work_side)
                for order in VIEWS:
                    neighbours = indexes[order].neighbours(store.get(case['site'], case['img_id'], cond, order), args.knn)
                    y_tool = majority_label([n['label'] for n in neighbours], [n['similarity'] for n in neighbours])
                    append_jsonl({'site': case['site'], 'img_id': case['img_id'], 'condition': cond, 'family': family, 'order': order,
                                  'encoder': selected, 'y_tool': y_tool, 'segment': segment, 'stats': stats,
                                  'retrieval': {'order': order, 'neighbours': neighbours}}, ws.tools)


def load_tools(ws):
    return {(r['site'], r['img_id'], r['condition'], r['order']): r for r in read_jsonl(ws.tools)}


def stage_gate(args, ws):
    from models import CraftScorer
    cases = load_cases(ws)
    segments = load_segments(ws)
    targets = []
    for case in role_cases(cases, ['validation']).to_dict('records'):
        targets.append((case, 'clean', 'validation'))
    for case in role_cases(cases, QUERY_ROLES).to_dict('records'):
        for conds in agent_conditions(case['label'], args.blur_severity, args.jpeg_severity).values():
            for cond in conds:
                targets.append((case, cond, case['role']))
    scorer = CraftScorer(resolve_device(args.device), os.path.join(os.environ.get('EASYOCR_MODULE_PATH', '.'), 'model'))
    cache, rows = {}, []
    for case, cond, role in tqdm(targets, desc='gate statistics'):
        key = (case['site'], case['img_id'], cond)
        if key not in cache:
            perturbed = perturbed_image(open_image(case['path']), cond, case, segments, args)
            stats = gate_statistics(perturbed, args)
            stats['craft'] = scorer(perturbed, args.craft_canvas)
            cache[key] = stats
        rows.append({'role': role, 'site': case['site'], 'img_id': case['img_id'], 'patient_id': case['patient_id'],
                     'label': case['label'], 'condition': cond, 'family': condition_family(cond), **cache[key]})
    del scorer
    free_memory()
    frame = pd.DataFrame(rows)
    validation = frame[frame['role'] == 'validation']
    params = fit_standardizer(validation)
    frame['q'] = q_values(frame, params)
    theta = select_threshold(frame.loc[frame['role'] == 'validation', 'q'], args.gate_target_rate)
    craft_theta = select_threshold(validation['craft'], args.gate_target_rate)
    frame.to_csv(ws.gate / 'gate_statistics.csv', index=False)
    save_json({'standardizer': params, 'theta': theta, 'craft_theta': craft_theta, 'target_rate': args.gate_target_rate,
               'n_validation': int(len(validation)),
               'validation_flag_rate_q': float(np.mean(frame.loc[frame['role'] == 'validation', 'q'] > theta)),
               'validation_flag_rate_craft': float(np.mean(validation['craft'] > craft_theta))}, ws.gate / 'gate_params.json')
    log(f'gate: theta={theta:.3f}, craft threshold={craft_theta:.3f} fitted on {len(validation)} clean validation images')


def stage_agent(args, ws):
    from models import VisionReasoner
    cases = role_cases(load_cases(ws), QUERY_ROLES)
    segments = load_segments(ws)
    tools = load_tools(ws)
    device = resolve_device(args.device)
    for model_id in args.reasoners:
        tag = model_tag(model_id)
        out = ws.agent / f'{tag}.jsonl'
        if args.overwrite and out.exists():
            out.unlink()
        done = {(r['site'], r['img_id'], r['condition'], r['arm']) for r in read_jsonl(out)}
        jobs = []
        for case in cases.to_dict('records'):
            for family, conds in agent_conditions(case['label'], args.blur_severity, args.jpeg_severity).items():
                for cond in conds:
                    arms = [a for a in args.arms if (case['site'], case['img_id'], cond, a) not in done]
                    if arms:
                        jobs.append((case, family, cond, arms))
        if not jobs:
            log(f'agent: {model_id} complete')
            continue
        reasoner = VisionReasoner(model_id, device, resolve_dtype(args.reasoner_dtype), args.max_new_tokens,
                                  args.temperature, args.top_p, args.reasoner_image_max_side)
        images = {}
        for case, family, cond, arms in tqdm(jobs, desc=f'agent {tag}'):
            if case['img_id'] not in images:
                images = {case['img_id']: open_image(case['path'])}
            prepared = reasoner.prepare_image(perturbed_image(images[case['img_id']], cond, case, segments, args))
            for arm in arms:
                start = time.time()
                decodes = []
                for d in range(args.decodes):
                    seed = stable_hash(args.seed, tag, case['site'], case['img_id'], cond, arm, d) % (2 ** 31 - 1024)
                    if arm == 'single':
                        result = run_single_shot(reasoner, prepared, seed)
                    else:
                        order = arm.split('-')[0]
                        toolbox = ToolBox(tools[(case['site'], case['img_id'], cond, order)], not arm.endswith('noret'), args.knn)
                        result = run_react(reasoner, prepared, toolbox, args.react_max_steps, seed)
                    decodes.append(result)
                labels = [r['label'] for r in decodes]
                confidences = [r['confidence'] for r in decodes]
                modal, kappa = modal_answer(labels, confidences)
                append_jsonl({'reasoner': model_id, 'site': case['site'], 'img_id': case['img_id'], 'patient_id': case['patient_id'],
                              'label': int(case['label']), 'condition': cond, 'family': family, 'arm': arm, 'modal': modal,
                              'kappa': kappa, 'conf_verb': verbal_confidence(labels, confidences, modal),
                              'seconds': time.time() - start, 'decodes': decodes}, out)
        del reasoner
        free_memory()
