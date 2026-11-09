import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .utils import log, save_json, load_json, read_jsonl, model_tag, ensure_dir, load_mask
from .bootstrap import ClusterBootstrap
from .metrics import (INVALID, CELLS, balanced_accuracy, confident_wrong, expected_calibration_error, wmean, coupling_cell, cell_shares,
                      faulty_shares, disagreement_ceiling, amplification)
from .gate import gate_decisions
from .perturbations import FAMILIES, apply_condition, effective_height
from .phrases import contradicting_direction
from .engine import load_cases, role_cases, load_tools, open_image, QUERY_ROLES, ARMS, VIEWS


SITES = ('PAD', 'HIBA')
CHANNEL_FAMILIES = ('injection', 'injection-high', 'margin', 'neutral', 'blur', 'jpeg', 'corruption')


def safe_mean(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float('nan')


def group_arrays(frame, keys, columns):
    if frame.empty:
        return {}
    out = {}
    for key, part in frame.groupby(list(keys)):
        key = key if isinstance(key, tuple) else (key,)
        out[key] = {c: part[c].to_numpy() for c in columns}
    return out


def flatten(prefix, results, rows):
    if isinstance(results, dict) and 'point' in results and 'lo' in results:
        rows.append({'statistic': prefix, **results})
        return
    for key, value in results.items():
        flatten(f'{prefix}{key}' if not prefix else f'{prefix}/{key}', value, rows)


def load_agent_frame(args, ws, cases, boot):
    rows = []
    for model_id in args.reasoners:
        for r in read_jsonl(ws.agent / f'{model_tag(model_id)}.jsonl'):
            rows.append({
                'reasoner': model_id, 'site': r['site'], 'img_id': r['img_id'], 'patient_id': r['patient_id'], 'label': int(r['label']),
                'condition': r['condition'], 'family': r['family'], 'arm': r['arm'], 'modal': int(r['modal']), 'kappa': float(r['kappa']),
                'conf_verb': np.nan if r['conf_verb'] is None else float(r['conf_verb']),
                'attempts': sum(d['attempts'] for d in r['decodes']), 'valid_calls': sum(d['valid_calls'] for d in r['decodes']),
                'invalid_decodes': sum(d['label'] is None for d in r['decodes']), 'n_decodes': len(r['decodes']),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    biopsy = cases.drop_duplicates(['site', 'img_id']).set_index(['site', 'img_id'])['biopsied']
    frame['biopsied'] = [bool(biopsy.get((s, i), False)) for s, i in zip(frame['site'], frame['img_id'])]
    frame['correct'] = (frame['modal'] == frame['label']).astype(float)
    frame['pcode'] = boot.codes(frame['patient_id'])
    return frame


def arm_values(a, w, tau, bins):
    y, modal, kappa = a['label'], a['modal'], a['kappa']
    return {
        'bal_acc': balanced_accuracy(y, modal, w),
        'cwr': wmean(confident_wrong(y, modal, kappa, tau), w),
        'kappa': wmean(kappa, w),
        'conf_verb': wmean(a['conf_verb'], w),
        'ece': expected_calibration_error(a['conf_verb'], a['correct'], w, bins),
        'invalid': wmean((modal == INVALID).astype(float), w),
    }


def class_weight_fn(case_codes, case_labels, case_mask_target, case_mask_source):
    def make(m):
        w = m[case_codes]
        def prevalence(mask):
            total = w[mask].sum()
            return np.array([w[mask & (case_labels == c)].sum() / total if total > 0 else np.nan for c in (0, 1)])
        target, source = prevalence(case_mask_target), prevalence(case_mask_source)
        ratio = np.where(source > 0, target / np.where(source > 0, source, 1.0), 0.0)
        return np.nan_to_num(ratio, nan=0.0)
    return make


def agent_statistics(groups, reasoners, site, m, args, label_weights=None):
    out, pooled = {}, {}
    for family in FAMILIES:
        for arm in ARMS:
            per = {}
            for r in reasoners:
                a = groups.get((r, site, family, arm))
                if a is None:
                    continue
                w = m[a['pcode']] * (label_weights[a['label']] if label_weights is not None else 1.0)
                per[r] = arm_values(a, w, args.tau, args.ece_bins)
            if not per:
                continue
            values = {k: safe_mean([v[k] for v in per.values()]) for k in next(iter(per.values()))}
            pooled[(family, arm)] = values
            for k, v in values.items():
                out[f'{k}|{family}|{arm}'] = v
            out[f'conf_gap|{family}|{arm}'] = values['conf_verb'] - values['kappa']
            for r, v in per.items():
                out[f'cwr|{family}|{arm}|{r}'] = v['cwr']
                out[f'bal_acc|{family}|{arm}|{r}'] = v['bal_acc']

    def eps(family, arm, functional):
        values = pooled.get((family, arm))
        if values is None:
            return float('nan')
        return values['cwr'] if functional == 'cwr' else 1.0 - values['bal_acc']

    for family in ('injection', 'corruption', 'margin'):
        for arm in ARMS[1:]:
            for functional in ('cwr', 'err'):
                out[f'faf_{functional}|{family}|{arm}'] = amplification(
                    eps(family, 'single', functional), eps('clean', 'single', functional), eps(family, arm, functional), eps('clean', arm, functional))
    return out


def coupling_frame(agent, tools, gate_frame):
    base = agent[agent['arm'].isin(VIEWS)].copy()
    base['order'] = base['arm']
    noret = agent[agent['arm'].isin([f'{v}-noret' for v in VIEWS])][['reasoner', 'site', 'img_id', 'condition', 'arm', 'modal', 'kappa']].copy()
    noret['order'] = noret['arm'].str.split('-').str[0]
    noret = noret.rename(columns={'modal': 'modal_noret', 'kappa': 'kappa_noret'}).drop(columns='arm')
    frame = base.merge(noret, on=['reasoner', 'site', 'img_id', 'condition', 'order'], how='left')
    frame['requery_missing'] = frame['modal_noret'].isna()
    frame['modal_noret'] = frame['modal_noret'].fillna(INVALID).astype(int)
    frame['kappa_noret'] = frame['kappa_noret'].fillna(0.0)
    frame['y_tool'] = [tools[(s, i, c, o)]['y_tool'] for s, i, c, o in zip(frame['site'], frame['img_id'], frame['condition'], frame['order'])]
    stats = gate_frame[gate_frame['role'] != 'validation'].drop_duplicates(['site', 'img_id', 'condition']).set_index(['site', 'img_id', 'condition'])
    keys = list(zip(frame['site'], frame['img_id'], frame['condition']))
    frame['q'] = [stats['q'].get(k, np.nan) for k in keys]
    frame['craft'] = [stats['craft'].get(k, np.nan) for k in keys]
    frame['cell'] = [coupling_cell(y, t, r) for y, t, r in zip(frame['label'], frame['y_tool'], frame['modal'])]
    frame['cell_noret'] = [('NA' if miss else coupling_cell(y, t, r)) for y, t, r, miss in
                           zip(frame['label'], frame['y_tool'], frame['modal_noret'], frame['requery_missing'])]
    return frame


def gate_values(a, w, theta, tau):
    decision = gate_decisions(a['y_tool'], a['modal'], a['kappa'], np.nan_to_num(a['q'], nan=-np.inf),
                              a['modal_noret'], a['kappa_noret'], theta, tau)
    values = {}
    for cell in ('C2', 'C3', 'C4a'):
        m = a['cell'] == cell
        values[f'flag|{cell}'] = wmean(decision['flag'][m].astype(float), w[m])
        values[f'flag_disagree|{cell}'] = wmean(decision['flag_disagree'][m].astype(float), w[m])
        values[f'flag_q_only|{cell}'] = wmean((decision['flag_q'] & ~decision['flag_disagree'])[m].astype(float), w[m])
    faulty = a['cell'] != 'C1'
    values['recall'] = wmean(decision['flag'][faulty].astype(float), w[faulty])
    values['flag_rate'] = wmean(decision['flag'].astype(float), w)
    values['coverage'] = wmean(decision['answered'].astype(float), w)
    answered = decision['answered']
    cw = ((decision['final'] != a['label']) & (decision['final_kappa'] >= tau - 1e-9)).astype(float)
    values['gated_cwr'] = wmean(cw[answered], w[answered])
    values['ungated_cwr'] = wmean(confident_wrong(a['label'], a['modal'], a['kappa'], tau), w)
    return values


def coupling_statistics(groups, reasoners, site, m, gate_params, args, label_weights=None):
    out = {}
    for family in FAMILIES:
        c4a, baseline = {}, {}
        for order in VIEWS:
            shares, noret, gates, recall_d, error_t, error_r = [], [], [], [], [], []
            for r in reasoners:
                a = groups.get((r, site, family, order))
                if a is None:
                    continue
                w = m[a['pcode']] * (label_weights[a['label']] if label_weights is not None else 1.0)
                shares.append(cell_shares(a['cell'], w))
                error_t.append(wmean((a['y_tool'] != a['label']).astype(float), w))
                error_r.append(wmean((a['modal'] != a['label']).astype(float), w))
                valid = a['cell_noret'] != 'NA'
                if valid.any():
                    noret.append(cell_shares(a['cell_noret'][valid], w[valid]))
                faulty = a['cell'] != 'C1'
                recall_d.append(wmean((a['y_tool'] != a['modal'])[faulty].astype(float), w[faulty]))
                if gate_params is not None:
                    gates.append(gate_values(a, w, gate_params['theta'], args.tau))
            if not shares:
                continue
            pooled = {c: safe_mean([s[c] for s in shares]) for c in shares[0]}
            for c, v in pooled.items():
                out[f'cell|{family}|{order}|{c}'] = v
            for c, v in faulty_shares(pooled).items():
                out[f'faulty_share|{family}|{order}|{c}'] = v
            out[f'faulty|{family}|{order}'] = 1.0 - pooled['C1']
            out[f'ceiling|{family}|{order}'] = disagreement_ceiling(pooled)
            out[f'recall_disagreement|{family}|{order}'] = safe_mean(recall_d)
            e_t, e_r = safe_mean(error_t), safe_mean(error_r)
            out[f'tool_error|{family}|{order}'] = e_t
            out[f'reasoner_error|{family}|{order}'] = e_r
            out[f'independence_c4a|{family}|{order}'] = e_t * e_r
            out[f'excess_c4a|{family}|{order}'] = pooled['C4a'] - e_t * e_r
            den = pooled['C3'] + pooled['C4a']
            out[f'recall_wrong_final|{family}|{order}'] = pooled['C3'] / den if den > 0 else float('nan')
            den = pooled['C2'] + pooled['C3']
            out[f'flag_precision_wrong_final|{family}|{order}'] = pooled['C3'] / den if den > 0 else float('nan')
            c4a[order] = pooled['C4a']
            baseline[order] = e_t * e_r
            if noret:
                pooled_noret = {c: safe_mean([s[c] for s in noret]) for c in noret[0]}
                for c, v in pooled_noret.items():
                    out[f'cell_noret|{family}|{order}|{c}'] = v
                c4a[f'{order}-noret'] = pooled_noret['C4a']
            if gates:
                for k in gates[0]:
                    out[f'gate|{family}|{order}|{k}'] = safe_mean([g[k] for g in gates])
        if 'crop' in c4a and 'full' in c4a:
            out[f'order_gap|{family}'] = c4a['crop'] - c4a['full']
            out[f'order_gap_from_tool_error|{family}'] = baseline['crop'] - baseline['full']
            out[f'order_gap_excess_dependence|{family}'] = (c4a['crop'] - baseline['crop']) - (c4a['full'] - baseline['full'])
        if 'crop-noret' in c4a and 'full-noret' in c4a:
            out[f'order_gap_noret|{family}'] = c4a['crop-noret'] - c4a['full-noret']
        if 'crop' in c4a and 'crop-noret' in c4a:
            out[f'c4a_drop_without_retrieval|{family}|crop'] = c4a['crop'] - c4a['crop-noret']
        if 'full' in c4a and 'full-noret' in c4a:
            out[f'c4a_drop_without_retrieval|{family}|full'] = c4a['full'] - c4a['full-noret']
    return out


PAIR_COLUMNS = ('cell', 'modal', 'y_tool', 'modal_noret', 'requery_missing')


def paired_frame(coupling):
    keys = ['reasoner', 'site', 'img_id', 'patient_id', 'label', 'condition', 'family', 'pcode']
    cols = list(PAIR_COLUMNS)
    full = coupling[coupling['order'] == 'full'][keys + cols].rename(columns={c: f'{c}_full' for c in cols})
    crop = coupling[coupling['order'] == 'crop'][keys + cols].rename(columns={c: f'{c}_crop' for c in cols})
    return full.merge(crop, on=keys, how='inner')


def transition_statistics(pgroups, reasoners, site, m):
    out = {}
    for family in FAMILIES:
        joints, flips_order, flips_noise, tool_flips = [], [], [], []
        for r in reasoners:
            a = pgroups.get((r, site, family))
            if a is None:
                continue
            w = m[a['pcode']]
            joints.append({(s, d): wmean(((a['cell_full'] == s) & (a['cell_crop'] == d)).astype(float), w) for s in CELLS for d in CELLS})
            flips_order.append(wmean((a['modal_full'] != a['modal_crop']).astype(float), w))
            tool_flips.append(wmean((a['y_tool_full'] != a['y_tool_crop']).astype(float), w))
            both = ~(a['requery_missing_full'].astype(bool) | a['requery_missing_crop'].astype(bool))
            if both.any():
                flips_noise.append(wmean((a['modal_noret_full'] != a['modal_noret_crop'])[both].astype(float), w[both]))
        if not joints:
            continue
        joint = {k: safe_mean([j[k] for j in joints]) for k in joints[0]}
        for (s, d), v in joint.items():
            out[f'trans|{family}|{s}->{d}'] = v

        def row(s):
            return sum(joint[(s, d)] for d in CELLS)

        def col(d):
            return sum(joint[(s, d)] for s in CELLS)

        def ratio(num, den):
            return num / den if den > 0 else float('nan')

        detectable_full = row('C2') + row('C3')
        out[f'c4a_crop_given_c3_full|{family}'] = ratio(joint[('C3', 'C4a')], row('C3'))
        out[f'c3_full_given_c4a_crop|{family}'] = ratio(joint[('C3', 'C4a')], col('C4a'))
        out[f'hidden_given_detectable|{family}'] = ratio(joint[('C2', 'C4a')] + joint[('C3', 'C4a')], detectable_full)
        out[f'detectable_given_hidden|{family}'] = ratio(joint[('C4a', 'C2')] + joint[('C4a', 'C3')], row('C4a'))
        out[f'detectable_to_hidden|{family}'] = joint[('C2', 'C4a')] + joint[('C3', 'C4a')]
        out[f'hidden_to_detectable|{family}'] = joint[('C4a', 'C2')] + joint[('C4a', 'C3')]
        out[f'into_c4a|{family}'] = col('C4a') - joint[('C4a', 'C4a')]
        out[f'out_of_c4a|{family}'] = row('C4a') - joint[('C4a', 'C4a')]
        out[f'reasoner_flip_between_orders|{family}'] = safe_mean(flips_order)
        out[f'reasoner_flip_noise|{family}'] = safe_mean(flips_noise)
        out[f'tool_flip_between_orders|{family}'] = safe_mean(tool_flips)
    return out


FACTORIAL_METRICS = ('shift', 'tool_wrong', 'delta_cap')


def factorial_statistics(fac, m, site):
    out = {}
    part = fac[fac['site'] == site]
    if part.empty:
        return out
    cells = {}
    for (view, scale), p in part.groupby(['view', 'scale']):
        w = m[p['pcode'].to_numpy()]
        for metric in FACTORIAL_METRICS:
            cells[(metric, view, scale)] = wmean(p[metric].to_numpy(), w)
            out[f'{metric}|{view}|{scale}'] = cells[(metric, view, scale)]
    for metric in FACTORIAL_METRICS:
        fs, fl = cells.get((metric, 'full', 'small'), np.nan), cells.get((metric, 'full', 'large'), np.nan)
        cs, cl = cells.get((metric, 'crop', 'small'), np.nan), cells.get((metric, 'crop', 'large'), np.nan)
        scale_main, view_main, natural = 0.5 * ((fl - fs) + (cl - cs)), 0.5 * ((cs - fs) + (cl - fl)), cl - fs
        out[f'{metric}|scale_effect|full'] = fl - fs
        out[f'{metric}|scale_effect|crop'] = cl - cs
        out[f'{metric}|view_effect|small'] = cs - fs
        out[f'{metric}|view_effect|large'] = cl - fl
        out[f'{metric}|scale_main'] = scale_main
        out[f'{metric}|view_main'] = view_main
        out[f'{metric}|interaction'] = (cl - cs) - (fl - fs)
        out[f'{metric}|natural_contrast'] = natural
        out[f'{metric}|scale_share'] = scale_main / natural if np.isfinite(natural) and abs(natural) > 1e-9 else float('nan')
    return out


def payload_scale_summary(inj, null_edge):
    def q(series, level):
        values = series.dropna().to_numpy(dtype=float)
        return float(np.percentile(values, level)) if values.size else float('nan')
    out = {'n': int(len(inj)), 'payload_height_px_mean': float(inj['payload_height_px'].mean())}
    for order in ('full', 'crop'):
        col = inj[f'eff_height_{order}']
        out[f'eff_height_{order}_mean'] = float(col.mean())
        out[f'eff_height_{order}_median'] = q(col, 50)
        out[f'eff_height_{order}_q25'] = q(col, 25)
        out[f'eff_height_{order}_q75'] = q(col, 75)
        out[f'eff_height_{order}_frac_at_or_above_null_edge'] = float((col.dropna() >= null_edge).mean()) if np.isfinite(null_edge) and col.notna().any() else float('nan')
    out['null_edge_px'] = null_edge
    return out


def risk_coverage(groups, reasoners, site, family, order, targets, args):
    rows = [groups.get((r, site, family, order)) for r in reasoners]
    rows = [a for a in rows if a is not None]
    if not rows:
        return {}
    qs = np.concatenate([np.nan_to_num(a['q'], nan=-np.inf) for a in rows])
    grid = np.unique(np.concatenate([qs[np.isfinite(qs)], [-np.inf, np.inf]]))
    points = []
    for theta in grid:
        values = [gate_values(a, np.ones(len(a['label'])), theta, args.tau) for a in rows]
        points.append((safe_mean([v['coverage'] for v in values]), safe_mean([v['gated_cwr'] for v in values]), float(theta)))
    ungated = safe_mean([wmean(confident_wrong(a['label'], a['modal'], a['kappa'], args.tau), np.ones(len(a['label']))) for a in rows])
    points.append((1.0, ungated, float('nan')))
    curve = sorted({(round(c, 6), r, t) for c, r, t in points if np.isfinite(c) and np.isfinite(r)}, key=lambda p: p[0])
    coverage = np.array([p[0] for p in curve])
    risk = np.array([p[1] for p in curve])
    at = {f'{t:.2f}': float(np.interp(t, coverage, risk)) if len(coverage) and coverage.min() <= t <= coverage.max() else float('nan') for t in targets}
    return {'curve': [{'coverage': c, 'cwr': r, 'theta': t} for c, r, t in curve], 'at_coverage': at}


def channel_statistics(capture, shift, sweep, m, site):
    out = {}
    cap = capture[capture['site'] == site]
    for (encoder, view, family), part in cap.groupby(['encoder', 'view', 'family']):
        w = m[part['pcode'].to_numpy()]
        out[f'delta_cap|{encoder}|{view}|{family}'] = wmean(part['delta_cap'].to_numpy(), w)
        out[f'delta_norm|{encoder}|{view}|{family}'] = wmean(part['delta_norm'].to_numpy(), w)
        out[f'da_payload|{encoder}|{view}|{family}'] = wmean(part['da_payload'].to_numpy(), w)
    corr = cap[cap['family'].isin(['blur', 'jpeg'])]
    for (encoder, view), part in corr.groupby(['encoder', 'view']):
        w = m[part['pcode'].to_numpy()]
        out[f'delta_cap|{encoder}|{view}|corruption'] = wmean(part['delta_cap'].to_numpy(), w)
        out[f'delta_norm|{encoder}|{view}|corruption'] = wmean(part['delta_norm'].to_numpy(), w)
    sh = shift[(shift['site'] == site) & (shift['eligible'] == 1)]
    means = {}
    for (order, direction), part in sh.groupby(['order', 'direction']):
        w = m[part['pcode'].to_numpy()]
        means[(order, direction)] = wmean(part['shift'].to_numpy(), w)
        out[f'retrieval_shift|{order}|{direction}'] = means[(order, direction)]
        out[f'retrieval_level_clean|{order}|{direction}'] = wmean(part['target_fraction_clean'].to_numpy(), w)
        out[f'retrieval_level_injected|{order}|{direction}'] = wmean(part['target_fraction_injected'].to_numpy(), w)
    for direction in ('noncancer', 'cancer'):
        if ('crop', direction) in means and ('full', direction) in means:
            out[f'retrieval_order_difference|{direction}'] = means[('crop', direction)] - means[('full', direction)]
    if all(k in out for k in ('retrieval_order_difference|noncancer', 'retrieval_order_difference|cancer')):
        out['retrieval_direction_order_interaction'] = out['retrieval_order_difference|noncancer'] - out['retrieval_order_difference|cancer']
    sw = sweep[sweep['site'] == site]
    for (encoder, height), part in sw.groupby(['encoder', 'target_height']):
        w = m[part['pcode'].to_numpy()]
        out[f'sweep_da_payload|{encoder}|{height:g}'] = wmean(part['da_payload'].to_numpy(), w)
        out[f'sweep_da_control|{encoder}|{height:g}'] = wmean(part['da_control'].to_numpy(), w)
    return out


def sweep_summary(sweep, site, low=8.0, high=20.0, plateau_frac=0.9):
    sw = sweep[sweep['site'] == site]
    if sw.empty:
        return {}
    curves = sw.groupby(['encoder', 'target_height']).agg(da_payload=('da_payload', 'mean'), da_control=('da_control', 'mean'),
                                                            effective_height=('effective_height', 'mean')).reset_index()
    null_band = max(float(part['da_control'].mean() + part['da_control'].std(ddof=0)) for _, part in sw.groupby('encoder'))
    ratios = []
    for encoder, part in curves.groupby('encoder'):
        part = part.set_index('target_height')
        if low in part.index and high in part.index and part.loc[high, 'da_payload'] != 0:
            ratios.append(part.loc[low, 'da_payload'] / part.loc[high, 'da_payload'])
    mean_curve = curves.groupby('target_height')['da_payload'].mean().sort_index()
    above = mean_curve[mean_curve > null_band]
    start = float(above.index.min()) if not above.empty else float('nan')
    end = float('nan')
    if np.isfinite(start) and mean_curve.iloc[-1] > 0:
        plateau = mean_curve[(mean_curve.index >= start) & (mean_curve >= plateau_frac * mean_curve.iloc[-1])]
        end = float(plateau.index.min()) if not plateau.empty else float('nan')
    return {
        'null_band': null_band,
        'ratio_low_to_high': safe_mean(ratios),
        'low_height': low, 'high_height': high,
        'lowest_height_above_null_band': start,
        'transition': [start, end],
        'plateau_fraction': plateau_frac,
        'curves': curves.to_dict('records'),
    }


def provenance_scale(args, segments, cases):
    rows = segments[segments['img_id'] == args.provenance_image]
    if rows.empty:
        return {}
    injected = rows[rows['family'] == 'injection']
    if not injected.empty:
        r = injected.iloc[0]
        return {'img_id': args.provenance_image, 'payload_height_px': float(r['payload_height_px']), 'eff_height_full': float(r['eff_height_full']),
                'eff_height_crop': float(r['eff_height_crop']), 'source': 'perturbed-image mask'}
    clean = rows[rows['condition'] == 'clean']
    if clean.empty:
        return {}
    clean = clean.iloc[0]
    case = cases[cases['img_id'] == args.provenance_image].iloc[0].to_dict()
    bbox = None if clean['bbox_x0'] < 0 else tuple(int(clean[k]) for k in ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'))
    _, meta = apply_condition(open_image(case['path']), f'inj-lesion-low-{contradicting_direction(case["label"])}', case, bbox, args)
    side = int(clean['crop_y1']) - int(clean['crop_y0'])
    return {'img_id': args.provenance_image, 'payload_height_px': float(meta['payload_height_px']),
            'eff_height_full': effective_height(meta['payload_height_px'], int(clean['height']), args.encoder_resolution),
            'eff_height_crop': effective_height(meta['payload_height_px'], side, args.encoder_resolution), 'source': 'clean-image mask'}


def mask_iou_summary(ws, segments, cases, args):
    out, review = {}, []
    rng = np.random.default_rng(args.seed)
    seg = segments[segments['condition'] != 'clean']
    for site in SITES:
        query = role_cases(cases, ['test'] if site == 'PAD' else ['external'])
        query = query[query['site'] == site]
        if query.empty:
            continue
        chosen = query.sample(n=min(len(query), args.mask_iou_subset), random_state=int(rng.integers(0, 2 ** 31 - 1)))
        part = seg[(seg['site'] == site) & seg['img_id'].isin(chosen['img_id'])]
        out[site] = {'n_images': int(len(chosen)), 'by_family': part.groupby('family')['iou_clean'].mean().to_dict(),
                     'by_condition': part.groupby('condition')['iou_clean'].mean().to_dict()}
        review.extend((site, r) for r in chosen.to_dict('records'))
    return out, review


def export_mask_review(ws, review, args):
    folder = ensure_dir(ws.metrics / 'mask_review')
    rows = []
    for site, case in review:
        image = open_image(case['path'])
        mask = load_mask(ws.mask_path(site, case['img_id'], 'clean'))
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        alpha = Image.fromarray((mask * 90).astype(np.uint8))
        overlay.paste(Image.new('RGBA', image.size, (0, 255, 0, 255)), (0, 0), alpha)
        composite = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        composite.thumbnail((768, 768))
        path = folder / f'{site}_{case["img_id"]}.jpg'
        composite.save(path, quality=90)
        rows.append({'site': site, 'img_id': case['img_id'], 'overlay': str(path), 'reviewer_1_mask_covers_lesion': '', 'reviewer_2_mask_covers_lesion': ''})
    pd.DataFrame(rows).to_csv(folder / 'review_sheet.csv', index=False)


def stage_evaluate(args, ws):
    cases = load_cases(ws)
    queries = role_cases(cases, QUERY_ROLES)
    boot = ClusterBootstrap(queries['patient_id'], args.n_boot, args.seed, args.ci)
    results = {'config': {k: v for k, v in vars(args).items()}, 'dataset': load_json(ws.summary),
               'n_patients_bootstrap': boot.n_clusters, 'n_boot': args.n_boot}
    segments = pd.read_csv(ws.segments, dtype={'img_id': str, 'patient_id': str})
    capture = pd.read_csv(ws.channel / 'capture.csv', dtype={'img_id': str, 'patient_id': str})
    shift = pd.read_csv(ws.channel / 'retrieval_shift.csv', dtype={'img_id': str, 'patient_id': str})
    sweep = pd.read_csv(ws.channel / 'render_sweep.csv', dtype={'img_id': str, 'patient_id': str})
    for frame in (capture, shift, sweep):
        frame['pcode'] = boot.codes(frame['patient_id'])
    results['encoder_selection'] = load_json(ws.channel / 'encoder_selection.json')
    log('evaluate: channel-level statistics')
    results['channel'] = {site: boot.run(lambda m, s=site: channel_statistics(capture, shift, sweep, m, s))
                          for site in SITES if (capture['site'] == site).any()}
    results['render_scale'] = {site: sweep_summary(sweep, site) for site in SITES if (sweep['site'] == site).any()}
    query_ids = set(zip(queries['site'], queries['img_id']))
    contradicting = segments['condition'] == ['inj-lesion-low-' + contradicting_direction(y) for y in segments['label']]
    inj = segments[contradicting & [k in query_ids for k in zip(segments['site'], segments['img_id'])]]
    null_edge = (results['render_scale'].get('PAD') or {}).get('lowest_height_above_null_band', float('nan'))
    results['payload_scale'] = {}
    for site, part in inj.groupby('site'):
        results['payload_scale'][site] = payload_scale_summary(part, null_edge)
        results['payload_scale'][site]['per_image'] = (
            part[['img_id', 'condition', 'payload_height_px', 'eff_height_full', 'eff_height_crop', 'font_size']].to_dict('records')
            if len(part) <= 5 else [])
    factorial_path = ws.channel / 'factorial.csv'
    if factorial_path.exists():
        fac = pd.read_csv(factorial_path, dtype={'img_id': str, 'patient_id': str})
        fac['pcode'] = boot.codes(fac['patient_id'])
        log('evaluate: scale x cropping factorial')
        results['factorial'] = {site: boot.run(lambda m, s=site: factorial_statistics(fac, m, s))
                                for site in SITES if (fac['site'] == site).any()}
        results['factorial_heights'] = {'small': float(fac['target_height'][fac['scale'] == 'small'].iloc[0]),
                                        'large': float(fac['target_height'][fac['scale'] == 'large'].iloc[0]),
                                        'achieved_mean': {f'{v}|{s}': float(x) for (v, s), x in
                                                          fac.groupby(['view', 'scale'])['achieved_height'].mean().items()},
                                        'encoder': str(fac['encoder'].iloc[0])}
    else:
        log('evaluate: no factorial outputs (run --stage factorial); factorial statistics skipped')
    results['payload_scale']['provenance'] = provenance_scale(args, segments, cases)
    results['mask_iou'], review = mask_iou_summary(ws, segments, cases, args)
    export_mask_review(ws, review, args)
    agent = load_agent_frame(args, ws, cases, boot)
    if agent.empty:
        log('evaluate: no agent outputs found; agent-level statistics skipped')
        save_json(results, ws.metrics / 'results.json')
        return results
    reasoners = [r for r in args.reasoners if (agent['reasoner'] == r).any()]
    tools = load_tools(ws)
    gate_frame = pd.read_csv(ws.gate / 'gate_statistics.csv', dtype={'img_id': str, 'patient_id': str})
    gate_params = load_json(ws.gate / 'gate_params.json')
    results['gate_params'] = gate_params
    agent_cols = ['label', 'modal', 'kappa', 'conf_verb', 'correct', 'pcode']
    groups = group_arrays(agent, ['reasoner', 'site', 'family', 'arm'], agent_cols)
    coupling = coupling_frame(agent, tools, gate_frame)
    coupling_cols = agent_cols + ['y_tool', 'cell', 'cell_noret', 'modal_noret', 'kappa_noret', 'q', 'craft']
    cgroups = group_arrays(coupling, ['reasoner', 'site', 'family', 'order'], coupling_cols)
    coupling.drop(columns=['pcode']).to_csv(ws.metrics / 'coupling_cases.csv', index=False)
    log('evaluate: agent-level statistics')
    results['agent'] = {site: boot.run(lambda m, s=site: agent_statistics(groups, reasoners, s, m, args))
                        for site in SITES if (agent['site'] == site).any()}
    log('evaluate: coupling and gate statistics')
    results['coupling'] = {site: boot.run(lambda m, s=site: coupling_statistics(cgroups, reasoners, s, m, gate_params, args))
                           for site in SITES if (coupling['site'] == site).any()}
    paired = paired_frame(coupling)
    pcols = ['label', 'pcode'] + [f'{c}_{o}' for c in PAIR_COLUMNS for o in VIEWS]
    pgroups = group_arrays(paired, ['reasoner', 'site', 'family'], pcols)
    paired.drop(columns=['pcode']).to_csv(ws.metrics / 'transition_cases.csv', index=False)
    log('evaluate: paired transitions between pipeline orders')
    results['transitions'] = {site: boot.run(lambda m, s=site: transition_statistics(pgroups, reasoners, s, m))
                              for site in SITES if (paired['site'] == site).any()}
    results['transition_counts'] = paired.groupby(['site', 'reasoner', 'family', 'cell_full', 'cell_crop']).size().rename('n').reset_index().to_dict('records')
    results['cell_counts'] = coupling.groupby(['site', 'reasoner', 'family', 'order', 'cell']).size().rename('n').reset_index().to_dict('records')
    results['risk_coverage'] = {site: {order: risk_coverage(cgroups, reasoners, site, 'injection', order, args.coverage_targets, args) for order in VIEWS}
                                for site in SITES if (coupling['site'] == site).any()}
    gate_queries = gate_frame[gate_frame['role'] != 'validation'].drop_duplicates(['site', 'img_id', 'condition']).copy()
    gate_queries['pcode'] = boot.codes(gate_queries['patient_id'])

    def craft_fn(m, site):
        out = {}
        for family, part in gate_queries[gate_queries['site'] == site].groupby('family'):
            out[f'craft_flag_rate|{family}'] = wmean((part['craft'] > gate_params['craft_theta']).astype(float).to_numpy(), m[part['pcode'].to_numpy()])
            out[f'q_flag_rate|{family}'] = wmean((part['q'] > gate_params['theta']).astype(float).to_numpy(), m[part['pcode'].to_numpy()])
        for order in VIEWS:
            vals = []
            for r in reasoners:
                a = cgroups.get((r, site, 'injection', order))
                if a is None:
                    continue
                sel = a['cell'] == 'C4a'
                vals.append(wmean((a['craft'][sel] > gate_params['craft_theta']).astype(float), m[a['pcode'][sel]]))
            out[f'craft_flag_rate_c4a|{order}'] = safe_mean(vals)
        return out

    results['detectors'] = {site: boot.run(lambda m, s=site: craft_fn(m, s)) for site in SITES if (gate_queries['site'] == site).any()}
    agentic = agent[agent['arm'] != 'single']
    results['tool_call_validity'] = {r: float(part['valid_calls'].sum() / part['attempts'].sum()) if part['attempts'].sum() else float('nan')
                                     for r, part in agentic.groupby('reasoner')}
    results['invalid_decode_rate'] = {r: float(part['invalid_decodes'].sum() / part['n_decodes'].sum()) for r, part in agent.groupby('reasoner')}
    pad_cases = queries[queries['site'] == 'PAD'].reset_index(drop=True)
    if not pad_cases.empty:
        case_codes = boot.codes(pad_cases['patient_id'])
        case_labels = pad_cases['label'].to_numpy()
        biopsied = pad_cases['biopsied'].astype(bool).to_numpy()
        weights = class_weight_fn(case_codes, case_labels, np.ones(len(pad_cases), dtype=bool), biopsied)
        bgroups = group_arrays(agent[agent['biopsied']], ['reasoner', 'site', 'family', 'arm'], agent_cols)
        log('evaluate: biopsy-confirmed subset')
        results['biopsy_subset'] = {
            'n_images': int(biopsied.sum()),
            'raw': boot.run(lambda m: {k: v for k, v in agent_statistics(bgroups, reasoners, 'PAD', m, args).items() if k.startswith('faf_')}),
            'class_standardised': boot.run(lambda m: {k: v for k, v in agent_statistics(bgroups, reasoners, 'PAD', m, args, weights(m)).items() if k.startswith('faf_')}),
        }
    if (agent['site'] == 'HIBA').any() and (agent['site'] == 'PAD').any():
        all_codes = boot.codes(queries['patient_id'])
        all_labels = queries['label'].to_numpy()
        is_pad = (queries['site'] == 'PAD').to_numpy()
        ext_weights = class_weight_fn(all_codes, all_labels, is_pad, ~is_pad)

        def external_fn(m):
            out = {}
            internal_a = agent_statistics(groups, reasoners, 'PAD', m, args)
            internal_c = coupling_statistics(cgroups, reasoners, 'PAD', m, None, args)
            for tag, lw in (('raw', None), ('standardised', ext_weights(m))):
                external_a = agent_statistics(groups, reasoners, 'HIBA', m, args, lw)
                external_c = coupling_statistics(cgroups, reasoners, 'HIBA', m, None, args, lw)
                quantities = {
                    'bal_acc_clean_crop': ('bal_acc|clean|crop', internal_a, external_a),
                    'c4a_crop': ('cell|injection|crop|C4a', internal_c, external_c),
                    'c4a_full': ('cell|injection|full|C4a', internal_c, external_c),
                    'faf_cwr_crop': ('faf_cwr|injection|crop', internal_a, external_a),
                    'order_gap': ('order_gap|injection', internal_c, external_c),
                }
                for name, (key, ia, ea) in quantities.items():
                    i_val, e_val = ia.get(key, float('nan')), ea.get(key, float('nan'))
                    out[f'{tag}|{name}|internal'] = i_val
                    out[f'{tag}|{name}|external'] = e_val
                    out[f'{tag}|{name}|difference'] = e_val - i_val
            return out

        log('evaluate: external-site contrasts')
        results['external'] = boot.run(external_fn)
    rows = []
    for section in ('channel', 'factorial', 'agent', 'coupling', 'transitions', 'detectors', 'biopsy_subset', 'external'):
        if section in results:
            flatten(section, {k: v for k, v in results[section].items() if k != 'n_images'}, rows)
    pd.DataFrame(rows).to_csv(ws.metrics / 'summary.csv', index=False)
    save_json(results, ws.metrics / 'results.json')
    log(f'evaluate: wrote {len(rows)} statistics to {ws.metrics / "summary.csv"}')
    return results
