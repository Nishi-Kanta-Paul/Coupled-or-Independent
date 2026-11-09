import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter
from matplotlib.transforms import blended_transform_factory

from .utils import log, load_json
from .metrics import CELLS


INK = '#4a4a4a'
RED = '#b2261e'
BLUE = '#1f6fb2'
PURPLE = '#7b1fa2'
ORANGE = '#c0622f'
GREEN_BAR = '#5f9e60'
RED_BAR = '#c0473f'
GATE_BLUE = '#3a80bd'
CELL_COLOURS = {'C1': '#4a8c4a', 'C2': '#3a80bd', 'C3': '#c27645', 'C4a': '#bf3f38', 'C4b': '#6d2e46', 'CX': '#a8a8a8'}
ENCODER_NAMES = {'biomedclip': 'BiomedCLIP', 'pubmedclip': 'PubMedCLIP', 'clip': 'CLIP ViT-B/32'}
ENCODER_TICKS = {'biomedclip': 'BiomedCLIP', 'pubmedclip': 'PubMedCLIP', 'clip': 'CLIP\nViT-B/32'}
ENCODER_STYLES = {'clip': (BLUE, 'o', '-'), 'biomedclip': (PURPLE, 's', '-'), 'pubmedclip': (ORANGE, '^', '--')}
ARM_NAMES = {'single': 'single-shot', 'full': 'full-image', 'crop': 'crop-first', 'full-noret': 'full-image, no retrieval', 'crop-noret': 'crop-first, no retrieval'}
STYLE = {
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 13.5, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 12.5,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5, 'xtick.major.size': 5, 'ytick.major.size': 5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'grid.color': '#dcdcdc', 'grid.linewidth': 1.0,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'pdf.fonttype': 42,
}


def stat(results, *path):
    node = results
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def point(s):
    return float('nan') if s is None or s.get('point') is None else float(s['point'])


def interval(s):
    if s is None or s.get('lo') is None or s.get('hi') is None:
        return float('nan'), float('nan')
    return float(s['lo']), float(s['hi'])


def fmt(s, digits=2, ci=True):
    p = point(s)
    if not np.isfinite(p):
        return 'n/a'
    lo, hi = interval(s)
    if not ci or not np.isfinite(lo):
        return f'{p:.{digits}f}'
    return f'{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'


def finite(*values):
    return all(v is not None and np.isfinite(v) for v in values)


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf')
    fig.savefig(out / f'{name}.png', dpi=300)
    plt.close(fig)


def no_data(ax, text='not estimable on this run'):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center', style='italic', color=INK)


def figure_directional(results, args, out):
    encoders = [e for e in ('clip', 'biomedclip', 'pubmedclip') if e in args.channel_encoders]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.axvline(0, color='#a0a0a0', lw=3.2, zorder=1)
    ax.axvline(0, color='#707070', lw=0.8, zorder=1)
    values = [0.0]
    for i, encoder in enumerate(encoders):
        for offset, family, colour in ((-0.17, 'injection', RED), (0.17, 'corruption', BLUE)):
            s = stat(results, 'channel', 'PAD', f'delta_cap|{encoder}|{args.capture_view}|{family}')
            p = point(s)
            if not finite(p):
                continue
            lo, hi = interval(s)
            xerr = [[p - lo], [hi - p]] if finite(lo, hi) else None
            ax.errorbar(p, i + offset, xerr=xerr, fmt='o', ms=9, color=colour, ecolor=colour, elinewidth=2.4, capsize=5,
                        capthick=2.4, zorder=3, label=family if i == 0 else None)
            values += [v for v in (p, lo, hi) if finite(v)]
    ax.set_yticks(range(len(encoders)))
    ax.set_yticklabels([ENCODER_TICKS[e] for e in encoders])
    ax.set_ylim(len(encoders) - 0.5, -0.5)
    span = max(values) - min(values) or 0.1
    ax.set_xlim(min(values) - 0.2 * span, max(values) + 0.2 * span)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'directional capture $\Delta_{\mathrm{cap}}$')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.4, columnspacing=1.6)
    save(fig, out, 'fig_r1_directional')


def figure_renderscale(results, args, out):
    summary = stat(results, 'render_scale', 'PAD')
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    if not summary:
        no_data(ax)
        save(fig, out, 'fig_r2_renderscale')
        return
    curves = pd.DataFrame(summary['curves'])
    heights = sorted(float(h) for h in curves['target_height'].unique())
    band = summary['null_band']
    prov = stat(results, 'payload_scale', 'provenance') or {}
    marks = [(prov.get('eff_height_full'), 'full-image'), (prov.get('eff_height_crop'), 'crop-first')]
    marks = [(x, label) for x, label in marks if finite(x)]
    xs = heights + [x for x, _ in marks]
    xmin, xmax = min(xs) / 1.18, max(xs) * 1.1
    ymin = min(0.0, float(curves['da_payload'].min()) - 0.03)
    ymax = max(float(curves['da_payload'].max()), band) + 0.06
    start, end = (summary.get('transition') or [float('nan'), float('nan')])[:2]
    if finite(start, end) and end > start:
        ax.axvspan(start, end, color='#f6e8e6', zorder=0)
    ax.axhspan(ymin, band, color='#efefef', zorder=0)
    ax.text(xmax / 1.03, band - 0.01 * (ymax - ymin), 'null band', ha='right', va='top', style='italic')
    for encoder in ('clip', 'biomedclip', 'pubmedclip'):
        part = curves[curves['encoder'] == encoder].sort_values('target_height')
        if part.empty:
            continue
        colour, marker, ls = ENCODER_STYLES[encoder]
        ax.plot(part['target_height'], part['da_payload'], color=colour, marker=marker, ls=ls, lw=2.2, ms=8, label=ENCODER_NAMES[encoder], zorder=3)
    label_trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, label in marks:
        ax.axvline(x, color=INK, ls=':', lw=1.8, zorder=2)
        ax.text(x * 1.06, 0.30, label, ha='left', va='bottom', transform=label_trans,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7, 'pad': 1.0})
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(heights))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{h:g}' for h in heights]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('payload height at encoder input (px)')
    ax.set_ylabel('alignment DA')
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    save(fig, out, 'fig_r2_renderscale')


def figure_retrieval(results, out):
    k = results.get('config', {}).get('knn', 5)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    width = 0.32
    for i, order in enumerate(('full', 'crop')):
        clean = point(stat(results, 'channel', 'PAD', f'retrieval_level_clean|{order}|noncancer'))
        injected = point(stat(results, 'channel', 'PAD', f'retrieval_level_injected|{order}|noncancer'))
        shift = stat(results, 'channel', 'PAD', f'retrieval_shift|{order}|noncancer')
        if finite(clean):
            ax.bar(i - width / 2, clean, width, color=GREEN_BAR, label='clean' if i == 0 else None, zorder=2)
        if finite(injected):
            ax.bar(i + width / 2, injected, width, color=RED_BAR, label='injected' if i == 0 else None, zorder=2)
            p, (lo, hi) = point(shift), interval(shift)
            top = injected
            if finite(p, lo, hi):
                ax.errorbar(i + width / 2, injected, yerr=[[p - lo], [hi - p]], color=INK, capsize=6, elinewidth=2.2, capthick=2.2, zorder=3)
                top = injected + (hi - p)
            if finite(p):
                label = f'{p:+.2f}' + (f'\n[{lo:.2f}, {hi:.2f}]' if finite(lo, hi) else '')
                ax.text(i + width / 2, min(top + 0.03, 0.97), label, ha='center', va='bottom', color=RED, linespacing=1.1)
    if not ax.patches:
        no_data(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['full-image', 'crop-first'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel(f'non-cancer fraction, top-{k}')
    ax.legend(loc='upper left')
    save(fig, out, 'fig_r3_retrieval')


def figure_coupling(results, out):
    groups = [('full', 'injection', 'lesion'), ('full', 'margin', 'margin'), ('crop', 'injection', 'lesion'), ('crop', 'margin', 'margin')]
    cells = ['C1', 'C2', 'C3', 'C4a'] + [c for c in ('C4b', 'CX') if any(
        point(stat(results, 'coupling', 'PAD', f'cell|{f}|{o}|{c}')) > 0 for o, f, _ in groups
        if finite(point(stat(results, 'coupling', 'PAD', f'cell|{f}|{o}|{c}'))))]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for x, (order, family, _) in enumerate(groups):
        bottom = 0.0
        for cell in cells:
            p = point(stat(results, 'coupling', 'PAD', f'cell|{family}|{order}|{cell}'))
            if not finite(p) or p <= 0:
                continue
            ax.bar(x, p, 0.62, bottom=bottom, color=CELL_COLOURS[cell], edgecolor='white', linewidth=1.6, zorder=2)
            if p >= 0.15:
                ax.text(x, bottom + p / 2, f'{p:.2f}', ha='center', va='center', color='white', fontsize=13)
            bottom += p
    for cell in cells:
        ax.bar(0, 0, color=CELL_COLOURS[cell], label=cell)
    if not any(patch.get_height() > 0 for patch in ax.patches):
        no_data(ax)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[2] for g in groups])
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for centre, label in ((0.5, 'full-image'), (2.5, 'crop-first')):
        ax.text(centre, -0.15, label, transform=trans, ha='center', va='top')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel('proportion of cases')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(cells), handlelength=1.3, columnspacing=1.2)
    save(fig, out, 'fig_r4_coupling')


def figure_faf(results, out):
    rows = [('crop', 'injection'), ('full', 'injection'), ('crop', 'corruption'), ('full', 'corruption')]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    values = [1.0]
    for i, (arm, family) in enumerate(rows):
        s = stat(results, 'agent', 'PAD', f'faf_cwr|{family}|{arm}')
        p, (lo, hi) = point(s), interval(s)
        if finite(p):
            values += [v for v in (p, lo, hi) if finite(v)]
    span = max(values) - min(values) or 0.5
    for i, (arm, family) in enumerate(rows):
        s = stat(results, 'agent', 'PAD', f'faf_cwr|{family}|{arm}')
        p, (lo, hi) = point(s), interval(s)
        colour = RED if family == 'injection' else INK
        if not finite(p):
            ax.text(1.0 + 0.05 * span, i, 'n/a', va='center', color=colour)
            continue
        hollow = finite(lo, hi) and lo <= 1.0 <= hi
        xerr = [[p - lo], [hi - p]] if finite(lo, hi) else None
        ax.errorbar(p, i, xerr=xerr, fmt='o', ms=11, color=colour, ecolor=colour, mfc='white' if hollow else colour, mew=2.6,
                    elinewidth=2.6, capsize=6, capthick=2.6, zorder=3)
        ax.text((hi if finite(hi) else p) + 0.05 * span, i, f'{p:.2f}', va='center', color=colour)
    ax.axvline(1.0, color=INK, ls='--', lw=1.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{ARM_NAMES[arm]}, {family}' for arm, family in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(min(values) - 0.25 * span, max(values) + 0.6 * span)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('fault amplification FAF (95% CI)')
    save(fig, out, 'fig_r5_faf')


def figure_gate(results, args, out):
    order = args.gate_figure_order
    cells = ('C2', 'C3', 'C4a')
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    clean = [point(stat(results, 'coupling', 'PAD', f'gate|clean|{o}|flag_rate')) for o in ('full', 'crop')]
    clean = [c for c in clean if finite(c)]
    if clean:
        rate = float(np.mean(clean))
        ax.axhline(rate, color=INK, ls=':', lw=2.4, zorder=3, label=f'clean ({rate:.3f})')
    for i, cell in enumerate(cells):
        dis = point(stat(results, 'coupling', 'PAD', f'gate|injection|{order}|flag_disagree|{cell}'))
        qonly = point(stat(results, 'coupling', 'PAD', f'gate|injection|{order}|flag_q_only|{cell}'))
        if not finite(dis) and not finite(qonly):
            ax.text(i, 0.04, 'n/a', ha='center', va='bottom')
            continue
        dis, qonly = (dis if finite(dis) else 0.0), (qonly if finite(qonly) else 0.0)
        ax.bar(i, dis, 0.56, color=GATE_BLUE, zorder=2, label='disagreement' if i == 0 else None)
        ax.bar(i, qonly, 0.56, bottom=dis, facecolor=(0.75, 0.25, 0.22, 0.45), edgecolor=(0.75, 0.25, 0.22, 0.65), hatch='//',
               zorder=2, label='$q(x)$' if i == 0 else None)
        ax.text(i, dis + qonly + 0.01, f'{dis + qonly:.2f}', ha='center', va='bottom')
    handles, labels = ax.get_legend_handles_labels()
    if 'disagreement' not in labels:
        ax.bar(0, 0, 0.56, color=GATE_BLUE, label='disagreement')
    if '$q(x)$' not in labels:
        ax.bar(0, 0, 0.56, facecolor=(0.75, 0.25, 0.22, 0.45), edgecolor=(0.75, 0.25, 0.22, 0.65), hatch='//', label='$q(x)$')
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells)
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel('fraction flagged')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2)
    save(fig, out, 'fig_r7_gate')


def figure_provenance(results, args, ws, out):
    from .engine import load_cases, load_segments, open_image, perturbed_image, row_crop
    from .perturbations import make_view
    prov = stat(results, 'payload_scale', 'provenance') or {}
    if not prov:
        return
    cases = load_cases(ws)
    case = cases[cases['img_id'] == prov['img_id']].iloc[0].to_dict()
    segments = load_segments(ws)
    condition = f'inj-lesion-low-{"noncancer" if int(case["label"]) == 1 else "cancer"}'
    key = (case['site'], case['img_id'], condition)
    if key not in segments:
        return
    image = perturbed_image(open_image(case['path']), condition, case, segments, args)
    crop = row_crop(segments[key])
    res = args.encoder_resolution
    panels = [(image, 'perturbed image'), (make_view(image, 'full', crop).resize((res, res)), f'full-image input: {prov["eff_height_full"]:.1f} px'),
              (make_view(image, 'crop', crop).resize((res, res)), f'crop-first input: {prov["eff_height_crop"]:.1f} px')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    save(fig, out, 'fig_provenance')


LATEX_SYMBOLS = (('Delta_cap', '$\\Delta_{\\mathrm{cap}}$'), ('R*', '$R^{\\star}$'), ('->', '$\\rightarrow$'),
                 ('e_T x e_R', '$e_{\\mathrm{T}}e_{\\mathrm{R}}$'), ('e_T', '$e_{\\mathrm{T}}$'), ('e_R', '$e_{\\mathrm{R}}$'),
                 ('p3/(p3+p4a)', '$p_{3}/(p_{3}+p_{4\\mathrm{a}})$'), ('p3/(p2+p3)', '$p_{3}/(p_{2}+p_{3})$'), ('*', '$^{*}$'))


def latex_cell(text):
    text = str(text)
    slots = []
    for i, (plain, tex) in enumerate(LATEX_SYMBOLS):
        if plain in text:
            text = text.replace(plain, f'@@{i}@@')
            slots.append((f'@@{i}@@', tex))
    for char, escaped in (('\\', '\\textbackslash '), ('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_'),
                          ('{', '\\{'), ('}', '\\}'), ('$', '\\$'), ('~', '\\~{}'), ('^', '\\^{}')):
        text = text.replace(char, escaped)
    text = text.replace(' - ', ' $-$ ').replace('[-', '[$-$').replace(', -', ', $-$').replace(' -0', ' $-$0').replace(' -1', ' $-$1')
    for key, tex in slots:
        text = text.replace(key, tex)
    return text


def write_table(rows, columns, title, out, name, note=''):
    frame = pd.DataFrame(rows, columns=columns)
    with open(out / f'{name}.md', 'w') as f:
        f.write(f'### {title}\n\n')
        f.write('| ' + ' | '.join(columns) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(columns)) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join(str(v) for v in row) + ' |\n')
        if note:
            f.write(f'\n{note}\n')
    with open(out / f'{name}.tex', 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write(f'\\caption{{{latex_cell(title)}}}\n\\label{{{name}_reproduced}}\n\\footnotesize\n')
        f.write('\\begin{tabular}{@{}' + 'l' * len(columns) + '@{}}\n\\toprule\n')
        f.write(' & '.join(latex_cell(c) for c in columns) + ' \\\\\n\\midrule\n')
        for row in rows:
            f.write(' & '.join(latex_cell(v) for v in row) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        if note:
            f.write('\\vspace{2pt}\n\\begin{minipage}{\\linewidth}\\raggedright\\footnotesize ' + latex_cell(note) + '\\end{minipage}\n')
        f.write('\\end{table}\n')
    frame.to_csv(out / f'{name}.csv', index=False)


def table_main(results, out):
    rows = []
    for family, name in (('clean', 'none'), ('injection', 'injection'), ('corruption', 'corruption'), ('margin', 'margin injection')):
        for arm in ('single', 'full', 'crop', 'full-noret', 'crop-noret'):
            if stat(results, 'agent', 'PAD', f'cwr|{family}|{arm}') is None:
                continue
            faf = '--' if family == 'clean' or arm == 'single' else fmt(stat(results, 'agent', 'PAD', f'faf_cwr|{family}|{arm}'))
            faf_err = '--' if family == 'clean' or arm == 'single' else fmt(stat(results, 'agent', 'PAD', f'faf_err|{family}|{arm}'))
            rows.append([name, ARM_NAMES[arm], fmt(stat(results, 'agent', 'PAD', f'bal_acc|{family}|{arm}'), 3),
                         fmt(stat(results, 'agent', 'PAD', f'cwr|{family}|{arm}')), faf, faf_err,
                         fmt(stat(results, 'agent', 'PAD', f'ece|{family}|{arm}'), 3)])
    write_table(rows, ['Perturbation', 'Arm', 'Bal. acc.', 'CWR', 'FAF cwr', 'FAF err', 'ECE'], 'Main results (internal test, pooled over reasoners)', out, 'tab_main')


def table_cells(results, out):
    rows = []
    for order in ('full', 'crop'):
        for cell in CELLS:
            s = stat(results, 'coupling', 'PAD', f'cell|injection|{order}|{cell}')
            f = stat(results, 'coupling', 'PAD', f'faulty_share|injection|{order}|{cell}')
            rows.append([ARM_NAMES[order], cell, fmt(s), '--' if cell == 'C1' else fmt(f)])
        rows.append([ARM_NAMES[order], 'ceiling R*', fmt(stat(results, 'coupling', 'PAD', f'ceiling|injection|{order}')), '--'])
        rows.append([ARM_NAMES[order], 'recall over wrong final diagnoses p3/(p3+p4a)', fmt(stat(results, 'coupling', 'PAD', f'recall_wrong_final|injection|{order}')), '--'])
        rows.append([ARM_NAMES[order], 'flag precision for wrong final diagnoses p3/(p2+p3)', fmt(stat(results, 'coupling', 'PAD', f'flag_precision_wrong_final|injection|{order}')), '--'])
        rows.append([ARM_NAMES[order], 'tool error e_T', fmt(stat(results, 'coupling', 'PAD', f'tool_error|injection|{order}'), ci=False), '--'])
        rows.append([ARM_NAMES[order], 'reasoner error e_R', fmt(stat(results, 'coupling', 'PAD', f'reasoner_error|injection|{order}'), ci=False), '--'])
        rows.append([ARM_NAMES[order], 'independence baseline e_T x e_R', fmt(stat(results, 'coupling', 'PAD', f'independence_c4a|injection|{order}')), '--'])
        rows.append([ARM_NAMES[order], 'excess C4a (observed - e_T x e_R)', fmt(stat(results, 'coupling', 'PAD', f'excess_c4a|injection|{order}')), '--'])
        rows.append([ARM_NAMES[order], 'C4a without retrieval', fmt(stat(results, 'coupling', 'PAD', f'cell_noret|injection|{order}|C4a')), '--'])
    rows.append(['order gap', 'C4a crop - full', fmt(stat(results, 'coupling', 'PAD', 'order_gap|injection')), '--'])
    rows.append(['order gap', 'from higher crop-first tool error', fmt(stat(results, 'coupling', 'PAD', 'order_gap_from_tool_error|injection')), '--'])
    rows.append(['order gap', 'excess dependence', fmt(stat(results, 'coupling', 'PAD', 'order_gap_excess_dependence|injection')), '--'])
    write_table(rows, ['Order', 'Cell', 'Share of all cases', 'Share of faulty cases'], 'Coupling cells under injection (pooled over reasoners)', out, 'tab_cells',
                note='Error rates and independence baselines are averaged over reasoners; all intervals are patient-clustered bootstraps.')


def transition_text(rs):
    start, end = (rs.get('transition') or [float('nan'), float('nan')])[:2]
    ratio = rs.get('ratio_low_to_high', float('nan'))
    if finite(start, end):
        head = f'transition at {start:g}-{end:g} px'
    elif finite(start):
        head = f'above null band from {start:g} px, no plateau'
    else:
        head = 'no height above null band'
    return head + (f'; DA(8px)/DA(20px) {ratio:.2f}' if finite(ratio) else '')


def table_controls(results, args, out):
    encoder = results.get('encoder_selection', {}).get('selected', 'biomedclip')
    view = args.capture_view
    mask = results.get('mask_iou', {})
    rs = results.get('render_scale', {}).get('PAD', {})
    rows = [
        ['Neutral overlay', 'occlusion', 'Delta_cap ' + fmt(stat(results, 'channel', 'PAD', f'delta_cap|{encoder}|{view}|neutral'))],
        ['Physical corruption', 'non-directional shift', 'Delta_cap ' + fmt(stat(results, 'channel', 'PAD', f'delta_cap|{encoder}|{view}|corruption'))],
        ['Render-scale sweep', 'scale-invariant capture*', transition_text(rs)],
        ['Mask IoU', 'mask movement', ' / '.join(f'{site} {v["by_family"].get("injection", float("nan")):.2f}' for site, v in mask.items())],
        ['Mask covers lesion', 'mask error', 'two-reviewer pass pending (metrics/mask_review/review_sheet.csv)'],
        ['Biopsy-only subset', 'label provenance', 'FAF ' + fmt(stat(results, 'biopsy_subset', 'raw', 'faf_cwr|injection|crop'))
         + '; standardised ' + fmt(stat(results, 'biopsy_subset', 'class_standardised', 'faf_cwr|injection|crop'))],
        ['Mirrored direction', 'class prior', 'interaction ' + fmt(stat(results, 'channel', 'PAD', 'retrieval_direction_order_interaction'))],
        ['Retrieval ablation', 'tool unused', 'CWR ' + fmt(stat(results, 'agent', 'PAD', 'cwr|injection|crop'), ci=False) + ' -> '
         + fmt(stat(results, 'agent', 'PAD', 'cwr|injection|crop-noret'), ci=False)],
    ]
    write_table(rows, ['Control', 'Rules out', 'Result'], 'Controls that could otherwise explain the order effect', out, 'tab_controls',
                note='Intervals are patient-clustered bootstraps. *Supports a scale account; does not separate scale from the context change caused by cropping.')


def table_external(results, out):
    if 'external' not in results:
        return
    rows = []
    for name, label in (('bal_acc_clean_crop', 'Bal. acc. (clean)'), ('c4a_crop', 'C4a, crop-first'), ('c4a_full', 'C4a, full-image'),
                        ('faf_cwr_crop', 'FAF cwr, crop-first'), ('order_gap', 'Order gap')):
        rows.append([label, fmt(stat(results, 'external', f'raw|{name}|internal')), fmt(stat(results, 'external', f'raw|{name}|external')),
                     fmt(stat(results, 'external', f'raw|{name}|difference')), fmt(stat(results, 'external', f'standardised|{name}|difference'))])
    write_table(rows, ['Quantity', 'Internal', 'External', 'Diff.', 'Class-standardised diff.'], 'External validity', out, 'tab_external')


def table_setup(results, out):
    cfg = results.get('config', {})
    keys = ['data_path', 'hiba_path', 'smoke_image', 'split_fractions', 'sam_model', 'crop_margin', 'channel_encoders', 'agent_encoder',
            'reasoners', 'reasoner_dtype', 'decodes', 'temperature', 'top_p', 'max_new_tokens', 'react_max_steps', 'tau', 'knn',
            'blur_severity', 'jpeg_severity', 'corruption_scale', 'payload_width_frac', 'low_contrast_delta', 'gate_target_rate',
            'n_boot', 'seed']
    rows = [[k, str(cfg.get(k))] for k in keys]
    rows.append(['selected agent encoder', str(results.get('encoder_selection', {}).get('selected'))])
    rows.append(['gate theta', str(results.get('gate_params', {}).get('theta'))])
    write_table(rows, ['Setting', 'Value'], 'Configuration of this run', out, 'tab_setup')


TRANSITION_CELLS = ('C1', 'C2', 'C3', 'C4a', 'CX')


def table_transitions(results, out):
    if 'transitions' not in results:
        return
    rows = []
    for src in TRANSITION_CELLS:
        rows.append([f'full-image {src}'] + [fmt(stat(results, 'transitions', 'PAD', f'trans|injection|{src}->{dst}'), ci=False)
                                              for dst in TRANSITION_CELLS])
    write_table(rows, ['Full-image cell'] + [f'crop-first {c}' for c in TRANSITION_CELLS],
                'Paired cell transitions under injection, share of all cases (rows: full-image, columns: crop-first)', out,
                'tab_transitions_matrix', note='Same images under both orders; joint shares averaged over reasoners. C4b is empty under a binary decision.')
    keys = [('c4a_crop_given_c3_full', 'P(crop-first C4a | full-image C3)'),
            ('c3_full_given_c4a_crop', 'P(full-image C3 | crop-first C4a)'),
            ('hidden_given_detectable', 'P(crop-first C4a | full-image C2 or C3)'),
            ('detectable_given_hidden', 'P(crop-first C2 or C3 | full-image C4a)'),
            ('detectable_to_hidden', 'share of all cases: C2/C3 -> C4a'),
            ('hidden_to_detectable', 'share of all cases: C4a -> C2/C3'),
            ('into_c4a', 'share of all cases entering C4a'),
            ('out_of_c4a', 'share of all cases leaving C4a'),
            ('tool_flip_between_orders', 'tool label differs between orders'),
            ('reasoner_flip_between_orders', 'reasoner label differs between orders'),
            ('reasoner_flip_noise', 'reasoner label differs between the two identical no-retrieval arms (noise)')]
    rows = [[label, fmt(stat(results, 'transitions', 'PAD', f'{key}|injection'))] for key, label in keys]
    write_table(rows, ['Quantity', 'Estimate [95% CI]'], 'Paired transitions between pipeline orders under injection', out, 'tab_transitions')


def table_factorial(results, out):
    if 'factorial' not in results:
        return
    heights = results.get('factorial_heights', {})
    rows = []
    for metric, label in (('shift', 'retrieval shift toward payload target'), ('tool_wrong', 'tool error'), ('delta_cap', 'Delta_cap')):
        for view in ('full', 'crop'):
            for scale in ('small', 'large'):
                rows.append([label, f'{ARM_NAMES[view]} view, {scale} payload ({num(heights.get(scale), 1)} px)',
                             fmt(stat(results, 'factorial', 'PAD', f'{metric}|{view}|{scale}'))])
        for key, name in (('scale_main', 'main effect of scale'), ('view_main', 'main effect of view (cropping)'),
                          ('interaction', 'scale x view interaction'), ('natural_contrast', 'natural order contrast (crop large - full small)'),
                          ('scale_share', 'share of natural contrast due to scale')):
            rows.append([label, name, fmt(stat(results, 'factorial', 'PAD', f'{metric}|{key}'))])
    write_table(rows, ['Metric', 'Cell or effect', 'Estimate [95% CI]'], 'Retrieval-only scale x cropping factorial (internal test)', out, 'tab_factorial',
                note='Payload rendered at the clean lesion centre so that its height at the encoder input is equal in both views; agent encoder only, no reasoner.')


def num(value, digits=2):
    return f'{float(value):.{digits}f}' if value is not None and np.isfinite(float(value)) else 'n/a'


def table_payload_scale(results, out):
    rows = []
    for site, s in (results.get('payload_scale') or {}).items():
        if site == 'provenance' or not isinstance(s, dict) or 'eff_height_full_median' not in s:
            continue
        for order in ('full', 'crop'):
            rows.append([site, ARM_NAMES[order], num(s[f'eff_height_{order}_median']),
                         f"{num(s[f'eff_height_{order}_q25'])}-{num(s[f'eff_height_{order}_q75'])}",
                         num(s[f'eff_height_{order}_frac_at_or_above_null_edge']), num(s['null_edge_px'], 1)])
    if rows:
        write_table(rows, ['Site', 'Order', 'Median px', 'IQR px', 'Share at or above null edge', 'Null edge px'],
                    'Payload height at the encoder input across injected query images', out, 'tab_payload_scale')


def stage_report(args, ws):
    results = load_json(ws.metrics / 'results.json')
    out = ws.report
    with plt.rc_context(STYLE):
        figure_directional(results, args, out)
        figure_renderscale(results, args, out)
        figure_retrieval(results, out)
        figure_provenance(results, args, ws, out)
        if 'coupling' in results:
            figure_coupling(results, out)
            figure_faf(results, out)
            figure_gate(results, args, out)
    if 'coupling' in results:
        table_main(results, out)
        table_cells(results, out)
        table_transitions(results, out)
    table_factorial(results, out)
    table_payload_scale(results, out)
    table_controls(results, args, out)
    table_external(results, out)
    table_setup(results, out)
    log(f'report: figures and tables written to {out}')
