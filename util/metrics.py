import numpy as np


CELLS = ('C1', 'C2', 'C3', 'C4a', 'C4b', 'CX')
INVALID = -1


def as_label(value):
    return INVALID if value is None else int(value)


def modal_answer(labels, confidences):
    labels = [as_label(v) for v in labels]
    values, counts = np.unique(labels, return_counts=True)
    best = counts.max()
    tied = [int(v) for v, c in zip(values, counts) if c == best]
    if len(tied) > 1:
        tied = [v for v in tied if v != INVALID] or tied
    if len(tied) > 1:
        def mean_confidence(v):
            vals = [c for l, c in zip(labels, confidences) if l == v and c is not None]
            return float(np.mean(vals)) if vals else -1.0
        tied = sorted(tied, key=lambda v: (-mean_confidence(v), v))
    return tied[0], float(best) / len(labels)


def verbal_confidence(labels, confidences, modal):
    vals = [c for l, c in zip(labels, confidences) if as_label(l) == modal and c is not None]
    return float(np.mean(vals)) if vals else float('nan')


def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x)
    total = w[ok].sum()
    return float((w[ok] * x[ok]).sum() / total) if total > 0 else float('nan')


def balanced_accuracy(y, yhat, w):
    recalls = []
    for c in (0, 1):
        m = y == c
        if w[m].sum() > 0:
            recalls.append(wmean((yhat[m] == c).astype(float), w[m]))
    return float(np.mean(recalls)) if recalls else float('nan')


def confident_wrong(y, modal, kappa, tau):
    return ((modal != y) & (kappa >= tau - 1e-9)).astype(float)


def expected_calibration_error(confidence, correct, w, bins):
    confidence = np.asarray(confidence, dtype=float)
    ok = np.isfinite(confidence)
    confidence, correct, w = confidence[ok], np.asarray(correct, dtype=float)[ok], np.asarray(w, dtype=float)[ok]
    total = w.sum()
    if total <= 0:
        return float('nan')
    index = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = index == b
        wb = w[m].sum()
        if wb > 0:
            ece += wb * abs(wmean(correct[m], w[m]) - wmean(confidence[m], w[m]))
    return float(ece / total)


def coupling_cell(y, y_tool, y_reasoner):
    if y_reasoner == INVALID:
        return 'CX'
    tool_ok, reasoner_ok = y_tool == y, y_reasoner == y
    if tool_ok and reasoner_ok:
        return 'C1'
    if reasoner_ok:
        return 'C2'
    if tool_ok:
        return 'C3'
    return 'C4a' if y_tool == y_reasoner else 'C4b'


def cell_shares(cells, w):
    cells = np.asarray(cells)
    return {c: wmean((cells == c).astype(float), w) for c in CELLS}


def faulty_shares(shares):
    faulty = 1.0 - shares['C1']
    return {c: (shares[c] / faulty if faulty > 0 else float('nan')) for c in CELLS if c != 'C1'}


def disagreement_ceiling(shares):
    den = shares['C2'] + shares['C3'] + shares['C4a']
    return (shares['C2'] + shares['C3']) / den if den > 0 else float('nan')


def amplification(eps_a_p, eps_a_0, eps_b_p, eps_b_0):
    den = eps_a_p - eps_a_0
    if not np.isfinite(den) or den <= 0:
        return float('nan')
    return float((eps_b_p - eps_b_0) / den)


def project_out(v, mu):
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    mu = np.asarray(mu, dtype=np.float64)
    return v - np.outer(v @ mu, mu) / float(mu @ mu)


def directional_alignment(delta, text, mu):
    a = project_out(delta, mu)
    b = project_out(text, mu)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def capture_margin(delta, payload_embedding, control_embeddings, mu):
    da_payload = directional_alignment(delta, payload_embedding, mu)
    da_controls = np.stack([directional_alignment(delta, c, mu) for c in control_embeddings], axis=0)
    return da_payload - da_controls.mean(axis=0), da_payload, da_controls.mean(axis=0)


def modality_gap(image_embeddings, text_embeddings):
    return np.asarray(image_embeddings, dtype=np.float64).mean(axis=0) - np.asarray(text_embeddings, dtype=np.float64).mean(axis=0)
