import numpy as np
import cv2
from PIL import Image
from scipy import ndimage, stats


STAT_KEYS = ('text_score', 'laplacian_variance', 'contrast_skew')


def working_gray(image, work_side):
    gray = image.convert('L')
    scale = work_side / float(max(gray.size))
    if scale < 1.0:
        gray = gray.resize((max(1, int(round(gray.size[0] * scale))), max(1, int(round(gray.size[1] * scale)))), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32) / 255.0


def text_score(gray, sigma, threshold, close_len, open_len):
    local_mean = ndimage.gaussian_filter(gray, sigma)
    centred = gray - local_mean
    local_std = np.sqrt(ndimage.gaussian_filter(centred ** 2, sigma)) + 1e-3
    response = (np.abs(centred / local_std) > threshold).astype(np.uint8)
    closed = cv2.morphologyEx(response, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_len, 1)))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_len, 1)))
    return float(opened.mean())


def laplacian_variance(gray):
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


def contrast_skew(gray, window):
    mean = ndimage.uniform_filter(gray.astype(np.float64), window)
    sq = ndimage.uniform_filter(gray.astype(np.float64) ** 2, window)
    local_std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    return float(stats.skew(local_std.ravel()))


def gate_statistics(image, args):
    gray = working_gray(image, args.gate_work_side)
    return {
        'text_score': text_score(gray, args.text_sigma, args.text_threshold, args.text_close, args.text_open),
        'laplacian_variance': laplacian_variance(gray),
        'contrast_skew': contrast_skew(gray, args.contrast_window),
    }


def fit_standardizer(frame):
    params = {}
    for key in STAT_KEYS:
        values = frame[key].to_numpy(dtype=float)
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float('nan')
        degenerate = not np.isfinite(sigma) or sigma <= 0
        params[key] = {'mu': mu, 'sigma': 1.0 if degenerate else sigma, 'degenerate': bool(degenerate)}
    return params


def q_values(frame, params):
    z = np.stack([(frame[k].to_numpy(dtype=float) - params[k]['mu']) / params[k]['sigma'] for k in STAT_KEYS], axis=1)
    return z.max(axis=1)


def select_threshold(values, target_rate):
    values = np.asarray(values, dtype=float)
    for candidate in np.sort(np.unique(values)):
        if np.mean(values > candidate) <= target_rate:
            return float(candidate)
    return float(np.max(values))


def gate_decisions(y_tool, y_reasoner, kappa, q, requery_label, requery_kappa, theta, tau):
    flag_disagree = y_tool != y_reasoner
    flag_q = q > theta
    flag = flag_disagree | flag_q
    stable = requery_kappa >= tau - 1e-9
    answered = ~flag | stable
    final = np.where(flag, requery_label, y_reasoner)
    final_kappa = np.where(flag, requery_kappa, kappa)
    return {'flag_disagree': flag_disagree, 'flag_q': flag_q, 'flag': flag, 'answered': answered, 'final': final, 'final_kappa': final_kappa}
