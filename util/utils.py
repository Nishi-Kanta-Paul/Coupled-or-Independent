import os
import gc
import json
import random
import hashlib
import datetime
from pathlib import Path

import numpy as np
import torch


def log(msg):
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] {msg}', flush=True)


def set_seed(seed):
    seed = int(seed) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(*parts):
    key = '|'.join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:15], 16)


def resolve_device(name):
    if name.startswith('cuda') and not torch.cuda.is_available():
        log('CUDA is not available, using CPU')
        return torch.device('cpu')
    return torch.device(name)


def resolve_dtype(name):
    return {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[name]


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Cannot serialize {type(obj)}')


def clean_floats(obj):
    if isinstance(obj, dict):
        return {str(k): clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_floats(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(clean_floats(obj), f, indent=2, default=to_serializable)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(record, path):
    ensure_dir(Path(path).parent)
    with open(path, 'a') as f:
        f.write(json.dumps(clean_floats(record), default=to_serializable) + '\n')


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_mask(mask, path):
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, bits=np.packbits(mask.astype(bool)), shape=np.array(mask.shape))


def load_mask(path):
    data = np.load(path)
    shape = tuple(int(s) for s in data['shape'])
    return np.unpackbits(data['bits'])[:shape[0] * shape[1]].reshape(shape).astype(bool)


def case_key(*parts):
    return '|'.join(str(p) for p in parts)


def model_tag(model_id):
    return model_id.replace('/', '__')
