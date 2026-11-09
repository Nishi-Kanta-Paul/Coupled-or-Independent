import os
import glob

import numpy as np
import pandas as pd

from .utils import log


PAD_CANCER = ('BCC', 'SCC', 'MEL')
PAD_NONCANCER = ('ACK', 'NEV', 'SEK')
HIBA_CANCER = ('BCC', 'SCC', 'MM')
HIBA_NONCANCER = ('AK', 'NV', 'SK')
HIBA_KEYWORDS = (
    ('basal cell carcinoma', 'BCC'),
    ('squamous cell carcinoma', 'SCC'),
    ('melanoma', 'MM'),
    ('actinic keratosis', 'AK'),
    ('seborrheic keratosis', 'SK'),
    ('nevus', 'NV'),
)
DIAGNOSIS_NAMES = {
    'BCC': 'basal cell carcinoma', 'SCC': 'squamous cell carcinoma', 'MEL': 'melanoma', 'MM': 'melanoma',
    'ACK': 'actinic keratosis', 'AK': 'actinic keratosis', 'NEV': 'nevus', 'NV': 'nevus',
    'SEK': 'seborrheic keratosis', 'SK': 'seborrheic keratosis',
}
CASE_COLUMNS = ['site', 'img_id', 'patient_id', 'lesion_id', 'diagnosis', 'label', 'biopsied', 'fitzpatrick', 'path']


def find_images(root, extensions=('.png', '.jpg', '.jpeg')):
    paths = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() in extensions:
                paths[stem] = os.path.join(dirpath, name)
    return paths


def load_pad(root):
    meta = pd.read_csv(os.path.join(root, 'metadata.csv'))
    paths = find_images(os.path.join(root, 'images'))
    frame = pd.DataFrame({
        'site': 'PAD',
        'img_id': meta['img_id'].str.replace(r'\.png$', '', regex=True),
        'patient_id': 'PAD:' + meta['patient_id'].astype(str),
        'lesion_id': 'PAD:' + meta['lesion_id'].astype(str),
        'diagnosis': meta['diagnostic'].astype(str),
        'biopsied': meta['biopsed'].astype(str).str.lower().eq('true'),
        'fitzpatrick': pd.to_numeric(meta['fitspatrick'], errors='coerce'),
    })
    frame['label'] = frame['diagnosis'].isin(PAD_CANCER).astype(int)
    frame['path'] = frame['img_id'].map(paths)
    missing = int(frame['path'].isna().sum())
    if missing:
        log(f'PAD-UFES-20: {missing} metadata rows have no image file')
    return frame[CASE_COLUMNS]


def map_hiba_diagnosis(row, columns):
    text = ' | '.join(str(row[c]) for c in columns if pd.notna(row[c])).lower()
    for keyword, code in HIBA_KEYWORDS:
        if keyword in text:
            return code
    return None


def load_hiba(root):
    meta = None
    for path in sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)):
        candidate = pd.read_csv(path)
        if 'isic_id' in candidate.columns:
            meta = candidate
            break
    if meta is None:
        raise FileNotFoundError(f'No ISIC metadata CSV with an isic_id column under {root}')
    n_all = len(meta)
    if 'image_type' in meta.columns:
        meta = meta[meta['image_type'].astype(str).str.lower().str.startswith('clinical')]
    n_clinical = len(meta)
    diag_columns = [c for c in ('diagnosis', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4', 'diagnosis_5') if c in meta.columns]
    meta = meta.assign(code=meta.apply(lambda r: map_hiba_diagnosis(r, diag_columns), axis=1))
    meta = meta[meta['code'].notna()]
    paths = find_images(root)
    patient = meta['patient_id'] if 'patient_id' in meta.columns else pd.Series(np.nan, index=meta.index)
    lesion = meta['lesion_id'] if 'lesion_id' in meta.columns else pd.Series(np.nan, index=meta.index)
    patient = patient.fillna(lesion).fillna(meta['isic_id'])
    confirm = meta['diagnosis_confirm_type'] if 'diagnosis_confirm_type' in meta.columns else pd.Series('', index=meta.index)
    fitz = meta['fitzpatrick_skin_type'] if 'fitzpatrick_skin_type' in meta.columns else pd.Series(np.nan, index=meta.index)
    roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
    frame = pd.DataFrame({
        'site': 'HIBA',
        'img_id': meta['isic_id'].astype(str),
        'patient_id': 'HIBA:' + patient.astype(str),
        'lesion_id': 'HIBA:' + lesion.fillna(meta['isic_id']).astype(str),
        'diagnosis': meta['code'],
        'biopsied': confirm.astype(str).str.lower().str.contains('histopathology'),
        'fitzpatrick': fitz.map(lambda v: roman.get(str(v).strip().upper(), np.nan)),
    })
    frame['label'] = frame['diagnosis'].isin(HIBA_CANCER).astype(int)
    frame['path'] = frame['img_id'].map(paths)
    log(f'HIBA: {n_all} rows, {n_clinical} clinical, {len(frame)} in the six shared diagnoses, '
        f'{frame["lesion_id"].nunique()} lesions, {frame["patient_id"].nunique()} patients')
    frame = frame[frame['path'].notna()]
    return frame[CASE_COLUMNS].reset_index(drop=True)


def patient_split(frame, fractions, seed, stratify=True):
    patients = frame.groupby('patient_id')['label'].max().reset_index()
    rng = np.random.default_rng(seed)
    patients['u'] = rng.random(len(patients))
    if stratify:
        rank = patients.groupby('label')['u'].rank(method='first')
        size = patients.groupby('label')['u'].transform('size')
        patients['key'] = (rank - rng.random(len(patients))) / size
    else:
        patients['key'] = patients['u']
    patients = patients.sort_values('key', kind='mergesort').reset_index(drop=True)
    n = len(patients)
    n_ref = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    names = np.array(['test'] * n, dtype=object)
    names[:n_ref] = 'reference'
    names[n_ref:n_ref + n_val] = 'validation'
    patients['split'] = names
    return frame.merge(patients[['patient_id', 'split']], on='patient_id', how='left')


def dataset_summary(frame):
    fitz = pd.to_numeric(frame['fitzpatrick'], errors='coerce')
    summary = {
        'n_images': int(len(frame)),
        'n_patients': int(frame['patient_id'].nunique()),
        'n_lesions': int(frame['lesion_id'].nunique()),
        'cancer_prevalence': float(frame['label'].mean()),
        'class_counts': frame['diagnosis'].value_counts().to_dict(),
        'biopsy_rate_by_class': frame.groupby('diagnosis')['biopsied'].mean().to_dict(),
        'biopsy_rate_overall': float(frame['biopsied'].mean()),
        'fitzpatrick_iv_vi': int((fitz >= 4).sum()),
        'fitzpatrick_recorded': int(fitz.notna().sum()),
    }
    if 'split' in frame.columns:
        summary['splits'] = {
            name: {
                'images': int(len(part)),
                'patients': int(part['patient_id'].nunique()),
                'cancer': int(part['label'].sum()),
                'noncancer': int((1 - part['label']).sum()),
            }
            for name, part in frame.groupby('split')
        }
    return summary
