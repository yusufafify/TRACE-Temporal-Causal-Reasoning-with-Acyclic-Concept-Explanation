"""LUMIERE Brain Tumor MRI Dataset.

18 explicit RANO-style concepts, 4 classes: PD, SD, PR, CR.
"""

from env import CACHE

import os
import re
import gc
import math
import logging
import random
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from sklearn.model_selection import StratifiedGroupKFold
from src.data.utils import split_dataset

# SPD extraction (RANO criteria)
from spd_extraction import calculate_2d_spd_from_mask


LABEL_MAP = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}

# Binary remap: CR/PR/SD → Non-Progressive (0), PD → Progressive (1)
BINARY_LABEL_REMAP = {0: 0, 1: 0, 2: 0, 3: 1}

CONCEPT_NAMES = [
    'enhancing_tumor_volume_cm3',
    'non_enhancing_volume_cm3',
    'followup_enhancing_volume_cm3',
    'followup_non_enhancing_volume_cm3',
    'baseline_spd_cm2',
    'followup_spd_cm2',
    'time_gap',
    'delta_enhancing_absolute',
    'delta_enhancing_percent',
    'delta_non_enhancing_absolute',
    'delta_non_enhancing_percent',
    'delta_spd_absolute',
    'delta_spd_percent',
    'new_lesion_flag',
    'vol_pd_flag',
    'vol_pr_flag',
    'spd_pd_flag',
    'spd_pr_flag',
]

VALIDATED_CONCEPTS = CONCEPT_NAMES[:]

N_CONCEPTS = len(CONCEPT_NAMES)

BINARY_CONCEPTS = {
    'new_lesion_flag',
    'vol_pd_flag',
    'vol_pr_flag',
    'spd_pd_flag',
    'spd_pr_flag',
}

RANO_CONCEPTS = set()

# Static per-timepoint volumes are supervised in log1p-space, following the
# explicit-delta Siamese reference implementation.
LOG_TRANSFORM_CONCEPTS = {
    'enhancing_tumor_volume_cm3',
    'non_enhancing_volume_cm3',
    'followup_enhancing_volume_cm3',
    'followup_non_enhancing_volume_cm3',
    'baseline_spd_cm2',
    'followup_spd_cm2',
}
LOG_TRANSFORM_INDICES = {i for i, n in enumerate(CONCEPT_NAMES) if n in LOG_TRANSFORM_CONCEPTS}

# All categorical concepts (cardinality > 1)
CATEGORICAL_CONCEPTS = BINARY_CONCEPTS | RANO_CONCEPTS

DELTA_EPSILON = 1e-4
MIN_MEASURABLE_ENHANCING_CM3 = 0.5
MIN_MEASURABLE_SPD_CM2 = 0.01

def _concept_cardinality(name):
    """Return cardinality for a concept by name."""
    if name in RANO_CONCEPTS:
        return 3
    if name in BINARY_CONCEPTS:
        return 2
    return 1

def _encode_rano_class(val_percent):
    """Encode a delta-percent value into RANO 3-class index.
    0: < -50%  (PR/CR territory — significant shrinkage)
    1: -50% to +25%  (SD territory — stable)
    2: > +25%  (PD territory — significant growth)
    """
    if val_percent < -50.0:
        return 0.0
    elif val_percent <= 25.0:
        return 1.0
    else:
        return 2.0


def _clamped_relative_change(followup_val, baseline_val, epsilon=DELTA_EPSILON):
    baseline_val = float(baseline_val)
    followup_val = float(followup_val)
    raw_delta = (followup_val - baseline_val) / (baseline_val + float(epsilon))
    return float(np.clip(raw_delta, -1.0, 5.0))


def _build_explicit_concept_vector(
    base_enh,
    base_ne,
    followup_enh,
    followup_ne,
    new_lesion_flag,
    base_spd=0.0,
    followup_spd=0.0,
    time_gap=0.0,
):
    """Build the 18-concept vector in the reference explicit-delta style."""
    base_enh = float(base_enh)
    base_ne = float(base_ne)
    followup_enh = float(followup_enh)
    followup_ne = float(followup_ne)
    base_spd = float(base_spd)
    followup_spd = float(followup_spd)
    time_gap = max(0.0, float(time_gap))

    delta_enh_abs = followup_enh - base_enh
    delta_ne_abs = followup_ne - base_ne
    delta_enh_pct = _clamped_relative_change(followup_enh, base_enh)
    delta_ne_pct = _clamped_relative_change(followup_ne, base_ne)
    delta_spd_abs = followup_spd - base_spd
    delta_spd_pct = _clamped_relative_change(followup_spd, base_spd)

    measurable_enh = base_enh >= MIN_MEASURABLE_ENHANCING_CM3
    measurable_spd = base_spd >= MIN_MEASURABLE_SPD_CM2

    vol_pd_flag = float((delta_enh_pct >= 0.40) and measurable_enh)
    vol_pr_flag = float((delta_enh_pct <= -0.65) and measurable_enh)
    spd_pd_flag = float((delta_spd_pct >= 0.25) and measurable_spd)
    spd_pr_flag = float((delta_spd_pct <= -0.50) and measurable_spd)

    return [
        math.log1p(max(0.0, base_enh)),
        math.log1p(max(0.0, base_ne)),
        math.log1p(max(0.0, followup_enh)),
        math.log1p(max(0.0, followup_ne)),
        math.log1p(max(0.0, base_spd)),
        math.log1p(max(0.0, followup_spd)),
        time_gap,
        float(delta_enh_abs),
        float(delta_enh_pct),
        float(delta_ne_abs),
        float(delta_ne_pct),
        float(delta_spd_abs),
        float(delta_spd_pct),
        float(new_lesion_flag > 0.5),
        vol_pd_flag,
        vol_pr_flag,
        spd_pd_flag,
        spd_pr_flag,
    ]




def _load_seg_volumes(folder_path):
    """Load segmentation and return (enhancing_vol, non_enhancing_vol, spd_cm2, seg_data_or_None)."""
    seg_path = os.path.join(folder_path, "seg_preproc.nii.gz")
    if not os.path.exists(seg_path):
        return 0.0, 0.0, 0.0, None
    try:
        nii = nib.load(seg_path)
        seg_data = nii.get_fdata()
        seg_labels = np.rint(seg_data).astype(int)
        enhancing_vol = float((seg_labels == 2).sum())
        non_enhancing_vol = float((seg_labels == 1).sum())
        # Extract SPD from segmentation using RANO criteria
        voxel_spacing = nii.header.get_zooms()[:3]
        spd_cm2 = calculate_2d_spd_from_mask(seg_labels, voxel_spacing)
        return enhancing_vol, non_enhancing_vol, spd_cm2, seg_labels
    except Exception as e:
        logging.warning(f"Error loading segmentation {seg_path}: {e}")
        return 0.0, 0.0, 0.0, None


def _load_seg_3ch(folder_path, target_size=224):
    """Load segmentation as 3-channel image: background, non-enhancing, enhancing.

    Extracts the axial slice with the most tumor, converts seg labels to 3
    binary channels, resizes to ``target_size x target_size``.
    Returns: torch.Tensor [3, target_size, target_size]
    """
    seg_path = os.path.join(folder_path, "seg_preproc.nii.gz")
    if not os.path.exists(seg_path):
        return torch.zeros(3, target_size, target_size)
    try:
        seg_data = np.asarray(nib.load(seg_path).dataobj, dtype=np.int8)
        if len(seg_data.shape) != 3:
            return torch.zeros(3, target_size, target_size)

        # Best axial slice (most tumor voxels)
        tumor_per_slice = (seg_data > 0).sum(axis=(0, 1))
        if tumor_per_slice.max() > 0:
            best_z = int(np.argmax(tumor_per_slice))
        else:
            best_z = seg_data.shape[2] // 2
        sl = seg_data[:, :, best_z]

        # 3 binary channels
        ch_bg  = (sl == 0).astype(np.float32)
        ch_ne  = (sl == 1).astype(np.float32)   # non-enhancing / edema
        ch_enh = (sl == 2).astype(np.float32)   # enhancing tumor
        img_np = np.stack([ch_bg, ch_ne, ch_enh], axis=0)  # [3, H, W]

        # Resize to target_size
        resize_tf = transforms.Compose([
            transforms.Resize((target_size, target_size)),
        ])
        img_t = torch.from_numpy(img_np)  # [3, H, W]
        img_t = resize_tf(img_t)          # [3, target_size, target_size]
        return img_t
    except Exception as e:
        logging.warning(f"Error loading seg for 3-ch: {e}")
        return torch.zeros(3, target_size, target_size)


def _detect_new_lesion(current_seg, baseline_seg):
    """Detect if there are new connected components in enhancing (label=2)
    compared to baseline. Returns 1 if new lesion, 0 otherwise."""
    if current_seg is None or baseline_seg is None:
        return 0
    try:
        current_enh = (current_seg == 2).astype(int)
        baseline_enh = (baseline_seg == 2).astype(int)
        _, n_current = connected_components(current_enh)
        _, n_baseline = connected_components(baseline_enh)
        return 1 if n_current > n_baseline else 0
    except Exception:
        return 0


def _load_4channel(folder_path, best_slice_idx):
    """Load T1, T1Gd, T2, FLAIR NIfTI files and extract 5 adjacent slices.
    Returns: torch.Tensor of shape (20, 224, 224) -> 4 modalities * 5 slices."""
    modalities = ['t1_preproc', 't1Gd_preproc', 't2_preproc', 'flair_preproc']
    channels = []
    ref_shape = None

    for mod in modalities:
        img_path = os.path.join(folder_path, f"{mod}.nii.gz")
        if os.path.exists(img_path):
            try:
                nii_img = nib.load(img_path)
                img_data = np.asarray(nii_img.dataobj, dtype=np.float32)  # float32 saves ~50% RAM
                if len(img_data.shape) == 3:
                    if best_slice_idx is not None:
                        center_idx = min(best_slice_idx, img_data.shape[2] - 1)
                    else:
                        center_idx = img_data.shape[2] // 2
                    
                    # Extract 5 adjacent slices
                    slice_indices = [
                        max(0, center_idx - 2),
                        max(0, center_idx - 1),
                        center_idx,
                        min(img_data.shape[2] - 1, center_idx + 1),
                        min(img_data.shape[2] - 1, center_idx + 2)
                    ]
                    
                    for slice_idx in slice_indices:
                        slice_2d = img_data[:, :, slice_idx]
                        
                        # Min-max rescale to [0, 1] per slice
                        denom = slice_2d.max() - slice_2d.min()
                        if denom > 1e-8:
                            slice_2d = (slice_2d - slice_2d.min()) / denom
                        else:
                            slice_2d = np.zeros_like(slice_2d)

                        if ref_shape is None:
                            ref_shape = slice_2d.shape
                        channels.append(slice_2d)
                
                elif len(img_data.shape) == 2: # Very rare 2D case
                    slice_2d = img_data
                    denom = slice_2d.max() - slice_2d.min()
                    if denom > 1e-8:
                        slice_2d = (slice_2d - slice_2d.min()) / denom
                    else:
                        slice_2d = np.zeros_like(slice_2d)
                        
                    if ref_shape is None:
                        ref_shape = slice_2d.shape
                    
                    # Duplicate the 2D slice 5 times
                    for _ in range(5):
                        channels.append(slice_2d)
                else:
                    for _ in range(5):
                        channels.append(None)
                    continue

            except Exception as e:
                logging.warning(f"Error loading {img_path}: {e}")
                for _ in range(5):
                    channels.append(None)
        else:
            for _ in range(5):
                channels.append(None)

    if ref_shape is None:
        ref_shape = (224, 224)

    # Replace None with zeros
    # We expect 20 channels (4 modalities * 5 slices)
    expected_channels = len(modalities) * 5
    channels = [ch if ch is not None else np.zeros(ref_shape) for ch in channels]

    # Ensure same shape
    unified = []
    for ch in channels:
        if ch.shape != ref_shape:
            ch_uint8 = ((ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255).astype(np.uint8)
            ch_pil = Image.fromarray(ch_uint8, mode='L').resize((ref_shape[1], ref_shape[0]))
            ch = np.array(ch_pil).astype(np.float64) / 255.0
        unified.append(ch)
    channels = unified

    while len(channels) < expected_channels:
        channels.append(channels[-1] if channels else np.zeros(ref_shape))

    # Resize each channel to 224x224 and per-channel z-score
    resize_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    resized = []
    for ch_idx in range(expected_channels):
        ch = channels[ch_idx]
        ch_uint8 = (ch * 255).astype(np.uint8)
        ch_pil = Image.fromarray(ch_uint8, mode='L')
        ch_tensor = resize_tf(ch_pil)  # [1, 224, 224]
        resized.append(ch_tensor)
    img_tensor = torch.cat(resized, dim=0)  # [20, 224, 224]

    # Per-channel z-score normalization
    for ch_idx in range(img_tensor.shape[0]):
        ch = img_tensor[ch_idx]
        ch_mean = ch.mean()
        ch_std = ch.std()
        if ch_std > 1e-8:
            img_tensor[ch_idx] = (ch - ch_mean) / ch_std
        else:
            img_tensor[ch_idx] = torch.zeros_like(ch)

    # Safety: replace NaN/Inf
    if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
        img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return img_tensor  # (20, 224, 224)


def _get_best_slice_idx(folder_path):
    """Determine best axial slice using segmentation mask (max tumor area)."""
    seg_path = os.path.join(folder_path, "seg_preproc.nii.gz")
    if not os.path.exists(seg_path):
        return None
    try:
        seg_data = np.asarray(nib.load(seg_path).dataobj, dtype=np.int8)  # int8 is enough for labels 0/1/2
        if len(seg_data.shape) == 3:
            tumor_per_slice = (seg_data > 0).sum(axis=(0, 1))
            if tumor_per_slice.max() > 0:
                return int(np.argmax(tumor_per_slice))
    except Exception:
        pass
    return None




def _as_shape_tuple(shape):
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _normalize_3d_channels(volume):
    volume = torch.nan_to_num(volume.float(), nan=0.0, posinf=0.0, neginf=0.0)
    for ch_idx in range(volume.shape[0]):
        ch = volume[ch_idx]
        std = ch.std()
        if std > 1e-8:
            volume[ch_idx] = (ch - ch.mean()) / std
        else:
            volume[ch_idx] = torch.zeros_like(ch)
    return volume


def _resize_3d_volume(volume, target_shape):
    if target_shape is None or tuple(volume.shape[-3:]) == tuple(target_shape):
        return volume
    return F.interpolate(
        volume.unsqueeze(0),
        size=target_shape,
        mode='trilinear',
        align_corners=False,
    ).squeeze(0)


def _load_full_mri_volume(folder_path, cache_dir=None, target_shape=(128, 128, 128)):
    """Load the full 3D 4-modality MRI volume for MedicalNet encoder training.

    Returns a fixed-shape tensor [4, D, H, W]. Prefer cached case tensors when
    available; fall back to the NIfTI files in processed_images.
    """
    target_shape = _as_shape_tuple(target_shape)
    folder_name = os.path.basename(folder_path)

    if cache_dir:
        cache_path = os.path.join(cache_dir, f'{folder_name}.pt')
        if os.path.exists(cache_path):
            try:
                cached = _torch_load_cpu(cache_path)
                if isinstance(cached, dict) and 'volume' in cached:
                    volume = cached['volume'].float()
                    if volume.ndim == 3:
                        volume = volume.unsqueeze(0)
                    volume = _resize_3d_volume(volume, target_shape)
                    return _normalize_3d_channels(volume)
            except Exception as e:
                logging.warning(f"Error loading cached 3D volume {cache_path}: {e}")

    modalities = ['t1_preproc', 't1Gd_preproc', 't2_preproc', 'flair_preproc']
    arrays = []
    ref_shape = None
    for mod in modalities:
        img_path = os.path.join(folder_path, f'{mod}.nii.gz')
        arr = None
        if os.path.exists(img_path):
            try:
                arr = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
                if arr.ndim != 3:
                    arr = None
            except Exception as e:
                logging.warning(f"Error loading 3D MRI volume {img_path}: {e}")
                arr = None
        if arr is not None and ref_shape is None:
            ref_shape = arr.shape
        arrays.append(arr)

    if ref_shape is None:
        ref_shape = target_shape if target_shape is not None else (128, 128, 128)

    channels = []
    for arr in arrays:
        if arr is None:
            channels.append(torch.zeros(ref_shape, dtype=torch.float32))
            continue
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        if tuple(arr.shape) != tuple(ref_shape):
            tensor = F.interpolate(tensor, size=ref_shape, mode='trilinear', align_corners=False)
        channels.append(tensor.squeeze(0).squeeze(0))

    volume = torch.stack(channels, dim=0)
    volume = _resize_3d_volume(volume, target_shape)
    return _normalize_3d_channels(volume)


def _seg_to_3ch(seg_tensor):
    seg_tensor = torch.round(seg_tensor.float()).long()
    return torch.stack(
        [
            (seg_tensor == 0).float(),
            (seg_tensor == 1).float(),
            (seg_tensor == 2).float(),
        ],
        dim=0,
    )


def _load_full_seg_volume(folder_path, cache_dir=None, target_shape=(128, 128, 128)):
    """Load full 3D segmentation as [background, non-enhancing, enhancing]."""
    target_shape = _as_shape_tuple(target_shape)
    folder_name = os.path.basename(folder_path)

    if cache_dir:
        cache_path = os.path.join(cache_dir, f'{folder_name}.pt')
        if os.path.exists(cache_path):
            try:
                cached = _torch_load_cpu(cache_path)
                if isinstance(cached, dict) and 'segmentation' in cached:
                    seg = cached['segmentation']
                    if isinstance(seg, np.ndarray):
                        seg = torch.from_numpy(seg)
                    seg_3ch = _seg_to_3ch(seg)
                    return _resize_3d_volume(seg_3ch, target_shape)
            except Exception as e:
                logging.warning(f"Error loading cached 3D segmentation {cache_path}: {e}")

    seg_path = os.path.join(folder_path, 'seg_preproc.nii.gz')
    if os.path.exists(seg_path):
        try:
            seg = torch.from_numpy(np.asarray(nib.load(seg_path).dataobj, dtype=np.float32))
            seg_3ch = _seg_to_3ch(seg)
            return _resize_3d_volume(seg_3ch, target_shape)
        except Exception as e:
            logging.warning(f"Error loading 3D segmentation {seg_path}: {e}")

    shape = target_shape if target_shape is not None else (128, 128, 128)
    return torch.zeros((3, *shape), dtype=torch.float32)


class _LumiereDataset(Dataset):
    """Inner dataset class for LUMIERE, following the CUB pattern."""

    def __init__(self, csv_path, img_dir, split='train',
                 label_map=None, use_delta_only=False, use_8_channels=True,
                 seed=42, concept_names=None, concepts_3d_path=None,
                 cv_fold=None, cv_n_splits=5, cv_val_fold=None,
                 task_cardinality=4, case_cache_dir=None, full_volume_shape=(128, 128, 128)):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.split = split
        self.label_map = label_map or LABEL_MAP
        self.use_delta_only = use_delta_only
        self.use_8_channels = use_8_channels
        self.use_8_channels = use_8_channels
        self.seed = seed
        self.concept_names = concept_names
        self.concepts_3d_path = concepts_3d_path
        self.cv_fold = cv_fold
        self.cv_n_splits = cv_n_splits
        self.cv_val_fold = cv_val_fold
        self.task_cardinality = int(task_cardinality)
        self.case_cache_dir = case_cache_dir
        self.full_volume_shape = _as_shape_tuple(full_volume_shape)
        if self.task_cardinality not in (2, 4):
            raise ValueError(f"Lumiere task_cardinality must be 2 or 4, got {self.task_cardinality}")

        # Will be populated by update_lists() / preprocessing
        self.X = None
        self.c = None
        self.y = None
        self.graph = {}

        try:
            data = pd.read_csv(csv_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            logging.warning(f"Could not read CSV at {csv_path}")
            data = pd.DataFrame(columns=['Patient', 'Date', 'Rating'])

        if not data.empty and len(data.columns) >= 6:
            data.rename(columns={
                data.columns[0]: 'Patient',
                data.columns[1]: 'Date',
                data.columns[4]: 'Rating',
                data.columns[5]: 'Rating rationale',
            }, inplace=True)

        # Pass 1: Parse rows
        # Seg volumes are loaded lazily per-sample to avoid ~25 GB RAM usage
        temp_metadata = []
        for _, row in data.iterrows():
            patient_id = str(row['Patient']).strip()
            date_id = row['Date']
            rating = row['Rating']

            folder_name = f"{patient_id}_{date_id}"
            folder_path = os.path.join(img_dir, folder_name)
            if not os.path.exists(folder_path):
                continue

            rating_key = str(rating).split(' ')[0]
            if rating_key not in self.label_map:
                continue
            label = self.label_map[rating_key]

            # Week number
            week_match = re.search(r'week-(\d+)', folder_name)
            week_num = int(week_match.group(1)) if week_match else 0

            temp_metadata.append({
                'path': folder_path,
                'label': label,
                'patient_id': patient_id,
                'week': week_num,
            })

        # Build timeline map
        patient_to_indices = defaultdict(list)
        for i, meta in enumerate(temp_metadata):
            patient_to_indices[meta['patient_id']].append(i)

        timeline_map = {}  # scan_idx -> baseline_idx
        for pid, idx_list in patient_to_indices.items():
            sorted_by_week = sorted(idx_list, key=lambda i: temp_metadata[i]['week'])
            baseline_idx = sorted_by_week[0]
            for scan_idx in sorted_by_week:
                timeline_map[scan_idx] = baseline_idx

        # ── Concept computation ────────────────────────────────────────
        self.all_filenames = []
        self.all_labels = []
        self.all_concepts = []  # list of concept vectors (raw continuous values)
        self.all_spd_values = []
        self.all_time_values = []
        self.all_patient_ids = []
        self.timeline_map = {}  # final index -> baseline final index
        self.baseline_path_map = {}  # final index -> baseline folder path (for 3D mode)

        if self.concepts_3d_path is not None:
            # ── 3D-derived concepts from precomputed cache ──────────
            import json as _json
            logging.info(f"Loading precomputed 3D concepts from {self.concepts_3d_path}")
            with open(self.concepts_3d_path) as _f:
                concepts_3d = _json.load(_f)

            # Build folder-name → temp_idx lookup
            folder_name_to_temp = {}
            for i, meta in enumerate(temp_metadata):
                folder_name_to_temp[os.path.basename(meta['path'])] = i

            final_idx_map = {}
            seen_cached_followups = set()
            for i, meta in enumerate(temp_metadata):
                folder_name = os.path.basename(meta['path'])
                if folder_name not in concepts_3d:
                    continue  # skip baseline-only scans not in cache
                if folder_name in seen_cached_followups:
                    continue
                seen_cached_followups.add(folder_name)

                c3d = concepts_3d[folder_name]
                final_idx = len(self.all_filenames)
                final_idx_map[i] = final_idx

                self.all_filenames.append(meta['path'])
                # Use rano_label from cache; optionally remap CR/PR/SD vs PD for binary training.
                label = c3d['rano_label']
                if self.task_cardinality == 2:
                    label = BINARY_LABEL_REMAP[label]
                self.all_labels.append(label)
                self.all_patient_ids.append(meta['patient_id'])
                c3d_raw = c3d['concepts']
                if len(c3d_raw) != N_CONCEPTS:
                    raise ValueError(
                        "The configured concepts_3d.json uses an older concept schema "
                        f"(found {len(c3d_raw)} values, expected {N_CONCEPTS}). "
                        "Re-run compute_3d_concepts.py to regenerate the cache in the new "
                        f"{N_CONCEPTS}-concept explicit-delta format."
                    )
                spd_baseline = float(c3d.get('spd_t', 0.0))
                spd_current = float(c3d.get('spd_t1', spd_baseline))
                c3d_concepts = [float(v) for v in c3d_raw]
                self.all_concepts.append(c3d_concepts)
                self.all_spd_values.append(float(max(0.0, spd_current)))
                self.all_time_values.append(float(c3d.get('time_gap', 0.0)))

                # Store baseline path for image loading (even if baseline not in the final set)
                baseline_name = c3d['baseline_case']
                baseline_temp = folder_name_to_temp.get(baseline_name)
                if baseline_temp is not None:
                    self.baseline_path_map[final_idx] = temp_metadata[baseline_temp]['path']
                    if baseline_temp in final_idx_map:
                        self.timeline_map[final_idx] = final_idx_map[baseline_temp]
                    else:
                        self.timeline_map[final_idx] = final_idx  # baseline not in final set
                else:
                    self.timeline_map[final_idx] = final_idx  # baseline folder missing

            logging.info(f"Loaded 3D concepts for {len(self.all_filenames)} samples "
                         f"(skipped {len(temp_metadata) - len(self.all_filenames)} baseline-only scans)")

            del temp_metadata, final_idx_map
            gc.collect()

        else:
            # ── Original NIfTI-based concept computation ────────────
            # Pass 2a: Compute per-scan seg volumes + SPD
            logging.info("Computing volumetric concepts + SPD from segmentation masks...")
            scan_volumes = {}  # temp_idx -> (enh_vol, non_enh_vol, n_enh_components, spd_cm2)
            for i, meta in enumerate(temp_metadata):
                enh_vol, non_enh_vol, spd_cm2, seg_data = _load_seg_volumes(meta['path'])
                n_enh_comp = 0
                if seg_data is not None:
                    _, n_enh_comp = connected_components(seg_data == 2)
                scan_volumes[i] = (enh_vol, non_enh_vol, n_enh_comp, spd_cm2)
                del seg_data  # free immediately
                if (i + 1) % 50 == 0:
                    logging.info(f"  Processed {i + 1}/{len(temp_metadata)} seg volumes")
                    gc.collect()
            logging.info(f"Computed volumes for {len(scan_volumes)} scans.")

            # Pass 2b: Derive volumetric concepts
            final_idx_map = {}  # temp_idx -> final_idx

            for i, meta in enumerate(temp_metadata):
                final_idx = len(self.all_filenames)
                final_idx_map[i] = final_idx

                self.all_filenames.append(meta['path'])
                label = meta['label']
                if self.task_cardinality == 2:
                    label = BINARY_LABEL_REMAP[label]
                self.all_labels.append(label)
                self.all_patient_ids.append(meta['patient_id'])

                baseline_temp_idx = timeline_map[i]
                curr_enh, curr_ne, curr_comp, curr_spd = scan_volumes[i]
                base_enh, base_ne, base_comp, base_spd = scan_volumes[baseline_temp_idx]

                new_lesion    = 1.0 if (curr_comp > base_comp) else 0.0

                # Time gap: weeks from baseline to current scan, normalized by 52
                curr_week = meta['week']
                base_week = temp_metadata[baseline_temp_idx]['week']
                time_gap = max(0.0, float(curr_week - base_week)) / 52.0

                c_vals = _build_explicit_concept_vector(
                    base_enh=base_enh,
                    base_ne=base_ne,
                    followup_enh=curr_enh,
                    followup_ne=curr_ne,
                    new_lesion_flag=new_lesion,
                    base_spd=base_spd,
                    followup_spd=curr_spd,
                    time_gap=time_gap,
                )
                self.all_concepts.append(c_vals)
                self.all_spd_values.append(float(curr_spd))
                self.all_time_values.append(float(time_gap))

            del scan_volumes

            # Rebuild timeline_map with final indices
            for temp_idx, final_idx in final_idx_map.items():
                baseline_temp_idx = timeline_map[temp_idx]
                self.timeline_map[final_idx] = final_idx_map[baseline_temp_idx]

            # Free temp structures
            del temp_metadata, final_idx_map, timeline_map
            gc.collect()

        # Filter concepts if requested
        # Filter concepts if requested
        if self.concept_names is not None:
            # Validate names but do NOT filter self.all_concepts here
            # We filter in _binarize_concepts/update_lists to keep raw data intact
            self.valid_concept_indices = [CONCEPT_NAMES.index(n) for n in self.concept_names if n in CONCEPT_NAMES]
            if len(self.valid_concept_indices) < len(self.concept_names):
                logging.warning(f"Some requested concepts not found. Using {len(self.valid_concept_indices)} found.")
            self.n_concepts = len(self.valid_concept_indices)
        else:
            self.valid_concept_indices = None
            self.n_concepts = N_CONCEPTS
        self.n_classes = self.task_cardinality

        # Patient-level StratifiedGroupKFold splitting. Each fold holds out
        # whole patients by grouping on patient_id while stratifying on scan labels.
        labels_arr = np.array(self.all_labels)
        patient_ids_arr = np.array(self.all_patient_ids)
        indices_arr = np.arange(len(self.all_labels))

        n_splits = int(cv_n_splits or 5)
        if n_splits < 2:
            raise ValueError(f"cv_n_splits must be at least 2, got {n_splits}")

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(sgkf.split(indices_arr, labels_arr, groups=patient_ids_arr))

        test_fold = 0 if cv_fold is None else int(cv_fold)
        if test_fold < 0 or test_fold >= n_splits:
            raise ValueError(f"cv_fold must be in [0, {n_splits - 1}], got {test_fold}")

        val_fold = (test_fold + 1) % n_splits if cv_val_fold is None else int(cv_val_fold)
        if val_fold < 0 or val_fold >= n_splits:
            raise ValueError(f"cv_val_fold must be in [0, {n_splits - 1}], got {val_fold}")
        test_indices = sorted(int(i) for i in folds[test_fold][1])
        val_indices = sorted(int(i) for i in folds[val_fold][1])
        heldout = set(test_indices) | set(val_indices)
        train_indices = [int(i) for i in indices_arr if int(i) not in heldout]

        self.cv_fold = test_fold
        self.cv_val_fold = val_fold
        self.cv_n_splits = n_splits
        self.split_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
        }

        train_patients = set(patient_ids_arr[train_indices])
        val_patients = set(patient_ids_arr[val_indices])
        test_patients = set(patient_ids_arr[test_indices])
        assert len(train_patients & val_patients) == 0, "PATIENT LEAKAGE DETECTED between train and val"
        assert len(train_patients & test_patients) == 0, "PATIENT LEAKAGE DETECTED between train and test"
        assert (train_patients | val_patients | test_patients) == set(patient_ids_arr), "Not all patients assigned to folds"

        logging.info(f"LUMIERE StratifiedGroupKFold split: fold={test_fold + 1}/{n_splits}, "
                     f"val_fold={val_fold + 1}/{n_splits}, "
                     f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    def compute_thresholds(self, train_indices):
        """Compute per-concept medians from training data for binarization,
        and mean/std for z-score normalization of continuous concepts."""
        train_raw = np.array([self.all_concepts[i] for i in train_indices])
        self.thresholds = np.median(train_raw, axis=0)  # shape [N_CONCEPTS]
        # Compute mean and std for z-score normalization of continuous concepts
        self.concept_means = np.mean(train_raw, axis=0)
        self.concept_stds = np.std(train_raw, axis=0)
        self.concept_stds[self.concept_stds < 1e-8] = 1.0  # avoid div by zero
        logging.info(f"Computed binarization thresholds (medians): {self.thresholds}")
        logging.info(f"Computed concept means: {self.concept_means}")
        logging.info(f"Computed concept stds: {self.concept_stds}")

    def compute_concept_class_weights(self, train_indices):
        """Compute class-balanced weights for binary concepts.
        Continuous concepts get dummy [1.0, 1.0] padding to keep the tensor rectangular."""
        indices = self.valid_concept_indices if self.valid_concept_indices is not None \
                  else list(range(N_CONCEPTS))

        # Only need to inspect binary concepts
        concepts_array = np.array([
            self._process_concepts(torch.tensor(self.all_concepts[i], dtype=torch.float32)).numpy()
            for i in train_indices
        ])
        self.concept_weights = []
        max_card = 3  # pad all weight lists to this length for rectangular tensor
        for j, ci in enumerate(indices):
            name = CONCEPT_NAMES[ci]
            card = _concept_cardinality(name)
            if card > 1:
                # Categorical concept: compute inverse-frequency class weights
                col = concepts_array[:, j]
                n_total = len(col)
                class_weights = []
                for cls_idx in range(card):
                    n_cls = float((col == cls_idx).sum())
                    class_weights.append(float(n_total / (card * max(n_cls, 1))))
                # Pad to max_card
                class_weights += [0.0] * (max_card - card)
                self.concept_weights.append(class_weights)
            else:
                # Continuous: dummy weights (never used — MSE loss path ignores them)
                self.concept_weights.append([1.0] * max_card)
        logging.info(f"Concept class weights: {self.concept_weights}")

    @property
    def _indices(self):
        return self.split_indices.get(self.split, [])

    def __len__(self):
        return len(self._indices)

    def register_graph(self, graph):
        self.graph = graph

    def _get_binary_mask(self):
        """Return list of bools: True for categorical concepts (card>1), False for continuous."""
        indices = self.valid_concept_indices if self.valid_concept_indices is not None \
                  else list(range(N_CONCEPTS))
        return [CONCEPT_NAMES[i] in CATEGORICAL_CONCEPTS for i in indices]

    def _process_concepts(self, raw_concepts):
        """Process concepts:
        - Continuous (cardinality=1): z-score normalize
        - Binary (cardinality=2): preserve their explicit 0/1 semantics
        - RANO 3-class (cardinality=3): encode via RANO thresholds
        """
        thresholds = getattr(self, 'thresholds', None)
        means = getattr(self, 'concept_means', None)
        stds = getattr(self, 'concept_stds', None)
        indices = self.valid_concept_indices if self.valid_concept_indices is not None \
                  else list(range(N_CONCEPTS))

        processed = []
        for i in indices:
            val = raw_concepts[i]
            name = CONCEPT_NAMES[i]
            if name in RANO_CONCEPTS:
                # 3-class RANO encoding from raw percentage
                processed.append(_encode_rano_class(float(val)))
            elif name in BINARY_CONCEPTS:
                # Already-binary clinical flags. Keep their 0/1 semantics
                # instead of median-thresholding, which can collapse a fold.
                processed.append(1.0 if float(val) > 0.5 else 0.0)
            else:
                # Continuous: z-score normalise
                m = float(means[i]) if means is not None else 0.0
                s = float(stds[i]) if stds is not None else 1.0
                processed.append((float(val) - m) / s)
        return torch.tensor(processed, dtype=torch.float32)

    def _binarize_concepts(self, raw_concepts):
        """Legacy: still available for backward compat. Now calls _process_concepts."""
        return self._process_concepts(raw_concepts)

    def _clinical_features(self, global_idx):
        return torch.tensor([
            math.log1p(max(0.0, float(self.all_spd_values[global_idx]))),
            max(0.0, float(self.all_time_values[global_idx])),
        ], dtype=torch.float32)

    def update_lists(self):
        """Pre-compute c and y tensors for all samples in the current split."""
        indices = self._indices
        c_list = []
        y_list = []
        import time
        start_time = time.time()
        print(f"Starting update_lists for split {self.split} with {len(indices)} samples")
        for i, idx in enumerate(indices):
            raw = torch.tensor(self.all_concepts[idx], dtype=torch.float32)
            c_list.append(self._process_concepts(raw))
            y_list.append(torch.tensor(self.all_labels[idx]))
            if i % 50 == 0 and i > 0:
                print(f"Processed {i}/{len(indices)} samples in {time.time() - start_time:.2f} seconds")

        self.c = torch.stack(c_list, dim=0).float()  # float for mixed continuous/binary
        self.y = torch.stack(y_list, dim=0).unsqueeze(-1).int()

    def __getitem__(self, idx):
        """Returns {'x': ..., 'c': concept_tensor, 'y': label_tensor, 'graph': graph}

        When self.X is not None, precomputed embeddings are returned directly.
        """
        global_idx = self._indices[idx]
        folder_path = self.all_filenames[global_idx]
        label = self.all_labels[global_idx]
        raw_concepts = torch.tensor(self.all_concepts[global_idx], dtype=torch.float32)
        concepts = self._process_concepts(raw_concepts)

        # Already-encoded embeddings
        if self.X is not None:
            return {
                'x': self.X[idx],
                'c': concepts,
                'y': torch.tensor(label),
                'graph': self.graph,
            }

        mode = getattr(self, 'input_mode', 'mri')

        if mode == 'radiomic':
            means = torch.tensor(self.concept_means, dtype=torch.float32)
            stds  = torch.tensor(self.concept_stds,  dtype=torch.float32)
            x = (raw_concepts - means) / stds
            return {
                'x': x,
                'c': concepts,
                'y': torch.tensor(label),
                'graph': self.graph,
            }

        elif mode == 'seg':
            seg_curr = _load_seg_3ch(folder_path)
            baseline_global = self.timeline_map.get(global_idx, global_idx)
            baseline_path_map = getattr(self, 'baseline_path_map', {})
            if global_idx in baseline_path_map:
                seg_base = _load_seg_3ch(baseline_path_map[global_idx])
            elif baseline_global != global_idx:
                seg_base = _load_seg_3ch(self.all_filenames[baseline_global])
            else:
                seg_base = seg_curr.clone()
            return {
                'x': seg_curr,
                'x_baseline': seg_base,
                'c': concepts,
                'y': torch.tensor(label),
                'graph': self.graph,
            }

        elif mode in ['m3d_lamed', 'medicalnet']:
            current_vol = _load_full_mri_volume(
                folder_path,
                cache_dir=getattr(self, 'case_cache_dir', None),
                target_shape=getattr(self, 'full_volume_shape', (128, 128, 128)),
            )
            current_seg = _load_full_seg_volume(
                folder_path,
                cache_dir=getattr(self, 'case_cache_dir', None),
                target_shape=getattr(self, 'full_volume_shape', (128, 128, 128)),
            )
            baseline_global = self.timeline_map.get(global_idx, global_idx)
            baseline_path_map = getattr(self, 'baseline_path_map', {})
            if global_idx in baseline_path_map:
                baseline_path = baseline_path_map[global_idx]
            elif baseline_global != global_idx:
                baseline_path = self.all_filenames[baseline_global]
            else:
                baseline_path = None
            if baseline_path is not None:
                baseline_vol = _load_full_mri_volume(
                    baseline_path,
                    cache_dir=getattr(self, 'case_cache_dir', None),
                    target_shape=getattr(self, 'full_volume_shape', (128, 128, 128)),
                )
                baseline_seg = _load_full_seg_volume(
                    baseline_path,
                    cache_dir=getattr(self, 'case_cache_dir', None),
                    target_shape=getattr(self, 'full_volume_shape', (128, 128, 128)),
                )
            else:
                baseline_vol = current_vol.clone()
                baseline_seg = current_seg.clone()
            return {
                'x': current_vol,
                'x_baseline': baseline_vol,
                'x_seg_curr': current_seg,
                'x_seg_base': baseline_seg,
                'clinical_features': self._clinical_features(global_idx),
                'c': concepts,
                'y': torch.tensor(label),
                'graph': self.graph,
            }

        elif mode == 'seg_guided' or mode == 'seg_guided_no_temporal':
            best_slice = _get_best_slice_idx(folder_path)
            current_mri = _load_4channel(folder_path, best_slice)
            current_seg = _load_seg_3ch(folder_path)
            baseline_global = self.timeline_map.get(global_idx, global_idx)
            # Use baseline_path_map if available (3D concept mode)
            baseline_path_map = getattr(self, 'baseline_path_map', {})
            if global_idx in baseline_path_map:
                baseline_path = baseline_path_map[global_idx]
            elif baseline_global != global_idx:
                baseline_path = self.all_filenames[baseline_global]
            else:
                baseline_path = None
            if baseline_path is not None:
                baseline_mri = _load_4channel(baseline_path, best_slice)
                baseline_seg = _load_seg_3ch(baseline_path)
            else:
                baseline_mri = current_mri.clone()
                baseline_seg = current_seg.clone()
            return {
                'x': current_mri,
                'x_baseline': baseline_mri,
                'x_seg_curr': current_seg,
                'x_seg_base': baseline_seg,
                'c': concepts,
                'y': torch.tensor(label),
                'graph': self.graph,
            }

        else:  # 'mri'
            best_slice = _get_best_slice_idx(folder_path)
            current_img = _load_4channel(folder_path, best_slice)
            baseline_global = self.timeline_map.get(global_idx, global_idx)
            baseline_path_map = getattr(self, 'baseline_path_map', {})
            if global_idx in baseline_path_map:
                baseline_img = _load_4channel(baseline_path_map[global_idx], best_slice)
            elif baseline_global != global_idx:
                baseline_img = _load_4channel(self.all_filenames[baseline_global], best_slice)
            else:
                baseline_img = current_img.clone()
            return {
                'x': current_img,
                'x_baseline': baseline_img,
                'c': concepts,
                'y': torch.tensor(label),
                'graph': self.graph,
            }




class LumiereDataset:
    """Outer container class following the CUB/CelebA pattern."""

    def __init__(self,
                 ftune_size: float = 0.,
                 ftune_val_size: float = 0.1,
                 task_label: str = 'TreatmentResponse',
                 task_cardinality: int = 4,
                 to_keep: dict = None,
                 root_dir: str = '/home/group2/dataset/Lumiere/',
                 img_dir: str = '/home/group2/dataset/processed_images/',
                 use_delta_only: bool = False,
                 use_8_channels: bool = True,
                 causal_struct: bool = False,
                 use_validated_set: bool = True,
                 input_mode: str = 'mri',
                 concepts_3d_path: str = None,
                 case_cache_dir: str = None,
                 full_volume_shape = (128, 128, 128),
                 graph_type: str = 'expert',
                 cv_fold: int = None,
                 cv_n_splits: int = 5,
                 cv_val_fold: int = None):

        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.use_delta_only = use_delta_only
        self.use_8_channels = use_8_channels
        self.causal_struct = causal_struct
        self._input_mode = input_mode
        self._concepts_3d_path = concepts_3d_path
        self._case_cache_dir = case_cache_dir
        self._full_volume_shape = _as_shape_tuple(full_volume_shape)
        self._graph_type = str(graph_type).lower()
        self._cv_fold = cv_fold
        self._cv_n_splits = cv_n_splits
        self._cv_val_fold = cv_val_fold

        self.get_dataset = _LumiereDataset
        self.use_validated_set = use_validated_set

        self.c_info = {
            'names': list(CONCEPT_NAMES),
            'cardinality': [_concept_cardinality(n) for n in CONCEPT_NAMES],
        }

        self.y_info = {
            'names': [task_label],
            'cardinality': [task_cardinality],
        }

        self.data = {}
        self.to_keep = to_keep

    def load_ground_truth_graph(self):
        """Create an expert-specified causal DAG over the validated concepts + task.

        The expert layout keeps the graph clinically sparse:
        baseline/follow-up measurements feed only derived delta metrics, delta
        percentages feed threshold flags, and TreatmentResponse depends mainly
        on the clinically meaningful flags plus new_lesion_flag and time_gap.
        If using a different concept subset, we fall back to a simple
        concept-to-task graph.
        """
        node_labels = self.c_info['names'] + self.y_info['names']
        n = len(node_labels)
        adj_matrix = np.zeros((n, n), dtype=int)
        
        task_idx = n - 1

        def _make_bipartite():
            for i in range(n - 1):
                adj_matrix[i, task_idx] = 1
        
        # Check if we are using exactly the validated explicit-delta set.
        # If not, fallback to a simple concept-to-task graph.
        if self._graph_type in {'bipartite', 'concept_to_task', 'concept-to-task'}:
            _make_bipartite()
        elif self._graph_type != 'expert':
            raise ValueError(
                f"Unsupported Lumiere graph_type={self._graph_type!r}. "
                "Use 'expert' or 'bipartite'."
            )
        elif set(self.c_info['names']) == set(VALIDATED_CONCEPTS):
            try:
                idx_base_enh = node_labels.index('enhancing_tumor_volume_cm3')
                idx_base_ne = node_labels.index('non_enhancing_volume_cm3')
                idx_fup_enh = node_labels.index('followup_enhancing_volume_cm3')
                idx_fup_ne = node_labels.index('followup_non_enhancing_volume_cm3')
                idx_base_spd = node_labels.index('baseline_spd_cm2')
                idx_fup_spd = node_labels.index('followup_spd_cm2')
                idx_time_gap = node_labels.index('time_gap')
                idx_d_enh_abs = node_labels.index('delta_enhancing_absolute')
                idx_d_enh_pct = node_labels.index('delta_enhancing_percent')
                idx_d_ne_abs = node_labels.index('delta_non_enhancing_absolute')
                idx_d_ne_pct = node_labels.index('delta_non_enhancing_percent')
                idx_d_spd_abs = node_labels.index('delta_spd_absolute')
                idx_d_spd_pct = node_labels.index('delta_spd_percent')
                idx_new_les = node_labels.index('new_lesion_flag')
                idx_vol_pd = node_labels.index('vol_pd_flag')
                idx_vol_pr = node_labels.index('vol_pr_flag')
                idx_spd_pd = node_labels.index('spd_pd_flag')
                idx_spd_pr = node_labels.index('spd_pr_flag')

                # Static measurements drive explicit deltas.
                for parent in (idx_base_enh, idx_fup_enh):
                    adj_matrix[parent, idx_d_enh_abs] = 1
                    adj_matrix[parent, idx_d_enh_pct] = 1
                for parent in (idx_base_ne, idx_fup_ne):
                    adj_matrix[parent, idx_d_ne_abs] = 1
                    adj_matrix[parent, idx_d_ne_pct] = 1
                for parent in (idx_base_spd, idx_fup_spd):
                    adj_matrix[parent, idx_d_spd_abs] = 1
                    adj_matrix[parent, idx_d_spd_pct] = 1

                # Threshold flags are derived from the corresponding delta percentages.
                for child in (idx_vol_pd, idx_vol_pr):
                    adj_matrix[idx_d_enh_pct, child] = 1

                # SPD threshold flags are likewise derived from SPD percent change.
                for child in (idx_spd_pd, idx_spd_pr):
                    adj_matrix[idx_d_spd_pct, child] = 1

                # Treatment response depends on:
                #   - metadata + new lesion (independent PD criterion)
                #   - the four RANO threshold flags (clean cases when they fire)
                #   - the percent-change magnitudes (graded signal below/above thresholds)
                # Baseline volumes are intentionally NOT direct parents of Y:
                # their embeddings would re-open a 128-dim leakage channel; the
                # deterministic flag formulas already gate on measurability internally.
                for idx in [
                    idx_time_gap, idx_new_les,
                    idx_vol_pd, idx_vol_pr, idx_spd_pd, idx_spd_pr,
                    idx_d_enh_pct, idx_d_ne_pct, idx_d_spd_pct,
                ]:
                    adj_matrix[idx, task_idx] = 1
            except ValueError:
                # Fallback to bipartite
                _make_bipartite()
        else:
            # Fallback to bipartite
            _make_bipartite()

        adj_pandas = pd.DataFrame(adj_matrix.astype(int),
                                  index=node_labels,
                                  columns=node_labels)
        self.adj = adj_pandas
        return self.adj

    def split(self):
        """Create train/val/test partitions."""
        csv_path = os.path.join(self.root_dir, 'ratings.csv')

        # Concept names for label descriptions
        if self.to_keep is not None:
            concept_names = list(self.to_keep.keys())
        elif self.use_validated_set:
            concept_names = list(VALIDATED_CONCEPTS)
        else:
            concept_names = list(CONCEPT_NAMES)
        self.c_info['names'] = concept_names
        # Mixed cardinality: binary flags → 2, continuous volumes/deltas → 1
        self.c_info['cardinality'] = [_concept_cardinality(n) for n in concept_names]

        # Create one full dataset — it handles splitting internally
        full_ds = _LumiereDataset(
            csv_path=csv_path,
            img_dir=self.img_dir,
            split='train',
            use_delta_only=self.use_delta_only,
            use_8_channels=self.use_8_channels,
            concept_names=concept_names,
            concepts_3d_path=self._concepts_3d_path,
            case_cache_dir=self._case_cache_dir,
            full_volume_shape=self._full_volume_shape,
            cv_fold=self._cv_fold,
            cv_n_splits=self._cv_n_splits,
            cv_val_fold=self._cv_val_fold,
            task_cardinality=self.y_info['cardinality'][0],
        )

        # Compute thresholds on training split and propagate
        full_ds.compute_thresholds(full_ds.split_indices['train'])
        full_ds.compute_concept_class_weights(full_ds.split_indices['train'])

        # Create split-specific views
        for split_name in ['train', 'val', 'test']:
            ds = _LumiereDataset.__new__(_LumiereDataset)
            # Copy shared state
            ds.csv_path = full_ds.csv_path
            ds.img_dir = full_ds.img_dir
            ds.label_map = full_ds.label_map
            ds.use_delta_only = full_ds.use_delta_only
            ds.use_8_channels = full_ds.use_8_channels
            ds.seed = full_ds.seed
            ds.concept_names = full_ds.concept_names
            ds.concepts_3d_path = full_ds.concepts_3d_path
            ds.cv_fold = full_ds.cv_fold
            ds.cv_n_splits = full_ds.cv_n_splits
            ds.cv_val_fold = full_ds.cv_val_fold
            ds.task_cardinality = full_ds.task_cardinality
            ds.case_cache_dir = full_ds.case_cache_dir
            ds.full_volume_shape = full_ds.full_volume_shape
            ds.valid_concept_indices = full_ds.valid_concept_indices
            ds.X = None
            ds.c = None
            ds.y = None
            ds.graph = {}
            ds.all_filenames = full_ds.all_filenames
            ds.all_labels = full_ds.all_labels
            ds.all_concepts = full_ds.all_concepts
            ds.all_spd_values = full_ds.all_spd_values
            ds.all_time_values = full_ds.all_time_values
            ds.all_patient_ids = full_ds.all_patient_ids
            ds.timeline_map = full_ds.timeline_map
            ds.baseline_path_map = full_ds.baseline_path_map
            ds.n_concepts = full_ds.n_concepts
            ds.n_classes = full_ds.n_classes
            ds.split_indices = full_ds.split_indices
            ds.thresholds = full_ds.thresholds  # Propagate thresholds
            ds.concept_means = full_ds.concept_means  # Propagate z-score means
            ds.concept_stds = full_ds.concept_stds    # Propagate z-score stds
            ds.concept_weights = full_ds.concept_weights  # Propagate weights
            ds.input_mode = getattr(self, '_input_mode', 'mri')
            ds.split = split_name
            self.data[split_name] = ds

        # Fine-tune split from test if requested
        if self.ftune_size > 0:
            self.data['test'], self.data['ftune'] = split_dataset(
                self.data['test'], self.ftune_size
            )
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(
                self.data['ftune'], self.ftune_val_size
            )
            self.data['ftune_val'].split_type = 'ftune_val'
