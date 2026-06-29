"""Compute the 18-concept explicit-delta schema from cached case volumes.

Reads ``pair_index.json`` from the cached MNI pair directory, loads
baseline/follow-up case tensors, and derives the same explicit-delta concept
layout used by the reference Siamese project:

1.  baseline enhancing volume          (log1p cm^3)
2.  baseline non-enhancing volume      (log1p cm^3)
3.  follow-up enhancing volume         (log1p cm^3)
4.  follow-up non-enhancing volume     (log1p cm^3)
5.  baseline SPD                       (log1p cm^2)
6.  follow-up SPD                      (log1p cm^2)
7.  time gap                           (weeks / 52)
8.  delta enhancing absolute           (cm^3)
9.  delta enhancing relative           ((t+1 - t) / t)
10. delta non-enhancing absolute       (cm^3)
11. delta non-enhancing relative       ((t+1 - t) / t)
12. delta SPD absolute                 (cm^2)
13. delta SPD relative                 ((t+1 - t) / t)
14. new lesion flag                    (0 / 1)
15. volumetric PD flag                 (0 / 1)
16. volumetric PR flag                 (0 / 1)
17. SPD PD flag                        (0 / 1)
18. SPD PR flag                        (0 / 1)
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


DEFAULT_CACHE_CANDIDATES = [
    Path(
        "/home/group2/youssef/XAI_Disease_progression_GP/Label-free-CBM/"
        "option4-siamese-explicit-delta/cached_data_MNI"
    ),
    Path("/home/group2/Hamza_new/lumiere_case_cache"),
    Path(__file__).resolve().parent / "cached_data_MNI",
]
OUT_PATH = Path(__file__).resolve().parent / "concepts_3d.json"

DELTA_EPSILON = 1e-4
MIN_MEASURABLE_ENHANCING_CM3 = 0.5
MIN_MEASURABLE_SPD_CM2 = 0.01


def _discover_cache_dir():
    for candidate in DEFAULT_CACHE_CANDIDATES:
        if (candidate / "pair_index.json").exists():
            return candidate
    searched = ", ".join(str(p) for p in DEFAULT_CACHE_CANDIDATES)
    raise FileNotFoundError(f"Could not find pair_index.json in any candidate cache dir: {searched}")


CACHE_DIR = _discover_cache_dir()


def _clamped_relative_change(followup_val, baseline_val, epsilon=DELTA_EPSILON):
    baseline_val = float(baseline_val)
    followup_val = float(followup_val)
    raw_delta = (followup_val - baseline_val) / (baseline_val + float(epsilon))
    return float(np.clip(raw_delta, -1.0, 5.0))


def _resolve_cache_path(raw_path):
    raw_path = Path(raw_path)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.parts and raw_path.parts[0] == CACHE_DIR.name:
        return (CACHE_DIR.parent / raw_path).resolve()
    return (CACHE_DIR / raw_path).resolve()


def _extract_case_measurements(case_data):
    if "volumes" in case_data:
        base_enh = float(case_data["volumes"][0])
        base_ne = float(case_data["volumes"][1])
    else:
        base_enh = float(case_data.get("enhancing_volume", 0.0))
        base_ne = float(case_data.get("non_enhancing_volume", 0.0))
    base_spd = float(case_data.get("spd", 0.0))
    return base_enh, base_ne, base_spd


def compute_concepts_for_pair(pair_data, baseline_data, followup_data):
    """Compute the 18 explicit-delta concept values for one pair."""
    base_enh, base_ne, spd_base_from_case = _extract_case_measurements(baseline_data)
    fup_enh, fup_ne, spd_fup_from_case = _extract_case_measurements(followup_data)

    delta_enh_abs = fup_enh - base_enh
    delta_ne_abs = fup_ne - base_ne
    delta_enh_pct = _clamped_relative_change(fup_enh, base_enh)
    delta_ne_pct = _clamped_relative_change(fup_ne, base_ne)
    delta_spd_abs = float(pair_data.get("spd_t1", spd_fup_from_case)) - float(pair_data.get("spd_t", spd_base_from_case))

    new_lesion = float(pair_data["new_lesion_flag"])
    spd_base = float(pair_data.get("spd_t", spd_base_from_case))
    spd_fup = float(pair_data.get("spd_t1", spd_fup_from_case))
    delta_spd_pct = _clamped_relative_change(spd_fup, spd_base)
    time_gap = max(0.0, float(pair_data.get("time_gap", 0.0)))

    measurable_enh = base_enh >= MIN_MEASURABLE_ENHANCING_CM3
    measurable_spd = spd_base >= MIN_MEASURABLE_SPD_CM2

    vol_pd_flag = float((delta_enh_pct >= 0.40) and measurable_enh)
    vol_pr_flag = float((delta_enh_pct <= -0.65) and measurable_enh)
    spd_pd_flag = float((delta_spd_pct >= 0.25) and measurable_spd)
    spd_pr_flag = float((delta_spd_pct <= -0.50) and measurable_spd)

    return [
        math.log1p(max(0.0, base_enh)),
        math.log1p(max(0.0, base_ne)),
        math.log1p(max(0.0, fup_enh)),
        math.log1p(max(0.0, fup_ne)),
        math.log1p(max(0.0, spd_base)),
        math.log1p(max(0.0, spd_fup)),
        time_gap,
        float(delta_enh_abs),
        float(delta_enh_pct),
        float(delta_ne_abs),
        float(delta_ne_pct),
        float(delta_spd_abs),
        float(delta_spd_pct),
        float(new_lesion > 0.5),
        vol_pd_flag,
        vol_pr_flag,
        spd_pd_flag,
        spd_pr_flag,
    ]


def main():
    with open(CACHE_DIR / "pair_index.json") as f:
        pairs = json.load(f)

    concepts_map = {}
    label_counter = Counter()

    for i, pair_info in enumerate(pairs):
        pair_data = torch.load(_resolve_cache_path(pair_info["cache_file"]), weights_only=False)
        baseline_data = torch.load(_resolve_cache_path(pair_data["baseline_cache"]), weights_only=False)
        followup_data = torch.load(_resolve_cache_path(pair_data["followup_cache"]), weights_only=False)

        concepts = compute_concepts_for_pair(pair_data, baseline_data, followup_data)
        followup_case = pair_data.get("followup_case") or pair_data.get("current_case")
        if followup_case is None:
            raise KeyError("Pair cache entry is missing both 'followup_case' and 'current_case'.")

        concepts_map[followup_case] = {
            "concepts": concepts,
            "rano_label": int(pair_data["rano_label"]),
            "patient_id": pair_data["patient_id"],
            "baseline_case": pair_data["baseline_case"],
            "followup_case": followup_case,
            "time_gap": float(pair_data.get("time_gap", 0.0)),
            "spd_t": float(pair_data.get("spd_t", 0.0)),
            "spd_t1": float(pair_data.get("spd_t1", 0.0)),
        }
        label_counter[pair_data["rano_label"]] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs")

    print(f"\nTotal pairs processed: {len(concepts_map)}")
    print(f"Label distribution: { {k: label_counter[k] for k in sorted(label_counter)} }")

    sample_keys = list(concepts_map.keys())[:3]
    for key in sample_keys:
        sample = concepts_map[key]
        print(
            f"  {key}: rano={sample['rano_label']}, "
            f"concepts={[round(c, 3) for c in sample['concepts']]}"
        )

    with open(OUT_PATH, "w") as f:
        json.dump(concepts_map, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
