"""
Rule-based RANO oracle baseline.

Applies the textbook RANO decision tree directly to the **ground-truth concept
vector** of every (baseline, follow-up) pair in each validation fold:

    PD  if  new_lesion_flag or vol_pd_flag or spd_pd_flag
    CR  elif followup_enhancing_vol < cr_thresh AND followup_non_enh < cr_thresh
        AND vol_pr_flag (or spd_pr_flag)
    PR  elif vol_pr_flag or spd_pr_flag
    SD  otherwise

If RANO labels were perfectly recoverable from segmentation thresholds, this
oracle would score 1.0 macro-F1. Whatever it does score is the *upper bound*
of what any "just compute the difference" approach can achieve, given the
exact same concept inputs the C2BM uses.

Reports per-fold and pooled 4-class and binary macro/weighted F1 + accuracy,
plus a confusion matrix. Writes a JSON summary alongside.

Usage:
    python rule_based_rano_oracle.py \
        --run-dir outputs/2026-05-14/10-30-17 \
        --out paper_figures/rule_based_oracle_results.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import math
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data.dataset_block import get_dataset
from src.utils import clean_empty_configs


LABEL_NAMES = {0: "CR", 1: "PR", 2: "SD", 3: "PD"}
BINARY_REMAP = {0: 0, 1: 0, 2: 0, 3: 1}  # CR/PR/SD -> Non-PD, PD -> PD

# Concept indices in dataset.all_concepts (matches _build_explicit_concept_vector):
IDX_LOG_FOLLOWUP_ENH = 2
IDX_LOG_FOLLOWUP_NE = 3
IDX_NEW_LESION = 13
IDX_VOL_PD = 14
IDX_VOL_PR = 15
IDX_SPD_PD = 16
IDX_SPD_PR = 17

CR_VOLUME_THRESH_CM3 = 0.10  # both enhancing AND non-enhancing must be ~zero


def _load_cfg(run_dir: str, fold_idx: int):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    cfg = clean_empty_configs(cfg)
    with open_dict(cfg):
        cfg.device = "cpu"
        cfg.dataset.loader.cv_fold = fold_idx
        cfg.dataset.loader.cv_n_splits = 5
        cfg.dataset.loader.cv_val_fold = fold_idx
    return cfg


def _rano_decision(c_vec) -> int:
    new_lesion = c_vec[IDX_NEW_LESION] >= 0.5
    vol_pd = c_vec[IDX_VOL_PD] >= 0.5
    vol_pr = c_vec[IDX_VOL_PR] >= 0.5
    spd_pd = c_vec[IDX_SPD_PD] >= 0.5
    spd_pr = c_vec[IDX_SPD_PR] >= 0.5

    if new_lesion or vol_pd or spd_pd:
        return 3  # PD

    # recover raw follow-up volumes (concept stored as log1p(cm3))
    follow_enh = math.expm1(max(0.0, float(c_vec[IDX_LOG_FOLLOWUP_ENH])))
    follow_ne = math.expm1(max(0.0, float(c_vec[IDX_LOG_FOLLOWUP_NE])))

    if (vol_pr or spd_pr) and follow_enh < CR_VOLUME_THRESH_CM3 and follow_ne < CR_VOLUME_THRESH_CM3:
        return 0  # CR
    if vol_pr or spd_pr:
        return 1  # PR
    return 2  # SD


def _evaluate_fold(run_dir: str, fold_idx: int) -> Dict[str, Any]:
    cfg = _load_cfg(run_dir, fold_idx)
    dataset, _, _ = get_dataset(cfg)
    ds_val = dataset.data["val"]

    y_true: List[int] = []
    y_pred: List[int] = []
    pids: List[str] = []
    for local_idx in range(len(ds_val)):
        global_idx = ds_val._indices[local_idx]
        c_vec = ds_val.all_concepts[global_idx]
        gt_label = int(ds_val.all_labels[global_idx])
        pred_label = _rano_decision(c_vec)
        y_true.append(gt_label)
        y_pred.append(pred_label)
        pids.append(ds_val.all_patient_ids[global_idx])

    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    y_true_b = np.array([BINARY_REMAP[v] for v in y_true])
    y_pred_b = np.array([BINARY_REMAP[v] for v in y_pred])

    return {
        "fold": fold_idx + 1,
        "n": len(y_true),
        "y_true_4": y_true,
        "y_pred_4": y_pred,
        "y_true_2": y_true_b.tolist(),
        "y_pred_2": y_pred_b.tolist(),
        "patient_ids": pids,
        "macro_f1_4": float(f1_score(y_true_a, y_pred_a, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "weighted_f1_4": float(f1_score(y_true_a, y_pred_a, average="weighted", labels=[0, 1, 2, 3], zero_division=0)),
        "accuracy_4": float(accuracy_score(y_true_a, y_pred_a)),
        "macro_f1_2": float(f1_score(y_true_b, y_pred_b, average="macro", labels=[0, 1], zero_division=0)),
        "weighted_f1_2": float(f1_score(y_true_b, y_pred_b, average="weighted", labels=[0, 1], zero_division=0)),
        "accuracy_2": float(accuracy_score(y_true_b, y_pred_b)),
    }


def _summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = ["macro_f1_4", "weighted_f1_4", "accuracy_4",
               "macro_f1_2", "weighted_f1_2", "accuracy_2"]
    agg = {}
    for m in metrics:
        vals = np.array([r[m] for r in results])
        agg[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                  "per_fold": [float(v) for v in vals]}

    # pooled confusion matrix
    y_true_pool = np.concatenate([np.array(r["y_true_4"]) for r in results])
    y_pred_pool = np.concatenate([np.array(r["y_pred_4"]) for r in results])
    cm = confusion_matrix(y_true_pool, y_pred_pool, labels=[0, 1, 2, 3]).tolist()

    y_true_pool_b = np.concatenate([np.array(r["y_true_2"]) for r in results])
    y_pred_pool_b = np.concatenate([np.array(r["y_pred_2"]) for r in results])
    cm_b = confusion_matrix(y_true_pool_b, y_pred_pool_b, labels=[0, 1]).tolist()

    pooled = {
        "macro_f1_4_pooled": float(f1_score(y_true_pool, y_pred_pool, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "weighted_f1_4_pooled": float(f1_score(y_true_pool, y_pred_pool, average="weighted", labels=[0, 1, 2, 3], zero_division=0)),
        "accuracy_4_pooled": float(accuracy_score(y_true_pool, y_pred_pool)),
        "macro_f1_2_pooled": float(f1_score(y_true_pool_b, y_pred_pool_b, average="macro", labels=[0, 1], zero_division=0)),
        "weighted_f1_2_pooled": float(f1_score(y_true_pool_b, y_pred_pool_b, average="weighted", labels=[0, 1], zero_division=0)),
        "accuracy_2_pooled": float(accuracy_score(y_true_pool_b, y_pred_pool_b)),
        "confusion_4_pooled": cm,
        "confusion_2_pooled": cm_b,
        "label_order_4": ["CR", "PR", "SD", "PD"],
        "label_order_2": ["Non-PD", "PD"],
    }
    return {"per_metric": agg, "pooled": pooled}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Run directory whose .hydra/config.yaml will be reused (for splits).")
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results = []
    for f in args.folds:
        print(f"[oracle] evaluating fold {f}")
        res = _evaluate_fold(args.run_dir, f - 1)
        results.append(res)
        print(f"          n={res['n']}  4cls macro-F1={res['macro_f1_4']:.4f}  "
              f"weighted-F1={res['weighted_f1_4']:.4f}  acc={res['accuracy_4']:.4f}  "
              f"binary macro-F1={res['macro_f1_2']:.4f}")

    summary = _summary(results)
    payload = {
        "run_dir": args.run_dir,
        "folds": args.folds,
        "results": results,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n[oracle] === RULE-BASED RANO ORACLE (GT concepts) ===")
    print(f"           4-class macro-F1   : {summary['per_metric']['macro_f1_4']['mean']:.4f} "
          f"+/- {summary['per_metric']['macro_f1_4']['std']:.4f}")
    print(f"           4-class weighted-F1: {summary['per_metric']['weighted_f1_4']['mean']:.4f} "
          f"+/- {summary['per_metric']['weighted_f1_4']['std']:.4f}")
    print(f"           4-class accuracy   : {summary['per_metric']['accuracy_4']['mean']:.4f} "
          f"+/- {summary['per_metric']['accuracy_4']['std']:.4f}")
    print(f"           Binary  macro-F1   : {summary['per_metric']['macro_f1_2']['mean']:.4f} "
          f"+/- {summary['per_metric']['macro_f1_2']['std']:.4f}")
    print(f"           Binary  weighted-F1: {summary['per_metric']['weighted_f1_2']['mean']:.4f} "
          f"+/- {summary['per_metric']['weighted_f1_2']['std']:.4f}")
    print("\n           Pooled 4x4 confusion (rows=GT CR/PR/SD/PD, cols=Pred):")
    for row in summary["pooled"]["confusion_4_pooled"]:
        print("            ", row)
    print(f"\n[oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
