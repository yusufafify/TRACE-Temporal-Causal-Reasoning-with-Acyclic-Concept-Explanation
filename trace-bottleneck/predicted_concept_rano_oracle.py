"""
Predicted-concept RANO threshold oracle.

Applies textbook RANO threshold logic to concept predictions produced by the
trained C2BM (no learned task head usage):

    PD if new_lesion_flag or vol_pd_flag or spd_pd_flag
    CR elif (vol_pr_flag or spd_pr_flag) and both follow-up volumes < threshold
    PR elif vol_pr_flag or spd_pr_flag
    SD otherwise

This isolates the decision-layer behavior when concept estimates are noisy.

Usage:
  python predicted_concept_rano_oracle.py \
      --run-dir outputs/2026-05-14/10-30-17 \
      --out paper_figures/predicted_concept_oracle_results.json
"""

import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from extract_patient_example import extract_examples


BINARY_REMAP = {"CR": 0, "PR": 0, "SD": 0, "PD": 1}
CR_VOLUME_THRESH_CM3 = 0.10


def _concept_map(example: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["name"]: c for c in example["concepts"]}


def _oracle_pred_label(example: Dict[str, Any]) -> str:
    cm = _concept_map(example)

    def val(name: str, default: float = 0.0) -> float:
        if name not in cm:
            return default
        return float(cm[name].get("pred_numeric", cm[name].get("pred_raw", default)))

    new_lesion = val("new_lesion_flag") >= 0.5
    vol_pd = val("vol_pd_flag") >= 0.5
    spd_pd = val("spd_pd_flag") >= 0.5
    vol_pr = val("vol_pr_flag") >= 0.5
    spd_pr = val("spd_pr_flag") >= 0.5

    if new_lesion or vol_pd or spd_pd:
        return "PD"

    follow_enh = val("followup_enhancing_volume_cm3")
    follow_ne = val("followup_non_enhancing_volume_cm3")
    if (vol_pr or spd_pr) and follow_enh < CR_VOLUME_THRESH_CM3 and follow_ne < CR_VOLUME_THRESH_CM3:
        return "CR"
    if vol_pr or spd_pr:
        return "PR"
    return "SD"


def _metrics_from_labels(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels4 = ["CR", "PR", "SD", "PD"]
    y_true_2 = [BINARY_REMAP[y] for y in y_true]
    y_pred_2 = [BINARY_REMAP[y] for y in y_pred]

    return {
        "macro_f1_4": float(f1_score(y_true, y_pred, labels=labels4, average="macro", zero_division=0)),
        "weighted_f1_4": float(f1_score(y_true, y_pred, labels=labels4, average="weighted", zero_division=0)),
        "accuracy_4": float(accuracy_score(y_true, y_pred)),
        "macro_f1_2": float(f1_score(y_true_2, y_pred_2, labels=[0, 1], average="macro", zero_division=0)),
        "weighted_f1_2": float(f1_score(y_true_2, y_pred_2, labels=[0, 1], average="weighted", zero_division=0)),
        "accuracy_2": float(accuracy_score(y_true_2, y_pred_2)),
        "confusion_4": confusion_matrix(y_true, y_pred, labels=labels4).tolist(),
        "confusion_2": confusion_matrix(y_true_2, y_pred_2, labels=[0, 1]).tolist(),
    }


def _aggregate(per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = [
        "macro_f1_4", "weighted_f1_4", "accuracy_4",
        "macro_f1_2", "weighted_f1_2", "accuracy_2",
    ]
    summary = {}
    for k in metric_keys:
        vals = np.array([f[k] for f in per_fold], dtype=float)
        summary[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "per_fold": [float(v) for v in vals],
        }

    y_true_all = sum([f["y_true"] for f in per_fold], [])
    y_pred_all = sum([f["y_pred"] for f in per_fold], [])
    pooled = _metrics_from_labels(y_true_all, y_pred_all)
    pooled.update({"n": len(y_true_all)})
    return {"per_metric": summary, "pooled": pooled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_fold = []
    for f in args.folds:
        rows = extract_examples(args.run_dir, f - 1)
        y_true = [r["gt_label"] for r in rows]
        y_pred = [_oracle_pred_label(r) for r in rows]
        met = _metrics_from_labels(y_true, y_pred)
        per_fold.append({
            "fold": f,
            "n": len(rows),
            "y_true": y_true,
            "y_pred": y_pred,
            **{k: met[k] for k in ["macro_f1_4", "weighted_f1_4", "accuracy_4", "macro_f1_2", "weighted_f1_2", "accuracy_2"]},
            "confusion_4": met["confusion_4"],
            "confusion_2": met["confusion_2"],
        })
        print(
            f"[pred-oracle] fold {f}: n={len(rows)} "
            f"4cls macro-F1={met['macro_f1_4']:.4f} "
            f"binary macro-F1={met['macro_f1_2']:.4f}"
        )

    summary = _aggregate(per_fold)
    payload = {
        "run_dir": args.run_dir,
        "folds": args.folds,
        "per_fold": per_fold,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(payload, fp, indent=2)

    s = summary["per_metric"]
    print("\n[pred-oracle] === Predicted-concept threshold oracle ===")
    print(f"4-class macro-F1    : {s['macro_f1_4']['mean']:.4f} +/- {s['macro_f1_4']['std']:.4f}")
    print(f"4-class weighted-F1 : {s['weighted_f1_4']['mean']:.4f} +/- {s['weighted_f1_4']['std']:.4f}")
    print(f"4-class accuracy    : {s['accuracy_4']['mean']:.4f} +/- {s['accuracy_4']['std']:.4f}")
    print(f"Binary macro-F1     : {s['macro_f1_2']['mean']:.4f} +/- {s['macro_f1_2']['std']:.4f}")
    print(f"Binary weighted-F1  : {s['weighted_f1_2']['mean']:.4f} +/- {s['weighted_f1_2']['std']:.4f}")
    print(f"[pred-oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
