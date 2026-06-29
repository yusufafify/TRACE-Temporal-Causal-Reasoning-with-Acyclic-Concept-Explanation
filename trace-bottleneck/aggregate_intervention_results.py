"""Aggregate per-fold intervention pkl files into a single sorted concept table.

Usage:
    python aggregate_intervention_results.py outputs/<date>/<run>

Reads, for each fold_<i>/results/:
  - single_c_interventions_on_y.pkl   (per-concept Δ F1 from setting concept = GT)
  - cace_per_concept.pkl              (causal effect, low_val vs high_val)
  - level_interventions_on_y.pkl      (cumulative-level intervention F1)
  - y_accuracy.pkl                    (baseline)

Prints a single sorted table with mean ± std across folds.
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _fmt(values):
    if not values:
        return "—"
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return f"{m*100:+.2f} ± {s*100:.2f}"


def _fmt_unitless(values):
    if not values:
        return "—"
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str,
                   help="Path to a run directory with fold_<i>/results/*.pkl")
    p.add_argument("--n_folds", type=int, default=5)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"run_dir does not exist: {run_dir}")

    # ------------------------------------------------------------------
    # Per-concept Δ F1 (from single_c_interventions_on_y) and CaCE
    # ------------------------------------------------------------------
    delta_f1 = defaultdict(list)
    cace = defaultdict(list)
    delta_y_baselines = []
    delta_y_oracle = []

    for i in range(1, args.n_folds + 1):
        sci = run_dir / f"fold_{i}" / "results" / "single_c_interventions_on_y.pkl"
        cace_pkl = run_dir / f"fold_{i}" / "results" / "cace_per_concept.pkl"
        lvl_pkl = run_dir / f"fold_{i}" / "results" / "level_interventions_on_y.pkl"

        if sci.exists():
            d = _load(sci)
            base = d.get("_baseline_f1_macro")
            if base is None:
                continue
            for k, v in d.items():
                if k.endswith("_delta_f1_macro"):
                    concept = k[: -len("_delta_f1_macro")]
                    delta_f1[concept].append(v)

        if cace_pkl.exists():
            d = _load(cace_pkl)
            for concept, info in d.items():
                if isinstance(info, dict) and "cace_tv" in info:
                    cace[concept].append(info["cace_tv"])

        if lvl_pkl.exists():
            d = _load(lvl_pkl)
            base_lvl = d.get("level 0")
            if base_lvl is None:
                continue
            # max key by integer level number
            levels = [k for k in d.keys() if k.startswith("level ")]
            top = max(levels, key=lambda s: int(s.split()[-1]))
            delta_y_oracle.append(d[top] - base_lvl)
            delta_y_baselines.append(base_lvl)

    # ------------------------------------------------------------------
    # Print sorted table
    # ------------------------------------------------------------------
    all_concepts = sorted(set(delta_f1.keys()) | set(cace.keys()))
    rows = []
    for c in all_concepts:
        df1 = delta_f1.get(c, [])
        cc = cace.get(c, [])
        rank_key = mean(df1) if df1 else 0.0
        rows.append((rank_key, c, df1, cc))
    rows.sort(reverse=True)

    print()
    print(f"Run: {run_dir}")
    print(f"Folds aggregated: {args.n_folds}")
    if delta_y_baselines:
        print(f"Mean baseline acc (level 0)       : {mean(delta_y_baselines)*100:.2f}%")
    if delta_y_oracle:
        print(f"Mean oracle Δ acc (intervene-all) : {_fmt(delta_y_oracle)} pp")
    print()
    print(f"{'concept':<40} {'Δ F1 (pp)':<22} {'CaCE':<22} n")
    print("-" * 95)
    for _, c, df1, cc in rows:
        n_df1 = len(df1)
        n_cc = len(cc)
        n = max(n_df1, n_cc)
        print(f"{c:<40} {_fmt(df1):<22} {_fmt_unitless(cc):<22} {n}")
    print("-" * 95)
    print("Δ F1 = (F1 after replacing predicted concept with GT) − (no-intervention F1).")
    print("       Positive ⇒ predicted concept was hurting Y; GT helps. Near zero ⇒ "
          "concept either accurate already or unused by the classifier.")
    print("CaCE = average treatment effect of toggling concept low→high on P(Y).")


if __name__ == "__main__":
    main()
