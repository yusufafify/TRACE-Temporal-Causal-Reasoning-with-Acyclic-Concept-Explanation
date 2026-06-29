"""
Recompute CaCE with a clinically constrained time_gap intervention (6 -> 16 weeks)
while keeping all other concepts exactly as in existing cace_per_concept.pkl.

This script does not overwrite original artifacts. It writes new table/figures with
suffixes under paper_figures/.

Usage:
  python recompute_cace_timegap_clinical.py \
      --run-dir outputs/2026-05-14/10-30-17 \
      --timegap-low-weeks 6 --timegap-high-weeks 16
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset_block import get_dataset
from src.data.utils import static_graph_collate
from src.utils import (
    clean_empty_configs,
    get_intervention_policy,
    maybe_update_config_with_graph,
    remove_cycles,
    remove_problematic_edges,
    update_config_from_data,
)


CLASSES = ["CR", "PR", "SD", "PD"]


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cv_settings(cfg):
    enabled = False
    n_splits = 5
    top_cv = cfg.get("cross_validation", None)
    if top_cv is not None:
        enabled = bool(top_cv.get("enabled", enabled))
        n_splits = int(top_cv.get("n_splits", n_splits))
    dataset_cv = cfg.dataset.get("cross_validation", None)
    if dataset_cv is not None:
        enabled = bool(dataset_cv.get("enabled", enabled))
        n_splits = int(dataset_cv.get("n_splits", n_splits))
    return enabled, n_splits


def _prepare_fold_cfg(base_cfg, fold_idx, n_splits):
    fold_cfg = copy.deepcopy(base_cfg)
    with open_dict(fold_cfg):
        fold_cfg.dataset.loader.cv_fold = fold_idx
        fold_cfg.dataset.loader.cv_n_splits = n_splits
        fold_cfg.dataset.loader.cv_val_fold = fold_idx
    return fold_cfg


def _choose_checkpoint(fold_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(fold_dir, "checkpoints", "*.ckpt")))
    non_last = [p for p in ckpts if os.path.basename(p) != "last.ckpt"]
    if len(non_last) == 1:
        return non_last[0]
    if non_last:
        return non_last[-1]
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {fold_dir}/checkpoints")


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _build_engine_and_loader(cfg, device, split: str, eval_batch_size: int):
    seed_everything(int(cfg.get("seed", 0)))
    with open_dict(cfg):
        cfg.update(device=str(device))
    cfg = clean_empty_configs(cfg)

    dataset, true_graph, dataset_directory = get_dataset(cfg)
    if cfg.dataset.load_true_graph:
        graph = true_graph
    elif cfg.dataset.load_graph:
        with open(os.path.join(dataset_directory, "graph.pkl"), "rb") as f:
            graph = pickle.load(f)
    else:
        raise ValueError("Run must use load_true_graph or load_graph.")

    graph, dataset = remove_problematic_edges(graph, dataset)
    y_index = list(graph.index).index(dataset.y_info["names"][0])
    graph = remove_cycles(graph, y_index)

    policy_cfg = cfg.dataset.get("policy", [])
    if not policy_cfg:
        interv_policy = []
    else:
        interv_policy, _ = get_intervention_policy(policy_cfg, graph, true_graph, y_index)

    cfg = update_config_from_data(cfg, dataset)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)
    for k in dataset.data:
        dataset.data[k].register_graph(graph)

    if split not in dataset.data:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset.data.keys())}")

    dataloader = DataLoader(
        dataset.data[split],
        batch_size=int(eval_batch_size),
        collate_fn=static_graph_collate,
        num_workers=cfg.dataset.num_workers,
    )

    engine = instantiate(cfg.engine)
    engine.to(device)
    engine.eval()
    return engine, dataloader


def _compute_timegap_cace(engine, dataloader, device, z_low: float, z_high: float):
    c_names = list(engine.c_names)
    if "time_gap" not in c_names:
        raise ValueError("time_gap concept not found in engine.c_names")
    ci = c_names.index("time_gap")

    probs_high_all, probs_low_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            x, c, _ = engine._unpack_batch(batch)

            interv_idx = torch.zeros_like(c)
            interv_idx[:, ci] = 1.0

            c_high = c.clone()
            c_high[:, ci] = float(z_high)
            y_out_h, c_out_h = engine.forward(**engine._model_inputs(x, c_high, interv_idx, batch))
            y_hat_h, _ = engine.model.filter_output_for_metric(y_out_h, c_out_h)
            if len(y_hat_h.shape) > 2:
                y_hat_h = y_hat_h.mean(dim=-1)
            probs_high_all.append(y_hat_h.detach().cpu())

            c_low = c.clone()
            c_low[:, ci] = float(z_low)
            y_out_l, c_out_l = engine.forward(**engine._model_inputs(x, c_low, interv_idx, batch))
            y_hat_l, _ = engine.model.filter_output_for_metric(y_out_l, c_out_l)
            if len(y_hat_l.shape) > 2:
                y_hat_l = y_hat_l.mean(dim=-1)
            probs_low_all.append(y_hat_l.detach().cpu())

    probs_high = torch.cat(probs_high_all, dim=0)
    probs_low = torch.cat(probs_low_all, dim=0)

    avg_high = probs_high.mean(dim=0)
    avg_low = probs_low.mean(dim=0)

    tv_distance = 0.5 * (avg_high - avg_low).abs().sum().item()
    per_class = (avg_high - avg_low).tolist()

    return {
        "cace_tv": float(tv_distance),
        "per_class": [float(v) for v in per_class],
        "high_val": float(z_high),
        "low_val": float(z_low),
        "cardinality": 1,
        "avg_probs_high": [float(v) for v in avg_high.tolist()],
        "avg_probs_low": [float(v) for v in avg_low.tolist()],
    }


def _plot_cace_bar(agg_rows, out_png: Path):
    concepts = [r["concept"] for r in agg_rows]
    means = [r["mean"] for r in agg_rows]
    stds = [r["std"] for r in agg_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(concepts))
    ax.barh(y, means, xerr=stds, color="#3a7ebf", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(concepts, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_xlabel("CaCE total-variation  (mean +/- std over 5 folds)")
    ax.set_title("Causal effect of each concept")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_cace_heatmap(per_class_mean, ordered_concepts, out_png: Path):
    mat = np.array([per_class_mean[c] for c in ordered_concepts], dtype=float)
    vmax = float(np.abs(mat).max()) if mat.size else 1.0

    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(ordered_concepts) + 1.2))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(ordered_concepts)))
    ax.set_yticklabels(ordered_concepts, fontsize=8)
    ax.set_title("Signed CaCE: dP(class) when concept set high vs low")
    fig.colorbar(im, ax=ax, shrink=0.85, label="dP")

    if vmax > 0:
        for i in range(len(ordered_concepts)):
            for j in range(4):
                val = mat[i, j]
                color = "black" if abs(val) < 0.6 * vmax else "white"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--timegap-low-weeks", type=float, default=6.0)
    ap.add_argument("--timegap-high-weeks", type=float, default=16.0)
    ap.add_argument("--out-dir", default="paper_figures")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"timegap_{int(args.timegap_low_weeks)}_{int(args.timegap_high_weeks)}w"

    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    cv_enabled, n_splits = _cv_settings(cfg)
    if not cv_enabled:
        raise ValueError("This script expects a CV run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_fold_updated = {}

    for fold in args.folds:
        print(f"[cace-clinical] fold {fold}: loading model and data ...")
        fold_cfg = _prepare_fold_cfg(cfg, fold - 1, n_splits)
        engine, dataloader = _build_engine_and_loader(
            fold_cfg, device, split=args.split, eval_batch_size=args.eval_batch_size
        )

        ckpt = _choose_checkpoint(str(run_dir / f"fold_{fold}"))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        engine.load_state_dict(state_dict, strict=False)
        engine.eval()

        c_names = list(engine.c_names)
        ti = c_names.index("time_gap")
        means = getattr(engine.model, "concept_means", None)
        stds = getattr(engine.model, "concept_stds", None)
        if means is None or stds is None:
            raise RuntimeError("Model is missing concept_means/concept_stds needed for raw->z conversion.")

        mu = float(means[ti].detach().cpu().item())
        sd = float(stds[ti].detach().cpu().item())
        # LUMIERE stores time_gap as weeks normalized by 52.
        raw_low_norm = float(args.timegap_low_weeks) / 52.0
        raw_high_norm = float(args.timegap_high_weeks) / 52.0
        z_low = float((raw_low_norm - mu) / (sd + 1e-8))
        z_high = float((raw_high_norm - mu) / (sd + 1e-8))

        new_timegap = _compute_timegap_cace(engine, dataloader, device, z_low=z_low, z_high=z_high)
        new_timegap["raw_low_weeks"] = float(args.timegap_low_weeks)
        new_timegap["raw_high_weeks"] = float(args.timegap_high_weeks)
        new_timegap["raw_low_norm"] = raw_low_norm
        new_timegap["raw_high_norm"] = raw_high_norm
        new_timegap["concept_space_mean"] = mu
        new_timegap["concept_space_std"] = sd

        base_pkl = run_dir / f"fold_{fold}/results/cace_per_concept.pkl"
        base = pickle.load(open(base_pkl, "rb"))
        base["time_gap"] = new_timegap
        per_fold_updated[str(fold)] = base

        fold_out_dir = out_dir / f"cace_{suffix}" / f"fold_{fold}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = fold_out_dir / "cace_per_concept.pkl"
        pickle.dump(base, open(out_pkl, "wb"))

        print(
            f"[cace-clinical] fold {fold}: time_gap CaCE={new_timegap['cace_tv']:.4f} "
            f"(z {z_low:.3f}->{z_high:.3f}, raw {args.timegap_low_weeks:.1f}->{args.timegap_high_weeks:.1f}w)"
        )

        del engine
        del dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate table + heatmap inputs
    concept_names = sorted(per_fold_updated[str(args.folds[0])].keys())
    agg_rows = []
    per_class_mean = {}
    for cname in concept_names:
        tvs = []
        pcs = []
        for fold in args.folds:
            item = per_fold_updated[str(fold)][cname]
            tvs.append(float(item["cace_tv"]))
            pcs.append(np.array(item["per_class"], dtype=float))
        arr = np.array(tvs, dtype=float)
        pc_arr = np.stack(pcs)
        agg_rows.append({
            "concept": cname,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "per_fold": [float(x) for x in arr.tolist()],
        })
        per_class_mean[cname] = [float(x) for x in pc_arr.mean(axis=0).tolist()]

    agg_rows = sorted(agg_rows, key=lambda x: -x["mean"])

    # CSV table (all concepts)
    out_csv = out_dir / f"cace_table_{suffix}.csv"
    with open(out_csv, "w") as fp:
        fp.write("concept,mean_cace_tv,std," + ",".join([f"fold{f}" for f in args.folds]) + "\n")
        for r in agg_rows:
            fp.write(
                f"{r['concept']},{r['mean']:.6f},{r['std']:.6f}," +
                ",".join([f"{v:.6f}" for v in r["per_fold"]]) + "\n"
            )

    # JSON summary
    out_json = out_dir / f"cace_summary_{suffix}.json"
    payload = {
        "run_dir": str(run_dir),
        "folds": args.folds,
        "split": args.split,
        "timegap_raw_weeks": {
            "low": float(args.timegap_low_weeks),
            "high": float(args.timegap_high_weeks),
        },
        "table": agg_rows,
        "time_gap_rank": int(next(i for i, r in enumerate(agg_rows, 1) if r["concept"] == "time_gap")),
        "top12": agg_rows[:12],
    }
    with open(out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    # Plots
    out_bar = out_dir / f"fig_cace_bar_{suffix}.png"
    _plot_cace_bar(agg_rows, out_bar)

    ordered_for_heatmap = [r["concept"] for r in sorted(agg_rows, key=lambda x: -abs(np.mean(per_class_mean[x["concept"]])))]
    out_heatmap = out_dir / f"cace_per_class_heatmap_{suffix}.png"
    _plot_cace_heatmap(per_class_mean, ordered_for_heatmap, out_heatmap)

    print("\n[cace-clinical] === done ===")
    print(f"[cace-clinical] wrote {out_csv}")
    print(f"[cace-clinical] wrote {out_json}")
    print(f"[cace-clinical] wrote {out_bar}")
    print(f"[cace-clinical] wrote {out_heatmap}")


if __name__ == "__main__":
    main()
