"""
Intervention policy ablation: ordered top-k vs random-k concept interventions.

For each fold, evaluates macro-F1 when replacing k concepts with GT values:
- Ordered policy: concepts sorted by mean per-concept lift from
  single_c_interventions_on_y.pkl across folds.
- Random policy: random subsets of size k, averaged over multiple trials.

Outputs JSON + CSV summary suitable for paper tables.

Usage:
  python intervention_policy_ablation.py \
      --run-dir outputs/2026-05-14/10-30-17 \
      --k-max 8 --num-random 12 \
      --out-json paper_figures/intervention_policy_ablation.json \
      --out-csv paper_figures/intervention_policy_ablation.csv
"""

import argparse
import copy
import glob
import itertools
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.data.dataset_block import get_dataset
from src.data.utils import static_graph_collate
from src.engines.predictor import Predictor
from src.models.layers.intervention import get_test_intervention_index
from src.utils import (
    clean_empty_configs,
    get_intervention_policy,
    maybe_update_config_with_graph,
    remove_cycles,
    remove_problematic_edges,
    update_config_from_data,
)


def seed_everything(seed: int):
    random.seed(seed)
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


def _class_names(task_cardinality: int):
    if task_cardinality == 4:
        return ["CR", "PR", "SD", "PD"]
    if task_cardinality == 2:
        return ["Non-PD", "PD"]
    return [str(i) for i in range(task_cardinality)]


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _collect_preds(engine, dataloader, device, intervention_nodes):
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            x, c, y = engine._unpack_batch(batch)
            intervention_index = get_test_intervention_index(c.shape, intervention_nodes)
            inputs = engine._model_inputs(x, c, intervention_index, batch)
            y_output, c_output = engine.forward(**inputs)
            y_hat, _ = engine.model.filter_output_for_metric(y_output, c_output)
            if len(y_hat.shape) > 2:
                y_hat = y_hat.mean(dim=-1)
            preds_all.append(y_hat.argmax(dim=-1).detach().cpu().numpy().reshape(-1))
            targets_all.append(y.detach().cpu().numpy().reshape(-1))
    if not preds_all:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(targets_all), np.concatenate(preds_all)


def _macro_f1(targets, preds, class_names):
    rep = Predictor._build_task_report(targets, preds, class_names)
    return float(rep["macro_f1"])


def _prepare_fold_cfg(base_cfg, fold_idx, n_splits):
    fold_cfg = copy.deepcopy(base_cfg)
    with open_dict(fold_cfg):
        fold_cfg.dataset.loader.cv_fold = fold_idx
        fold_cfg.dataset.loader.cv_n_splits = n_splits
        fold_cfg.dataset.loader.cv_val_fold = fold_idx
    return fold_cfg


def _build_engine_and_loader(cfg, device, eval_batch_size=None):
    seed_everything(cfg.get("seed", 0))
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
    [dataset.data[split].register_graph(graph) for split in dataset.data]

    bs = int(eval_batch_size) if eval_batch_size is not None else int(cfg.dataset.batch_size)
    val_loader = DataLoader(
        dataset.data["val"],
        batch_size=bs,
        collate_fn=static_graph_collate,
        num_workers=cfg.dataset.num_workers,
    )
    engine = instantiate(cfg.engine)
    engine.to(device)
    engine.eval()
    return engine, val_loader


def _load_order_from_single_interventions(run_dir: Path, folds: List[int]) -> List[str]:
    lifts = {}
    for f in folds:
        p = run_dir / f"fold_{f}/results/single_c_interventions_on_y.pkl"
        d = pickle.load(open(p, "rb"))
        bkeys = [k for k in d if k.startswith("_baseline") and "f1_macro" in k]
        baseline = float(d[bkeys[0]]) if bkeys else float(next(v for k, v in d.items() if k.endswith("_baseline_f1_macro")))
        for k, v in d.items():
            if not k.endswith("_f1_macro"):
                continue
            if k.startswith("_baseline"):
                continue
            c = k[:-len("_f1_macro")]
            # Skip bookkeeping keys like "<concept>_baseline_f1_macro".
            if c.endswith("_baseline"):
                continue
            lifts.setdefault(c, []).append(float(v) - baseline)
    # use mean lift (descending)
    return sorted(lifts.keys(), key=lambda c: -float(np.mean(lifts[c])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--num-random", type=int, default=12)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--eval-batch-size", type=int, default=4,
                    help="Evaluation batch size used for intervention sweeps (lower helps avoid OOM).")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    cv_enabled, n_splits = _cv_settings(cfg)
    if not cv_enabled:
        raise ValueError("This script expects a CV run.")

    ordered_names = _load_order_from_single_interventions(run_dir, args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)

    per_fold = {}
    for fold in args.folds:
        print(f"[policy-ablation] fold {fold}: loading model ...")
        fold_cfg = _prepare_fold_cfg(cfg, fold - 1, n_splits)
        engine, val_loader = _build_engine_and_loader(fold_cfg, device, eval_batch_size=args.eval_batch_size)
        ckpt = _choose_checkpoint(str(run_dir / f"fold_{fold}"))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        engine.load_state_dict(state_dict, strict=False)
        engine.eval()

        c_names = list(engine.c_names)
        idx_by_name = {n: i for i, n in enumerate(c_names)}
        ordered_idx = [idx_by_name[n] for n in ordered_names if n in idx_by_name]
        k_max = min(args.k_max, len(ordered_idx))
        class_names = _class_names(int(fold_cfg.dataset.loader.task_cardinality))

        fold_out = {"k": [], "ordered_macro_f1": [], "random_macro_f1_mean": [], "random_macro_f1_std": []}

        for k in range(0, k_max + 1):
            if k == 0:
                y_t, y_p = _collect_preds(engine, val_loader, device, [])
                ordered_f1 = _macro_f1(y_t, y_p, class_names)
                rnd_mean = ordered_f1
                rnd_std = 0.0
            else:
                nodes_ord = ordered_idx[:k]
                y_t, y_p = _collect_preds(engine, val_loader, device, nodes_ord)
                ordered_f1 = _macro_f1(y_t, y_p, class_names)

                rnd_scores = []
                for _ in range(args.num_random):
                    nodes_r = random.sample(ordered_idx, k)
                    y_rt, y_rp = _collect_preds(engine, val_loader, device, nodes_r)
                    rnd_scores.append(_macro_f1(y_rt, y_rp, class_names))
                rnd_mean = float(np.mean(rnd_scores))
                rnd_std = float(np.std(rnd_scores, ddof=0))

            fold_out["k"].append(k)
            fold_out["ordered_macro_f1"].append(float(ordered_f1))
            fold_out["random_macro_f1_mean"].append(float(rnd_mean))
            fold_out["random_macro_f1_std"].append(float(rnd_std))

        per_fold[str(fold)] = fold_out
        print(
            f"[policy-ablation] fold {fold}: "
            f"k={k_max} ordered={fold_out['ordered_macro_f1'][-1]:.4f} "
            f"random={fold_out['random_macro_f1_mean'][-1]:.4f}"
        )
        del engine
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate across folds
    k_vals = per_fold[str(args.folds[0])]["k"]
    ordered_all = np.array([per_fold[str(f)]["ordered_macro_f1"] for f in args.folds], dtype=float)
    random_all = np.array([per_fold[str(f)]["random_macro_f1_mean"] for f in args.folds], dtype=float)
    gap_all = ordered_all - random_all

    aggregate = {
        "k": k_vals,
        "ordered_mean": ordered_all.mean(axis=0).tolist(),
        "ordered_std": ordered_all.std(axis=0, ddof=0).tolist(),
        "random_mean": random_all.mean(axis=0).tolist(),
        "random_std": random_all.std(axis=0, ddof=0).tolist(),
        "gap_mean": gap_all.mean(axis=0).tolist(),
        "gap_std": gap_all.std(axis=0, ddof=0).tolist(),
        "ordered_ois": float(ordered_all[:, 1:].mean(axis=1).mean() - ordered_all[:, 0].mean()),
        "random_ois": float(random_all[:, 1:].mean(axis=1).mean() - random_all[:, 0].mean()),
    }

    payload = {
        "run_dir": str(run_dir),
        "folds": args.folds,
        "k_max": args.k_max,
        "num_random": args.num_random,
        "ordered_concepts": ordered_names,
        "per_fold": per_fold,
        "aggregate": aggregate,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    with open(args.out_csv, "w") as fp:
        fp.write("k,ordered_mean,ordered_std,random_mean,random_std,gap_mean,gap_std\n")
        for i, k in enumerate(k_vals):
            fp.write(
                f"{k},{aggregate['ordered_mean'][i]:.6f},{aggregate['ordered_std'][i]:.6f},"
                f"{aggregate['random_mean'][i]:.6f},{aggregate['random_std'][i]:.6f},"
                f"{aggregate['gap_mean'][i]:.6f},{aggregate['gap_std'][i]:.6f}\n"
            )

    print("\n[policy-ablation] === Ordered top-k vs random-k ===")
    print(f"ordered OIS-like gain: {aggregate['ordered_ois']:.4f}")
    print(f"random  OIS-like gain: {aggregate['random_ois']:.4f}")
    print(f"delta (ordered-random): {aggregate['ordered_ois'] - aggregate['random_ois']:.4f}")
    print(f"[policy-ablation] wrote {args.out_json}")
    print(f"[policy-ablation] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
