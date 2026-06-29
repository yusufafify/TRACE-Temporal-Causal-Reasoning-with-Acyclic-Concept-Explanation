#!/usr/bin/env python
"""
Run all available causal discovery algorithms on the LUMIERE dataset and
compare the discovered graphs against the expert-specified true graph.

Algorithms tested:
  1. PC   (constraint-based, chi-squared independence test)
  2. PC with G² test
  3. GES  (score-based, BDeu score)
  4. GRaSP (score-based, BDeu score)
  5. LLM-dummy (fully-undirected placeholder — no data-driven discovery)

FGES is skipped because it requires pytetrad (Java/JVM), which is not installed.

FIX: The original process_data_for_causal_discovery() casts z-scored continuous
concepts to torch.long, which truncates floats to integers and destroys most of
the signal.  This script uses proper quantile-based discretization for continuous
concepts so that chi-squared / G-squared tests can detect dependencies.

Outputs (saved to  ./causal_discovery_comparison/):
  - <algo>_graph.png         : visualisation of each discovered graph
  - true_graph.png           : the expert-specified ground truth
  - comparison_summary.txt   : Hamming distance table
  - adjacency matrices printed to stdout
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import CACHE  # noqa: E402

from src.causal_discovery.causal_discovery_block import (
    apply_causal_discovery,
    postprocess_graph,
)
from src.metrics import hamming_distance
from src.plots import maybe_plot_graph
from src.data.datasets.lumiere import CONCEPT_NAMES, CATEGORICAL_CONCEPTS, BINARY_CONCEPTS

# ── configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "causal_discovery_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITHMS = [
    {
        "name": "pc",
        "type": "constraint-based",
        "library": "causallearn",
        "kwargs": {"ind_test": "chisq"},
    },
    {
        "name": "pc_gsq",
        "display_name": "PC (G²)",
        "algo_name": "pc",
        "type": "constraint-based",
        "library": "causallearn",
        "kwargs": {"ind_test": "gsq"},
    },
    {
        "name": "ges",
        "type": "score-based",
        "library": "causallearn",
        "kwargs": {"score_func": "local_score_BDeu"},
    },
    {
        "name": "grasp",
        "type": "score-based",
        "library": "causallearn",
        "kwargs": {"score_func": "local_score_BDeu"},
    },
    {
        "name": "llm_dummy",
        "display_name": "LLM-dummy (fully undirected)",
        "type": None,
        "library": None,
        "kwargs": {},
    },
]

SEED = 42


def load_lumiere_dataset():
    """Instantiate and split the LUMIERE dataset (concept-only, no images)."""
    from src.data.datasets.lumiere import LumiereDataset

    print("Loading LUMIERE dataset ...")
    dataset = LumiereDataset(
        root_dir="/home/group2/dataset/Lumiere/",
        img_dir="/home/group2/dataset/processed_images/",
        use_delta_only=False,
        use_8_channels=True,
        input_mode="radiomic",
        concepts_3d_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "concepts_3d.json"
        ),
        case_cache_dir="/home/group2/Hamza_new/lumiere_case_cache",
        task_cardinality=4,
        cv_fold=0,
        cv_n_splits=5,
        cv_val_fold=1,
    )
    dataset.split()

    for split in dataset.data:
        dataset.data[split].update_lists()

    true_graph = dataset.load_ground_truth_graph()
    return dataset, true_graph


def discretize_for_causal_discovery(data, concept_names, n_bins=5):
    """Properly discretize data for causal discovery.

    - Categorical/binary concepts: kept as-is (already integer-valued)
    - Continuous concepts: quantile-binned into `n_bins` equal-frequency bins

    This avoids the torch.long truncation bug in the original
    process_data_for_causal_discovery() which collapses z-scored floats
    (mean≈0, std≈1) into a handful of integers, destroying the signal.
    """
    c = data.c.clone()   # [N, n_concepts]
    y = data.y.clone()   # [N, 1]

    categorical_set = set(CATEGORICAL_CONCEPTS) | set(BINARY_CONCEPTS)
    processed_cols = []

    for ci, name in enumerate(concept_names):
        col = c[:, ci].numpy()
        if name in categorical_set:
            # Already discrete — keep as int
            processed_cols.append(torch.tensor(col, dtype=torch.long))
        else:
            # Continuous: quantile-based discretization
            try:
                binned = pd.qcut(col, q=n_bins, labels=False, duplicates='drop')
            except ValueError:
                # Fallback if too few unique values for n_bins
                binned = pd.cut(col, bins=min(n_bins, len(np.unique(col))),
                                labels=False, duplicates='drop')
            binned = np.nan_to_num(binned, nan=0).astype(int)
            processed_cols.append(torch.tensor(binned, dtype=torch.long))

    # Add the label
    processed_cols.append(y.squeeze(-1).long())

    discretized = torch.stack(processed_cols, dim=1)

    # Print diagnostic info
    print(f"\n  Discretized data shape: {discretized.shape}")
    all_names = list(concept_names) + ['TreatmentResponse']
    for ci, name in enumerate(all_names):
        col = discretized[:, ci]
        unique = sorted(col.unique().tolist())
        print(f"    {name}: {len(unique)} unique values: {unique}")

    return discretized


def build_llm_dummy_graph(label_names):
    """Reproduce the 'llm' config behaviour: fully-undirected dummy matrix."""
    n = len(label_names)
    dummy = np.zeros((n, n))
    dummy[np.triu_indices(n, k=1)] = -1
    dummy[np.tril_indices(n, k=-1)] = -1
    return pd.DataFrame(dummy.astype(int), index=label_names, columns=label_names)


def run_comparison():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset, true_graph = load_lumiere_dataset()
    label_names = dataset.c_info["names"] + dataset.y_info["names"]

    # Save true graph visualisation
    true_graph_path = os.path.join(OUTPUT_DIR, "true_graph")
    maybe_plot_graph(true_graph, true_graph_path)
    print(f"\n{'='*70}")
    print("TRUE GRAPH (expert-specified)")
    print(f"{'='*70}")
    print(true_graph)

    # Count true graph edges
    adj_true = true_graph.values
    n = len(adj_true)
    true_directed = sum(1 for i in range(n) for j in range(i+1, n)
                        if (adj_true[i,j]==1 and adj_true[j,i]==0) or
                           (adj_true[i,j]==0 and adj_true[j,i]==1))
    true_undirected = sum(1 for i in range(n) for j in range(i+1, n)
                          if (adj_true[i,j]==-1 and adj_true[j,i]==-1) or
                             (adj_true[i,j]==1 and adj_true[j,i]==1))
    print(f"\n  True graph: {true_directed} directed, {true_undirected} undirected, "
          f"{n*(n-1)//2 - true_directed - true_undirected} absent edges")

    # Discretize data ONCE (shared across algorithms)
    print(f"\n{'='*70}")
    print("DISCRETIZING DATA (quantile-based, 5 bins for continuous)")
    print(f"{'='*70}")
    discretized_data = discretize_for_causal_discovery(
        dataset.data["train"],
        dataset.c_info["names"],
        n_bins=5,
    )

    results = []

    for algo_cfg in ALGORITHMS:
        name = algo_cfg["name"]
        display = algo_cfg.get("display_name", name.upper())
        algo_func_name = algo_cfg.get("algo_name", name)

        print(f"\n{'='*70}")
        print(f"Running: {display}")
        print(f"{'='*70}")

        t0 = time.time()

        if algo_cfg["type"] is None:
            predicted_graph = build_llm_dummy_graph(label_names)
        else:
            raw_graph, model_info = apply_causal_discovery(
                discretized_data,
                algo_func_name,
                algo_cfg["type"],
                algo_cfg["library"],
                **algo_cfg["kwargs"],
            )
            print(f"  model_info: {model_info}")

            predicted_graph = postprocess_graph(
                raw_graph,
                label_names,
                algo_func_name,
                algo_cfg["type"],
                algo_cfg["library"],
            )

        elapsed = time.time() - t0

        cost, count = hamming_distance(true_graph, predicted_graph)

        print(f"\n  Adjacency matrix:")
        print(predicted_graph)
        print(f"\n  Time: {elapsed:.2f}s")
        print(f"  Hamming distance: cost={cost:.4f}, mismatched_edges={count}")

        graph_path = os.path.join(OUTPUT_DIR, f"{name}_graph")
        maybe_plot_graph(predicted_graph, graph_path)

        adj = predicted_graph.values
        n = len(adj)
        directed = 0
        undirected = 0
        absent = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1 and adj[j, i] == 0:
                    directed += 1
                elif adj[i, j] == 0 and adj[j, i] == 1:
                    directed += 1
                elif adj[i, j] == -1 and adj[j, i] == -1:
                    undirected += 1
                elif adj[i, j] == 1 and adj[j, i] == 1:
                    undirected += 1
                else:
                    absent += 1

        results.append({
            "algorithm": display,
            "config_name": name,
            "time_seconds": round(elapsed, 2),
            "hamming_cost": round(cost, 4),
            "mismatched_edges": count,
            "directed_edges": directed,
            "undirected_edges": undirected,
            "absent_edges": absent,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    header = (f"{'Algorithm':<30} {'Hamming Cost':>13} {'Mismatches':>11} "
              f"{'Directed':>9} {'Undirected':>11} {'Absent':>7} {'Time(s)':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['algorithm']:<30} {r['hamming_cost']:>13.4f} {r['mismatched_edges']:>11} "
              f"{r['directed_edges']:>9} {r['undirected_edges']:>11} {r['absent_edges']:>7} "
              f"{r['time_seconds']:>8.2f}")

    summary_path = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Causal Discovery Algorithm Comparison on LUMIERE\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Train samples: {len(dataset.data['train'])}\n")
        f.write("Discretization: quantile-based, 5 bins for continuous concepts\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results:
            f.write(f"{r['algorithm']:<30} {r['hamming_cost']:>13.4f} {r['mismatched_edges']:>11} "
                    f"{r['directed_edges']:>9} {r['undirected_edges']:>11} {r['absent_edges']:>7} "
                    f"{r['time_seconds']:>8.2f}\n")

    json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - comparison_summary.txt")
    print(f"  - comparison_results.json")
    print(f"  - *_graph.png (one per algorithm + true_graph)")


if __name__ == "__main__":
    run_comparison()
