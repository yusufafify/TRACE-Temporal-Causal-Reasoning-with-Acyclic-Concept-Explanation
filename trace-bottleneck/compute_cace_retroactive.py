#!/usr/bin/env python3
"""Retroactively compute per-concept CaCE for runs that are missing it.

Usage:
    python compute_cace_retroactive.py
"""

import json
import os
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Bypass PyTorch 2.6+ weights_only restriction
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

from omegaconf import DictConfig, OmegaConf
from src.data.utils import static_graph_collate
from src.models.layers.intervention import get_test_intervention_index


def compute_cace_for_run(output_dir, device='cuda'):
    """Compute CaCE for a single run using its checkpoint and config."""
    results_dir = os.path.join(output_dir, 'results')
    cace_path = os.path.join(results_dir, 'cace_per_concept.pkl')

    if os.path.exists(cace_path):
        print(f"  Already has CaCE: {output_dir}")
        return True

    # Find best checkpoint
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        print(f"  No checkpoints: {output_dir}")
        return False

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and 'last' not in f]
    if not ckpts:
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpts:
        print(f"  No .ckpt files: {output_dir}")
        return False

    ckpt_path = os.path.join(ckpt_dir, ckpts[0])
    print(f"  Loading checkpoint: {ckpt_path}")

    # Load Hydra config
    config_path = os.path.join(output_dir, '.hydra', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"  No config: {config_path}")
        return False

    cfg = OmegaConf.load(config_path)
    original_dir = os.getcwd()

    try:
        from hydra.utils import instantiate
        from src.data.preprocessing import preprocess_dataset
        from src.utils import update_config_from_data
        from src.data.utils import static_graph_collate

        # 1) Reconstruct dataset
        dataset_raw = instantiate(cfg.dataset.loader)
        dataset_raw._input_mode = cfg.dataset.loader.get('input_mode', 'mri')
        dataset_raw.split()
        dataset_raw.data['train'].update_lists()
        dataset_raw.data['val'].update_lists()
        dataset_raw.data['test'].update_lists()

        backbone = cfg.dataset.get('backbone', 'resnet18')
        dataset = preprocess_dataset(cfg, dataset_raw, device, backbone)

        # 2) Update config with data dimensions (input_size, c_info etc.)
        cfg = update_config_from_data(cfg, dataset)

        # 3) Load and clean the causal graph
        from src.utils import remove_problematic_edges, remove_cycles
        graph = dataset.load_ground_truth_graph()
        graph, dataset = remove_problematic_edges(graph, dataset)
        y_index = list(graph.index).index(dataset.y_info['names'][0])
        graph = remove_cycles(graph, y_index)

        # 4) Build intervention policy
        policy_cfg = cfg.dataset.get('policy', [])
        if not policy_cfg:
            interv_policy = []
        else:
            from src.utils import get_intervention_policy
            interv_policy, _ = get_intervention_policy(policy_cfg, graph, graph, y_index)

        # 5) Update config with graph
        from src.utils import maybe_update_config_with_graph
        cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)

        # Register graph on dataset splits
        for split in dataset.data:
            dataset.data[split].register_graph(graph)

        # 6) Instantiate model + engine from config
        engine = instantiate(cfg.engine)
        engine.to(device)

        # Load checkpoint weights
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        engine.load_state_dict(ckpt['state_dict'], strict=False)
        engine.eval()
        print(f"  Loaded checkpoint weights")

        model = engine.model
        c_names = engine.c_names
        n_concepts = len(c_names)

        # 6) Create test dataloader
        test_ds = dataset.data['test']
        test_dl = DataLoader(test_ds, batch_size=16,
                             collate_fn=static_graph_collate, num_workers=0)

        # Get concept cardinalities
        if hasattr(model, 'combo_info'):
            cardinalities = model.combo_info['cardinality'][:n_concepts]
        else:
            cardinalities = dataset.c_info['cardinality']

        # Accumulate concepts from test set
        all_c = []
        for batch in test_dl:
            all_c.append(batch['c'])
        all_c = torch.cat(all_c, dim=0)

        # Compute high/low values
        high_vals, low_vals = [], []
        for ci in range(n_concepts):
            card = cardinalities[ci]
            if card == 1:
                high_vals.append(float(all_c[:, ci].quantile(0.9)))
                low_vals.append(float(all_c[:, ci].quantile(0.1)))
            else:
                high_vals.append(1.0)
                low_vals.append(0.0)

        print(f"  Computing CaCE for {n_concepts} concepts...")
        cace_results = {}

        for ci, c_name in enumerate(c_names):
            if c_name in getattr(model, 'virtual_roots', []):
                continue

            probs_high_all, probs_low_all = [], []
            with torch.no_grad():
                for batch in test_dl:
                    x = batch['x'].to(device)
                    c = batch['c'].to(device)

                    interv_idx = torch.zeros_like(c)
                    interv_idx[:, ci] = 1.0

                    # do(c_i = high)
                    c_high = c.clone()
                    c_high[:, ci] = high_vals[ci]
                    y_out_h, c_out_h = model(x=x, c=c_high, intervention_index=interv_idx)
                    y_hat_h, _ = model.filter_output_for_metric(y_out_h, c_out_h)
                    probs_high_all.append(y_hat_h.detach().cpu())

                    # do(c_i = low)
                    c_low = c.clone()
                    c_low[:, ci] = low_vals[ci]
                    y_out_l, c_out_l = model(x=x, c=c_low, intervention_index=interv_idx)
                    y_hat_l, _ = model.filter_output_for_metric(y_out_l, c_out_l)
                    probs_low_all.append(y_hat_l.detach().cpu())

            probs_high = torch.cat(probs_high_all, dim=0)
            probs_low = torch.cat(probs_low_all, dim=0)

            avg_high = probs_high.mean(dim=0)
            avg_low = probs_low.mean(dim=0)

            tv_distance = 0.5 * (avg_high - avg_low).abs().sum().item()
            per_class = (avg_high - avg_low).tolist()

            cace_results[c_name] = {
                'cace_tv': tv_distance,
                'per_class': per_class,
                'high_val': high_vals[ci],
                'low_val': low_vals[ci],
                'cardinality': cardinalities[ci],
                'avg_probs_high': avg_high.tolist(),
                'avg_probs_low': avg_low.tolist(),
            }
            print(f"    {c_name}: CaCE(TV)={tv_distance:.4f}")

        # Save
        os.makedirs(results_dir, exist_ok=True)
        pickle.dump(cace_results, open(cace_path, 'wb'))
        print(f"  Saved CaCE to {cace_path}")

        # Print ranking
        sorted_cace = sorted(cace_results.items(), key=lambda x: x[1]['cace_tv'], reverse=True)
        print(f"  CaCE Ranking:")
        for rank, (name, info) in enumerate(sorted_cace, 1):
            print(f"    {rank}. {name}: {info['cace_tv']:.4f}")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)


def main():
    results_file = 'experiment_results.json'
    if not os.path.exists(results_file):
        print(f"No {results_file} found")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    missing = []
    for run in data['runs']:
        if not run.get('success'):
            continue
        output_dir = run['output_dir']
        if not output_dir:
            continue
        cace_path = os.path.join(output_dir, 'results', 'cace_per_concept.pkl')
        if not os.path.exists(cace_path):
            missing.append(run)

    print(f"Found {len(missing)} runs missing CaCE")
    for run in missing:
        print(f"\n{'='*60}")
        print(f"  {run['experiment']} seed={run['seed']}  → {run['output_dir']}")
        print(f"{'='*60}")
        compute_cace_for_run(run['output_dir'], device=device)


if __name__ == '__main__':
    main()
