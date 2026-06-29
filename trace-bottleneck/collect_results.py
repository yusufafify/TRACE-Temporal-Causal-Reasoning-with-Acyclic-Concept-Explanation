#!/usr/bin/env python3
"""Aggregate multi-seed experiment results into summary tables and LaTeX output.

Usage:
    python collect_results.py [--results experiment_results.json]
"""

import argparse
import json
import os
import pickle
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = ['PD', 'SD', 'PR', 'CR']

EXPERIMENT_DISPLAY = {
    'full': 'Full model (Seg-Guided C2BM)',
    'ablA': 'No temporal diff',
    'ablB': 'No seg pretraining',
    'ablC': 'No bottleneck (cat_latent)',
}


def load_pickle_safe(path):
    """Load a pickle file, returning None on error."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def extract_metrics(output_dir):
    """Extract all relevant metrics from a single run's output directory."""
    if output_dir is None or not os.path.exists(output_dir):
        return None

    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        return None

    metrics = {}

    # Task accuracy
    y_data = load_pickle_safe(os.path.join(results_dir, 'y_accuracy.pkl'))
    if y_data:
        metrics['task_accuracy'] = y_data.get('_baseline', 0)

    # Concept accuracy
    c_data = load_pickle_safe(os.path.join(results_dir, 'c_accuracy.pkl'))
    if c_data:
        concept_accs = {}
        concept_f1s = {}
        for k, v in c_data.items():
            if k.endswith('_f1_macro'):
                cname = k.replace('_f1_macro', '')
                concept_f1s[cname] = v
            else:
                concept_accs[k] = v
        metrics['concept_accuracy'] = concept_accs
        metrics['concept_f1'] = concept_f1s
        if concept_accs:
            metrics['mean_concept_accuracy'] = np.mean(list(concept_accs.values()))

    # Confusion matrix (argmax)
    cm_data = load_pickle_safe(os.path.join(results_dir, 'confusion_matrix.pkl'))
    if cm_data:
        cm = np.array(cm_data['confusion_matrix'])
        metrics['confusion_matrix'] = cm

    # Threshold-optimized results
    cm_thresh = load_pickle_safe(os.path.join(results_dir, 'confusion_matrix_threshold.pkl'))
    if cm_thresh:
        metrics['threshold_macro_f1'] = cm_thresh.get('threshold_macro_f1', 0)
        metrics['val_macro_f1'] = cm_thresh.get('val_macro_f1', 0)
        metrics['best_boosts'] = cm_thresh.get('best_boosts', (1.0, 1.0))
        metrics['confusion_matrix_threshold'] = np.array(cm_thresh.get('confusion_matrix_threshold', []))

    # Single concept interventions
    interv_data = load_pickle_safe(os.path.join(results_dir, 'single_c_interventions_on_y.pkl'))
    if interv_data:
        baseline = interv_data.get('_baseline', 0)
        metrics['intervention_baseline'] = baseline
        # max accuracy after single-concept intervention
        interv_vals = {k: v for k, v in interv_data.items()
                       if k != '_baseline' and not k.endswith('_f1_macro')}
        if interv_vals:
            best_intervention = max(interv_vals.values())
            metrics['intervention_best_single'] = best_intervention
            metrics['intervention_delta_single'] = best_intervention - baseline
        metrics['intervention_single'] = interv_vals

    # Level interventions (oracle)
    level_data = load_pickle_safe(os.path.join(results_dir, 'level_interventions_on_y.pkl'))
    if level_data:
        metrics['intervention_levels'] = level_data
        levels = sorted(level_data.keys(), key=lambda k: int(k.split()[-1]))
        vals = [level_data[k] for k in levels]
        metrics['intervention_oracle'] = max(vals)
        metrics['intervention_delta_oracle'] = max(vals) - vals[0]

    # CaCE per concept
    cace_data = load_pickle_safe(os.path.join(results_dir, 'cace_per_concept.pkl'))
    if cace_data:
        metrics['cace'] = cace_data

    # Compute macro F1 from confusion matrix
    if cm_data:
        cm = np.array(cm_data['confusion_matrix'])
        per_class_recall = []
        per_class_precision = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i].sum() - tp
            fp = cm[:, i].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            per_class_precision.append(prec)
            per_class_recall.append(rec)
        per_class_f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0
                        for p, r in zip(per_class_precision, per_class_recall)]
        metrics['macro_f1'] = np.mean(per_class_f1)
        metrics['macro_precision'] = np.mean(per_class_precision)
        metrics['macro_recall'] = np.mean(per_class_recall)
        metrics['per_class_f1'] = per_class_f1
        metrics['per_class_recall'] = per_class_recall

    return metrics


def aggregate_metrics(runs_metrics):
    """Compute mean ± std over a list of metric dicts."""
    if not runs_metrics:
        return {}

    keys_to_aggregate = [
        'task_accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
        'mean_concept_accuracy', 'threshold_macro_f1', 'val_macro_f1',
        'intervention_baseline', 'intervention_best_single',
        'intervention_delta_single', 'intervention_oracle',
        'intervention_delta_oracle',
    ]

    agg = {}
    for key in keys_to_aggregate:
        vals = [m[key] for m in runs_metrics if key in m]
        if vals:
            agg[key] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}

    # Aggregate CaCE per concept
    all_cace = [m['cace'] for m in runs_metrics if 'cace' in m]
    if all_cace:
        concept_names = list(all_cace[0].keys())
        cace_agg = {}
        for cname in concept_names:
            tvs = [c[cname]['cace_tv'] for c in all_cace if cname in c]
            if tvs:
                cace_agg[cname] = {'mean': np.mean(tvs), 'std': np.std(tvs), 'n': len(tvs)}
        agg['cace'] = cace_agg

    # Aggregate concept-level accuracy
    all_c_acc = [m['concept_accuracy'] for m in runs_metrics if 'concept_accuracy' in m]
    if all_c_acc:
        concept_names = list(all_c_acc[0].keys())
        c_acc_agg = {}
        for cname in concept_names:
            vals = [c[cname] for c in all_c_acc if cname in c]
            if vals:
                c_acc_agg[cname] = {'mean': np.mean(vals), 'std': np.std(vals)}
        agg['concept_accuracy'] = c_acc_agg

    return agg


def print_table(header, rows, col_widths=None):
    """Print a nicely formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [header] + rows) + 2
                      for i in range(len(header))]

    separator = '+' + '+'.join('-' * w for w in col_widths) + '+'
    fmt = '|' + '|'.join(f'{{:^{w}}}' for w in col_widths) + '|'

    print(separator)
    print(fmt.format(*header))
    print(separator)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(separator)


def format_mean_std(d, key):
    """Format mean±std string."""
    if key not in d:
        return '—'
    info = d[key]
    return f"{info['mean']:.3f} ± {info['std']:.3f}"


def main():
    parser = argparse.ArgumentParser(description='Collect experiment results')
    parser.add_argument('--results', type=str, default='experiment_results.json',
                        help='Path to experiment_results.json')
    parser.add_argument('--output', type=str, default='aggregated_results.json',
                        help='Output JSON path')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. Run run_all_experiments.py first.")
        sys.exit(1)

    with open(args.results) as f:
        experiment_data = json.load(f)

    # Group runs by experiment
    by_experiment = defaultdict(list)
    for run in experiment_data['runs']:
        if run.get('success'):
            by_experiment[run['experiment']].append(run)

    print(f"\n{'#'*80}")
    print(f"  EXPERIMENT RESULTS AGGREGATION")
    print(f"{'#'*80}")

    # Extract metrics for each run
    all_aggregated = {}
    all_per_run = {}

    for exp_key, runs in by_experiment.items():
        print(f"\n  {EXPERIMENT_DISPLAY.get(exp_key, exp_key)}: {len(runs)} runs")
        per_run_metrics = []
        for run in runs:
            m = extract_metrics(run['output_dir'])
            if m:
                per_run_metrics.append(m)
                print(f"    seed={run['seed']}: acc={m.get('task_accuracy', 0):.3f} "
                      f"F1={m.get('macro_f1', 0):.3f} "
                      f"thresh_F1={m.get('threshold_macro_f1', 0):.3f}")
            else:
                print(f"    seed={run['seed']}: MISSING RESULTS ({run['output_dir']})")

        agg = aggregate_metrics(per_run_metrics)
        all_aggregated[exp_key] = agg
        all_per_run[exp_key] = per_run_metrics

    # TABLE 1: Main comparison
    print(f"\n\n{'='*80}")
    print("  TABLE 1: Multi-Seed Evaluation (mean ± std, n=5 seeds)")
    print(f"{'='*80}\n")

    header = ['Model', 'Accuracy', 'Macro F1', 'Thresh F1', 'Interv Δ (oracle)']
    rows = []
    for exp_key in ['full', 'ablA', 'ablB', 'ablC']:
        if exp_key not in all_aggregated:
            continue
        agg = all_aggregated[exp_key]
        rows.append([
            EXPERIMENT_DISPLAY.get(exp_key, exp_key),
            format_mean_std(agg, 'task_accuracy'),
            format_mean_std(agg, 'macro_f1'),
            format_mean_std(agg, 'threshold_macro_f1'),
            format_mean_std(agg, 'intervention_delta_oracle'),
        ])
    print_table(header, rows, col_widths=[35, 18, 18, 18, 22])

    # TABLE 2: CaCE Ranking
    if 'full' in all_aggregated and 'cace' in all_aggregated['full']:
        print(f"\n\n{'='*80}")
        print("  TABLE 2: Per-Concept CaCE (Concept Causal Effect) — Full Model")
        print(f"{'='*80}\n")

        cace = all_aggregated['full']['cace']
        sorted_cace = sorted(cace.items(), key=lambda x: x[1]['mean'], reverse=True)

        header = ['Rank', 'Concept', 'CaCE (TV)', 'n']
        rows = []
        for rank, (cname, info) in enumerate(sorted_cace, 1):
            rows.append([
                rank,
                cname,
                f"{info['mean']:.4f} ± {info['std']:.4f}",
                info['n'],
            ])
        print_table(header, rows, col_widths=[6, 35, 22, 6])

        # Expected vs learned ranking
        expected_ranking = [
            'delta_enhancing_percent',
            'vol_pd_flag',
            'delta_spd_percent',
            'spd_pd_flag',
            'vol_pr_flag',
            'spd_pr_flag',
            'new_lesion_flag',
            'delta_enhancing_absolute',
            'delta_spd_absolute',
            'followup_enhancing_volume_cm3',
            'followup_spd_cm2',
            'enhancing_tumor_volume_cm3',
            'baseline_spd_cm2',
            'time_gap',
            'delta_non_enhancing_percent',
            'delta_non_enhancing_absolute',
            'followup_non_enhancing_volume_cm3',
            'non_enhancing_volume_cm3',
        ]
        learned_ranking = [name for name, _ in sorted_cace]

        print("\n  Expected clinical ranking vs. Learned ranking:")
        print(f"    {'Expected':<35} {'Learned':<35} {'Match?'}")
        print(f"    {'─'*35} {'─'*35} {'─'*8}")
        for i, exp in enumerate(expected_ranking):
            learned = learned_ranking[i] if i < len(learned_ranking) else '—'
            match = '✓' if exp == learned else '✗'
            print(f"    {exp:<35} {learned:<35} {match}")

        # Kendall tau correlation
        try:
            from scipy.stats import kendalltau
            exp_ranks = {name: i for i, name in enumerate(expected_ranking)}
            learned_ranks = {name: i for i, name in enumerate(learned_ranking)}
            common = set(exp_ranks.keys()) & set(learned_ranks.keys())
            if len(common) >= 3:
                exp_order = [exp_ranks[n] for n in sorted(common, key=lambda x: exp_ranks[x])]
                learn_order = [learned_ranks[n] for n in sorted(common, key=lambda x: exp_ranks[x])]
                tau, p_val = kendalltau(exp_order, learn_order)
                print(f"\n  Kendall's τ = {tau:.3f} (p = {p_val:.3f})")
        except ImportError:
            pass

    # TABLE 3: Concept-level accuracy
    if 'full' in all_aggregated and 'concept_accuracy' in all_aggregated['full']:
        print(f"\n\n{'='*80}")
        print("  TABLE 3: Per-Concept Accuracy — Full Model")
        print(f"{'='*80}\n")

        c_acc = all_aggregated['full']['concept_accuracy']
        header = ['Concept', 'Accuracy']
        rows = []
        for cname, info in sorted(c_acc.items(), key=lambda x: x[1]['mean'], reverse=True):
            rows.append([cname, f"{info['mean']:.3f} ± {info['std']:.3f}"])
        print_table(header, rows, col_widths=[35, 22])

    # TABLE 4: Threshold optimization effect
    if 'full' in all_aggregated:
        print(f"\n\n{'='*80}")
        print("  TABLE 4: Threshold Optimization Effect")
        print(f"{'='*80}\n")

        agg = all_aggregated['full']
        header = ['Metric', 'Argmax F1', 'Threshold-opt F1', 'Δ F1']
        f1_mean = agg.get('macro_f1', {}).get('mean', 0)
        thresh_mean = agg.get('threshold_macro_f1', {}).get('mean', 0)
        delta = thresh_mean - f1_mean
        rows = [[
            'Macro F1',
            format_mean_std(agg, 'macro_f1'),
            format_mean_std(agg, 'threshold_macro_f1'),
            f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
        ]]
        print_table(header, rows, col_widths=[12, 18, 22, 10])

    # Save aggregated results
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for exp_key, agg in all_aggregated.items():
        serializable[exp_key] = {}
        for key, val in agg.items():
            if isinstance(val, dict):
                serializable[exp_key][key] = {
                    k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v)
                    for k, v in val.items()
                }
            else:
                serializable[exp_key][key] = convert(val)

    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2, default=convert)
    print(f"\n  Aggregated results saved to: {args.output}")

    # LaTeX table
    print(f"\n\n{'='*80}")
    print("  LATEX TABLE (copy-paste for paper)")
    print(f"{'='*80}\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Ablation study results (mean $\pm$ std, $n=5$ seeds)}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Model & Macro F1 $\uparrow$ & Thresh. F1 $\uparrow$ & Interv. $\Delta$ $\uparrow$ \\")
    print(r"\midrule")
    for exp_key in ['ablA', 'ablB', 'ablC', 'full']:
        if exp_key not in all_aggregated:
            continue
        agg = all_aggregated[exp_key]
        name = EXPERIMENT_DISPLAY.get(exp_key, exp_key)
        if exp_key == 'full':
            name = r'\textbf{' + name + '}'
        f1 = format_mean_std(agg, 'macro_f1')
        tf1 = format_mean_std(agg, 'threshold_macro_f1')
        delta = format_mean_std(agg, 'intervention_delta_oracle')
        # Escape ± for LaTeX
        f1_tex = f1.replace('±', r'$\pm$')
        tf1_tex = tf1.replace('±', r'$\pm$')
        delta_tex = delta.replace('±', r'$\pm$')
        print(f"{name} & {f1_tex} & {tf1_tex} & {delta_tex} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print(f"\n{'#'*80}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
