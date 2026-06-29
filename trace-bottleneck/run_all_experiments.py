#!/usr/bin/env python3
"""Multi-seed experiment runner for full model + ablation studies.

Usage:
    python run_all_experiments.py [--seeds 0 1 2 3 4] [--experiments full ablA ablB ablC]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


EXPERIMENTS = {
    'full': {
        'name': 'Full Model (Seg-Guided C2BM)',
        'overrides': [],
    },
    'ablA': {
        'name': 'Ablation A: No Temporal Difference',
        'overrides': ['dataset.loader.input_mode=seg_guided_no_temporal'],
    },
    'ablB': {
        'name': 'Ablation B: No Seg Pretraining (ImageNet)',
        'overrides': ['dataset.seg_guided_pretrain_epochs=0'],
    },
    'ablC': {
        'name': 'Ablation C: No Bottleneck (cat_latent)',
        'overrides': ['model.cat_latent=true'],
    },
}


def find_latest_output_dir(base_dir='outputs'):
    """Find the most recently created Hydra output directory."""
    latest_time = 0
    latest_dir = None
    base = Path(base_dir)
    if not base.exists():
        return None
    for date_dir in sorted(base.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir(), reverse=True):
            if not time_dir.is_dir():
                continue
            mtime = time_dir.stat().st_mtime
            if mtime > latest_time:
                latest_time = mtime
                latest_dir = str(time_dir)
        break  # only check most recent date
    return latest_dir


def run_experiment(experiment_key, seed, dry_run=False):
    """Run a single experiment and return (output_dir, success, duration)."""
    exp = EXPERIMENTS[experiment_key]
    cmd = ['python', 'main.py', 'dataset=lumiere', f'seed={seed}']
    cmd.extend(exp['overrides'])

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp['name']}")
    print(f"  Seed:       {seed}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return None, True, 0

    start = time.time()
    # Record the time before running so we can find the output dir
    pre_run_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200  # 2h max per run
        )
        duration = time.time() - start
        success = result.returncode == 0

        if not success:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  Last 30 lines of stderr:")
            for line in result.stderr.strip().split('\n')[-30:]:
                print(f"    {line}")
        else:
            print(f"  SUCCESS ({duration:.0f}s)")

        # Find the output directory created after pre_run_time
        output_dir = None
        base = Path('outputs')
        if base.exists():
            for date_dir in sorted(base.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for time_dir in sorted(date_dir.iterdir(), reverse=True):
                    if not time_dir.is_dir():
                        continue
                    if time_dir.stat().st_mtime >= pre_run_time:
                        output_dir = str(time_dir)
                        break
                if output_dir:
                    break

        print(f"  Output dir: {output_dir}")
        return output_dir, success, duration

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>2h)")
        return None, False, 7200
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, False, time.time() - start


def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='Seeds to run (default: 0 1 2 3 4)')
    parser.add_argument('--experiments', nargs='+', default=['full', 'ablA', 'ablB', 'ablC'],
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiments to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing results JSON to resume from')
    args = parser.parse_args()

    results_file = 'experiment_results.json'
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            all_results = json.load(f)
        print(f"Resuming from {args.resume} ({len(all_results['runs'])} previous runs)")
    else:
        all_results = {
            'start_time': datetime.now().isoformat(),
            'seeds': args.seeds,
            'experiments': args.experiments,
            'runs': [],
        }

    # Determine completed runs
    completed = set()
    for run in all_results['runs']:
        if run.get('success'):
            completed.add((run['experiment'], run['seed']))

    total = len(args.seeds) * len(args.experiments)
    remaining = total - len(completed)
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT PLAN")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Experiments: {args.experiments}")
    print(f"  Total runs:  {total}  (completed: {len(completed)}, remaining: {remaining})")
    print(f"{'#'*70}")

    run_idx = 0
    for exp_key in args.experiments:
        for seed in args.seeds:
            run_idx += 1
            if (exp_key, seed) in completed:
                print(f"\n[{run_idx}/{total}] {exp_key} seed={seed} — already complete, skipping")
                continue

            print(f"\n[{run_idx}/{total}] Running {exp_key} seed={seed}")
            output_dir, success, duration = run_experiment(exp_key, seed, dry_run=args.dry_run)

            run_info = {
                'experiment': exp_key,
                'seed': seed,
                'output_dir': output_dir,
                'success': success,
                'duration_s': round(duration, 1),
                'timestamp': datetime.now().isoformat(),
            }
            all_results['runs'].append(run_info)

            # Save after each run for crash recovery
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Progress saved to {results_file}")

    all_results['end_time'] = datetime.now().isoformat()
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results saved to: {results_file}")
    print(f"  Run `python collect_results.py` to aggregate and report")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
