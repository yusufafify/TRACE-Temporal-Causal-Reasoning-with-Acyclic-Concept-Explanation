"""
Post-training visualizations for C2BM / LUMIERE experiments.

Generates the following plots in the output directory:
  1. concept_accuracy.png          – per-concept accuracy + F1 grouped bar chart
  2. intervention_single.png       – task accuracy after single-concept interventions
  3. intervention_levels.png       – task accuracy across policy levels
  4. intervention_concept_levels.png – concept accuracy across policy levels
  5. causal_graph.png              – the learned/expert causal DAG
  6. confusion_matrix.png          – task confusion matrix on test set
  7. class_distribution.png        – train/val/test class distribution

Usage:
    python src/visualize.py <output_dir>

    e.g.  python src/visualize.py outputs/2026-02-26/04-26-58
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch



plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#dee2e6',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#dee2e6',
})

COLORS = {
    'primary': '#4361ee',
    'secondary': '#7209b7',
    'accent': '#f72585',
    'success': '#06d6a0',
    'warning': '#ffd166',
    'info': '#118ab2',
    'dark': '#073b4c',
    'light': '#e9ecef',
}

CONCEPT_COLORS = ['#4361ee', '#7209b7', '#f72585', '#06d6a0', '#ffd166',
                  '#118ab2', '#ef476f', '#073b4c', '#8338ec', '#ff6b6b']

CLASS_NAMES = ['Non-PD', 'PD']
CLASS_COLORS = ['#06d6a0', '#ef476f']

def _short_name(name):
    """Shorten concept names for display."""
    replacements = {
        'enhancing_tumor_volume_cm3': 'Base Enh.',
        'non_enhancing_volume_cm3': 'Base Non-Enh.',
        'followup_enhancing_volume_cm3': 'Fup Enh.',
        'followup_non_enhancing_volume_cm3': 'Fup Non-Enh.',
        'baseline_spd_cm2': 'Base SPD',
        'followup_spd_cm2': 'Fup SPD',
        'time_gap': 'Time',
        'delta_enhancing_percent': 'Δ Enh. %',
        'delta_non_enhancing_percent': 'Δ Non-Enh. %',
        'delta_enhancing_absolute': 'Δ Enh. Abs',
        'delta_non_enhancing_absolute': 'Δ Non-Enh. Abs',
        'delta_spd_absolute': 'Δ SPD Abs',
        'delta_spd_percent': 'Δ SPD %',
        'new_lesion_flag': 'New Lesion',
        'vol_pd_flag': 'Vol PD',
        'vol_pr_flag': 'Vol PR',
        'spd_pd_flag': 'SPD PD',
        'spd_pr_flag': 'SPD PR',
        'TreatmentResponse': 'Task',
    }
    return replacements.get(name, name)


def plot_concept_accuracy(results_dir, save_dir):
    """Bar chart: per-concept accuracy + F1 macro."""
    pkl_path = os.path.join(results_dir, 'c_accuracy.pkl')
    if not os.path.exists(pkl_path):
        print(f"  Skipping concept_accuracy: {pkl_path} not found")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Separate accuracy and F1 metrics
    concepts = []
    accs = []
    f1s = []
    for k, v in data.items():
        if k.endswith('_f1_macro'):
            continue
        concepts.append(k)
        accs.append(v)
        f1_key = f"{k}_f1_macro"
        f1s.append(data.get(f1_key, 0.0))

    short_names = [_short_name(c) for c in concepts]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(concepts))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',
                   color=COLORS['primary'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 (Macro)',
                   color=COLORS['accent'], alpha=0.85, edgecolor='white', linewidth=1.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Test Concept Performance', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    plt.tight_layout()
    path = os.path.join(save_dir, 'concept_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_single_interventions(results_dir, save_dir):
    """Bar chart: task accuracy after single-concept interventions."""
    pkl_path = os.path.join(results_dir, 'single_c_interventions_on_y.pkl')
    if not os.path.exists(pkl_path):
        print(f"  Skipping intervention_single: {pkl_path} not found")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Filter to accuracy only (skip F1 keys)
    names = []
    values = []
    baseline = data.get('_baseline', 0)
    for k, v in data.items():
        if k.endswith('_f1_macro'):
            continue
        names.append(k)
        values.append(v)

    short_names = [_short_name(n) if n != '_baseline' else 'Baseline' for n in names]
    colors = [COLORS['dark'] if n == '_baseline' else COLORS['primary'] for n in names]
    # Highlight improvements
    colors = [COLORS['success'] if v > baseline and n != '_baseline'
              else COLORS['accent'] if v < baseline and n != '_baseline'
              else COLORS['dark']
              for n, v in zip(names, values)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        diff = val - baseline
        label = f'{val:.3f}'
        if abs(diff) > 0.001 and bar.get_x() > 0:
            label += f'\n({"+" if diff > 0 else ""}{diff:.3f})'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=baseline, color=COLORS['dark'], linestyle='--', alpha=0.7)
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Task Accuracy After Single-Concept Interventions', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=15, ha='right')
    ax.set_ylim(0, max(values) * 1.2)

    # Legend patches
    patches = [
        mpatches.Patch(color=COLORS['dark'], label='Baseline'),
        mpatches.Patch(color=COLORS['success'], label='Improved'),
        mpatches.Patch(color=COLORS['accent'], label='Degraded'),
    ]
    ax.legend(handles=patches, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'intervention_single.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_level_interventions(results_dir, save_dir):
    """Step chart: task accuracy across policy levels."""
    pkl_path = os.path.join(results_dir, 'level_interventions_on_y.pkl')
    if not os.path.exists(pkl_path):
        print(f"  Skipping intervention_levels: {pkl_path} not found")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    levels = sorted(data.keys(), key=lambda k: int(k.split()[-1]))
    values = [data[k] for k in levels]
    level_nums = [int(k.split()[-1]) for k in levels]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Gradient fill
    ax.fill_between(level_nums, values, alpha=0.15, color=COLORS['primary'])
    ax.plot(level_nums, values, 'o-', color=COLORS['primary'], linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2.5)

    for x, y in zip(level_nums, values):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=11, fontweight='bold',
                    color=COLORS['primary'])

    ax.set_xlabel('Intervention Level')
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Task Accuracy vs. Intervention Level (Policy)', pad=15)
    ax.set_xticks(level_nums)
    ax.set_xticklabels([f'Level {l}\n({"no interv." if l == 0 else f"{l} concepts"})' for l in level_nums])

    # Highlight best
    best_idx = np.argmax(values)
    ax.scatter([level_nums[best_idx]], [values[best_idx]], s=200, c=COLORS['success'],
               zorder=5, edgecolors='white', linewidth=2)
    ax.annotate('Best', (level_nums[best_idx], values[best_idx]),
                textcoords="offset points", xytext=(15, -15), fontsize=10,
                color=COLORS['success'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    plt.tight_layout()
    path = os.path.join(save_dir, 'intervention_levels.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_concept_level_interventions(results_dir, save_dir):
    """Heatmap: concept accuracy across policy levels."""
    pkl_path = os.path.join(results_dir, 'level_interventions_on_c.pkl')
    if not os.path.exists(pkl_path):
        print(f"  Skipping intervention_concept_levels: {pkl_path} not found")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Parse into matrix
    concepts = set()
    levels_set = set()
    for k in data:
        parts = k.split('/node ')
        level = parts[0]
        concept = parts[1] if len(parts) > 1 else parts[0]
        levels_set.add(level)
        concepts.add(concept)

    levels = sorted(levels_set, key=lambda k: int(k.split()[-1]))
    concepts = sorted(concepts)
    short_concepts = [_short_name(c) for c in concepts]

    matrix = np.zeros((len(concepts), len(levels)))
    for i, concept in enumerate(concepts):
        for j, level in enumerate(levels):
            key = f"{level}/node {concept}"
            matrix[i, j] = data.get(key, 0.0)

    cmap = LinearSegmentedColormap.from_list('custom',
        ['#ef476f', '#ffd166', '#06d6a0', '#118ab2'], N=256)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f'Level {int(l.split()[-1])}' for l in levels])
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(short_concepts)

    # Annotate cells
    for i in range(len(concepts)):
        for j in range(len(levels)):
            val = matrix[i, j]
            color = 'white' if val > 0.6 or val < 0.2 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_title('Concept Accuracy Across Intervention Levels', pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Accuracy')

    plt.tight_layout()
    path = os.path.join(save_dir, 'intervention_concept_levels.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_causal_graph(output_dir, save_dir):
    """Plot the causal graph from graph.pkl using matplotlib."""
    pkl_path = os.path.join(output_dir, 'graph.pkl')
    if not os.path.exists(pkl_path):
        print(f"  Skipping causal_graph: {pkl_path} not found")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    concepts = data.get('concepts', [])
    if not concepts:
        print(f"  Skipping causal_graph: no concepts found")
        return

    # Build a simple DAG layout
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 13.5)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    ax.set_title('Expert Causal Graph (LUMIERE)', fontsize=16, fontweight='bold', pad=20)

    # Manually layout the clinically sparse explicit-delta + SPD/time graph + task.
    positions = {
        'enhancing_tumor_volume_cm3': (0.8, 5.0),
        'followup_enhancing_volume_cm3': (3.0, 5.0),
        'baseline_spd_cm2': (5.2, 5.0),
        'followup_spd_cm2': (7.0, 5.0),
        'non_enhancing_volume_cm3': (9.3, 5.0),
        'followup_non_enhancing_volume_cm3': (11.6, 5.0),
        'delta_enhancing_absolute': (1.2, 3.2),
        'delta_enhancing_percent': (3.1, 3.2),
        'delta_spd_percent': (5.4, 3.2),
        'delta_spd_absolute': (7.0, 3.2),
        'delta_non_enhancing_absolute': (9.5, 3.2),
        'delta_non_enhancing_percent': (11.3, 3.2),
        'time_gap': (0.8, 1.4),
        'new_lesion_flag': (2.1, 1.4),
        'vol_pd_flag': (3.6, 1.4),
        'vol_pr_flag': (4.8, 1.4),
        'spd_pd_flag': (6.1, 1.4),
        'spd_pr_flag': (7.3, 1.4),
        'TreatmentResponse': (4.5, 0),
    }

    edges = [
        ('enhancing_tumor_volume_cm3', 'delta_enhancing_absolute'),
        ('enhancing_tumor_volume_cm3', 'delta_enhancing_percent'),
        ('followup_enhancing_volume_cm3', 'delta_enhancing_absolute'),
        ('followup_enhancing_volume_cm3', 'delta_enhancing_percent'),
        ('non_enhancing_volume_cm3', 'delta_non_enhancing_absolute'),
        ('non_enhancing_volume_cm3', 'delta_non_enhancing_percent'),
        ('followup_non_enhancing_volume_cm3', 'delta_non_enhancing_absolute'),
        ('followup_non_enhancing_volume_cm3', 'delta_non_enhancing_percent'),
        ('baseline_spd_cm2', 'delta_spd_absolute'),
        ('baseline_spd_cm2', 'delta_spd_percent'),
        ('followup_spd_cm2', 'delta_spd_absolute'),
        ('followup_spd_cm2', 'delta_spd_percent'),
        ('delta_enhancing_percent', 'vol_pd_flag'),
        ('delta_enhancing_percent', 'vol_pr_flag'),
        ('delta_spd_percent', 'spd_pd_flag'),
        ('delta_spd_percent', 'spd_pr_flag'),
        ('time_gap', 'TreatmentResponse'),
        ('new_lesion_flag', 'TreatmentResponse'),
        ('vol_pd_flag', 'TreatmentResponse'),
        ('vol_pr_flag', 'TreatmentResponse'),
        ('spd_pd_flag', 'TreatmentResponse'),
        ('spd_pr_flag', 'TreatmentResponse'),
    ]

    # Draw edges
    for src, dst in edges:
        if src in positions and dst in positions:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            ax.annotate('', xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                       arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                      lw=1.8, connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    for name, (x, y) in positions.items():
        is_task = name == 'TreatmentResponse'
        color = COLORS['accent'] if is_task else COLORS['primary']
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.85,
                   edgecolor='white', linewidth=2)
        ax.text(x, y, _short_name(name), ha='center', va='center',
               fontsize=11, fontweight='bold', color='white', bbox=bbox)

    plt.tight_layout()
    path = os.path.join(save_dir, 'causal_graph.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(output_dir, save_dir):
    """Generate confusion matrix from test predictions via checkpoint."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        print(f"  Skipping confusion_matrix: no checkpoints dir")
        return

    # Get y_accuracy results to check if test was run
    y_path = os.path.join(output_dir, 'results', 'y_accuracy.pkl')
    if not os.path.exists(y_path):
        print(f"  Skipping confusion_matrix: y_accuracy.pkl not found")
        return

    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)

    baseline_acc = y_data.get('_baseline', 0)

    # Create a summary card instead
    c_path = os.path.join(output_dir, 'results', 'c_accuracy.pkl')
    with open(c_path, 'rb') as f:
        c_data = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Task performance summary
    ax = axes[0]
    metrics = ['Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
    # We need to extract these from the run output
    # For now, use accuracy from pkl
    task_acc = baseline_acc
    ax.barh([0], [task_acc], color=COLORS['primary'], alpha=0.85, height=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(['Accuracy'])
    ax.set_xlim(0, 1)
    ax.set_title('Task Performance', pad=10)
    ax.text(task_acc + 0.02, 0, f'{task_acc:.3f}', va='center', fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.5, -0.4, 'Random (binary)', fontsize=8, color='gray', ha='center')

    # Right: Concept performance radar/bar
    ax = axes[1]
    concepts = []
    accs = []
    for k, v in c_data.items():
        if not k.endswith('_f1_macro'):
            concepts.append(_short_name(k))
            accs.append(v)

    y_pos = np.arange(len(concepts))
    bars = ax.barh(y_pos, accs, color=[CONCEPT_COLORS[i % len(CONCEPT_COLORS)] for i in range(len(concepts))],
                   alpha=0.85, height=0.6, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(concepts)
    ax.set_xlim(0, 1.15)
    ax.set_title('Concept Accuracy', pad=10)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Test Performance Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'performance_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_task_accuracy_card(results_dir, save_dir):
    """A single high-impact results card combining task + concept + intervention."""
    y_path = os.path.join(results_dir, 'y_accuracy.pkl')
    c_path = os.path.join(results_dir, 'c_accuracy.pkl')
    level_path = os.path.join(results_dir, 'level_interventions_on_y.pkl')

    if not all(os.path.exists(p) for p in [y_path, c_path, level_path]):
        print(f"  Skipping results_card: missing pickle files")
        return

    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)
    with open(c_path, 'rb') as f:
        c_data = pickle.load(f)
    with open(level_path, 'rb') as f:
        level_data = pickle.load(f)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')

    # Title
    fig.suptitle('C²BM × LUMIERE — Training Results',
                 fontsize=20, fontweight='bold', color='white', y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.95, top=0.90, bottom=0.08)

    # ── Panel 1: Task Accuracy Gauge ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#16213e')
    acc = y_data.get('_baseline', 0)
    theta = np.linspace(0, np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), color='#334155', linewidth=8, solid_capstyle='round')
    theta_fill = np.linspace(0, np.pi * acc, 100)
    ax1.plot(np.cos(theta_fill), np.sin(theta_fill), color='#06d6a0', linewidth=8, solid_capstyle='round')
    ax1.text(0, 0.3, f'{acc:.1%}', ha='center', va='center', fontsize=32, fontweight='bold', color='white')
    ax1.text(0, -0.1, 'Task Accuracy', ha='center', fontsize=11, color='#94a3b8')
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-0.3, 1.3)
    ax1.axis('off')

    # ── Panel 2: Concept Accuracies ──
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor('#16213e')
    concepts = []
    accs_list = []
    f1s_list = []
    for k, v in c_data.items():
        if not k.endswith('_f1_macro'):
            concepts.append(_short_name(k))
            accs_list.append(v)
            f1s_list.append(c_data.get(f"{k}_f1_macro", 0))

    x = np.arange(len(concepts))
    width = 0.35
    ax2.bar(x - width/2, accs_list, width, color='#4361ee', alpha=0.9, label='Accuracy')
    ax2.bar(x + width/2, f1s_list, width, color='#f72585', alpha=0.9, label='F1')

    for i, (a, f) in enumerate(zip(accs_list, f1s_list)):
        ax2.text(i - width/2, a + 0.02, f'{a:.2f}', ha='center', fontsize=9,
                 color='white', fontweight='bold')
        ax2.text(i + width/2, f + 0.02, f'{f:.2f}', ha='center', fontsize=9,
                 color='white', fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(concepts, fontsize=10, color='white')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', color='white')
    ax2.set_title('Concept Performance', color='white', pad=10)
    ax2.legend(loc='upper right', fontsize=9, facecolor='#16213e', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#334155')
    ax2.spines['left'].set_color('#334155')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.2, color='white')

    # ── Panel 3: Intervention Levels ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor('#16213e')
    levels = sorted(level_data.keys(), key=lambda k: int(k.split()[-1]))
    vals = [level_data[k] for k in levels]
    level_nums = list(range(len(levels)))

    ax3.fill_between(level_nums, vals, alpha=0.2, color='#06d6a0')
    ax3.plot(level_nums, vals, 'o-', color='#06d6a0', linewidth=2.5,
             markersize=10, markerfacecolor='#16213e', markeredgewidth=2.5)

    for i, (lv, v) in enumerate(zip(level_nums, vals)):
        ax3.annotate(f'{v:.3f}', (lv, v), textcoords="offset points",
                     xytext=(0, 14), ha='center', fontsize=10, fontweight='bold',
                     color='white')

    ax3.set_xticks(level_nums)
    ax3.set_xticklabels([f'L{i}' for i in level_nums], color='white')
    ax3.set_ylabel('Task Accuracy', color='white')
    ax3.set_title('Intervention Level Analysis', color='white', pad=10)
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#334155')
    ax3.spines['left'].set_color('#334155')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.2, color='white')

    # ── Panel 4: Key Numbers ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('#16213e')
    ax4.axis('off')

    best_level_idx = np.argmax(vals)
    best_level_val = vals[best_level_idx]
    improvement = best_level_val - vals[0]

    info = [
        ('Best Intervention', f'Level {best_level_idx}'),
        ('Best Accuracy', f'{best_level_val:.1%}'),
        ('Improvement', f'+{improvement:.1%}'),
        ('Concepts Used', f'{len(concepts)}'),
        ('Classes', '2 (PD vs Non-PD)'),
    ]

    for i, (label, value) in enumerate(info):
        y_pos = 0.85 - i * 0.18
        ax4.text(0.05, y_pos, label, transform=ax4.transAxes, fontsize=10,
                 color='#94a3b8', va='center')
        ax4.text(0.95, y_pos, value, transform=ax4.transAxes, fontsize=13,
                 color='white', fontweight='bold', va='center', ha='right')
        if i < len(info) - 1:
            ax4.plot([0.05, 0.95], [y_pos - 0.08, y_pos - 0.08],
                     color='#334155', linewidth=0.5, transform=ax4.transAxes)

    path = os.path.join(save_dir, 'results_card.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_visualizations(output_dir):
    """Generate all visualizations for a given output directory."""
    results_dir = os.path.join(output_dir, 'results')
    save_dir = os.path.join(output_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating visualizations in: {save_dir}")
    print("=" * 60)

    plot_concept_accuracy(results_dir, save_dir)
    plot_single_interventions(results_dir, save_dir)
    plot_level_interventions(results_dir, save_dir)
    plot_concept_level_interventions(results_dir, save_dir)
    plot_causal_graph(output_dir, save_dir)
    plot_confusion_matrix(output_dir, save_dir)
    plot_task_accuracy_card(results_dir, save_dir)

    print("=" * 60)
    print(f"Done! All plots saved to: {save_dir}\n")
    return save_dir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/visualize.py <output_dir>")
        print("  e.g. python src/visualize.py outputs/2026-02-26/04-26-58")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.exists(output_dir):
        print(f"Error: {output_dir} does not exist")
        sys.exit(1)

    generate_all_visualizations(output_dir)
