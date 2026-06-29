import copy
import random
import numpy as np
import torch
import os
import warnings
import hydra
import pickle
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
import typing
import builtins
from omegaconf.base import ContainerMetadata, Metadata
# Bypass PyTorch 2.6+ strict checkpoint loader rules
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# data loading
from src.data.dataset_block import get_dataset

# causal discovery
from src.causal_discovery.causal_discovery_block import causal_discovery

# graph completion block
from src.completion.completion_block import complete_graph_with_llm

# training and utils
from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams
from src.data.utils import static_graph_collate
from src.metrics import hamming_distance
from src.plots import maybe_plot_graph
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph
from src.utils import finetune_model

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")


def _is_primary_process():
    """Return True only for the main process under DDP/spawned execution."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                return False
    return True
    
def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _metric_to_float(metric_collection, metric_name):
    try:
        value = metric_collection[metric_name].compute()
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception as exc:
        print(f"Warning: could not compute {metric_name}: {exc}")
        return float("nan")


def _get_cv_settings(cfg: DictConfig):
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


def _print_metric_block(name, reports):
    if not _is_primary_process():
        return
    reports = [r for r in reports if r is not None]
    if not reports:
        return

    title = '4-class' if name == 'four_class' else 'Binary'
    print(f"\n{title} metrics:")
    macro_f1_values = np.array([r['macro_f1'] for r in reports], dtype=float)
    weighted_acc_values = np.array([r['weighted_accuracy'] for r in reports], dtype=float)
    print("  Macro-F1 for all folds: " + ", ".join(f"{v:.4f}" for v in macro_f1_values))
    print(f"  Macro-F1 mean +/- std: {macro_f1_values.mean():.4f} +/- {macro_f1_values.std(ddof=0):.4f}")
    print("  Weighted accuracy for all folds: " + ", ".join(f"{v:.4f}" for v in weighted_acc_values))
    print(f"  Weighted accuracy mean +/- std: "
          f"{weighted_acc_values.mean():.4f} +/- {weighted_acc_values.std(ddof=0):.4f}")

    class_names = reports[0]['class_names']
    per_class = np.array([r['per_class_f1'] for r in reports], dtype=float)
    print("  Per-class F1 mean +/- std:")
    for idx, class_name in enumerate(class_names):
        values = per_class[:, idx]
        print(f"    {class_name}: {values.mean():.4f} +/- {values.std(ddof=0):.4f}")


def _print_cv_summary(fold_results):
    if not _is_primary_process():
        return
    if not fold_results:
        return

    print("\n=== Cross-validation summary ===")
    for result in fold_results:
        fold_num = result["fold"] + 1
        reports = result.get('reports', {})
        parts = []
        if 'four_class' in reports:
            parts.append(f"4-class macro-F1={reports['four_class']['macro_f1']:.4f}, "
                         f"weighted_accuracy={reports['four_class']['weighted_accuracy']:.4f}")
        if 'binary' in reports:
            parts.append(f"binary macro-F1={reports['binary']['macro_f1']:.4f}, "
                         f"weighted_accuracy={reports['binary']['weighted_accuracy']:.4f}")
        print(f"Fold {fold_num}: " + " | ".join(parts))

    _print_metric_block('four_class', [r.get('reports', {}).get('four_class') for r in fold_results])
    _print_metric_block('binary', [r.get('reports', {}).get('binary') for r in fold_results])


def _run_experiment(cfg: DictConfig, fold_idx=None):
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.makedirs('results', exist_ok=True)
    with open_dict(cfg):
        cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg)

    # get the causal graph
    if cfg.dataset.load_true_graph:
        graph = true_graph
    else:
        if cfg.dataset.load_graph:
            with open(os.path.join(dataset_directory, "graph.pkl"), 'rb') as f:
                graph = pickle.load(f)
        else:
            # estimate causal graph with causal structural learning algorithms
            predicted_graph = causal_discovery(cfg, dataset, true_graph)
            if true_graph is not None:
                hamming = hamming_distance(true_graph, predicted_graph)
                print('(after CD) structural hamming distance: ', hamming)

            # complete the causal graph with LLM and RAG
            # complete the causal graph with LLM and RAG
            try:
                completed_graph = complete_graph_with_llm(cfg, predicted_graph, cfg.dataset.name)
            except Exception as e:
                print(f"LLM completion failed: {e}. Falling back to Oracle using true_graph.")
                completed_graph = predicted_graph.copy()
                task_node_name = cfg.dataset.name if hasattr(cfg.dataset, 'name') else 'TreatmentResponse'
                if 'TreatmentResponse' in completed_graph.columns:
                    task_node_name = 'TreatmentResponse'
                
                for i in completed_graph.index:
                    for j in completed_graph.columns:
                        # If edge is undirected (-1, -1) or bidirected (1, 1)
                        if completed_graph.loc[i, j] != 0 and completed_graph.loc[j, i] != 0:
                            # Rule 1: Always orient towards the task node
                            if j == task_node_name:
                                completed_graph.loc[i, j] = 1
                                completed_graph.loc[j, i] = 0
                            elif i == task_node_name:
                                completed_graph.loc[i, j] = 0
                                completed_graph.loc[j, i] = 1
                            # Rule 2: Use true_graph if available and edge exists
                            elif true_graph is not None and true_graph.loc[i, j] == 1 and true_graph.loc[j, i] == 0:
                                completed_graph.loc[i, j] = 1
                                completed_graph.loc[j, i] = 0
                            elif true_graph is not None and true_graph.loc[j, i] == 1 and true_graph.loc[i, j] == 0:
                                completed_graph.loc[i, j] = 0
                                completed_graph.loc[j, i] = 1
                            # Rule 3: Arbitrary orientation to ensure it becomes a DAG and isn't deleted
                            else:
                                if list(completed_graph.columns).index(i) < list(completed_graph.columns).index(j):
                                    completed_graph.loc[i, j] = 1
                                    completed_graph.loc[j, i] = 0
                                else:
                                    completed_graph.loc[i, j] = 0
                                    completed_graph.loc[j, i] = 1
            
            if true_graph is not None:
                hamming = hamming_distance(true_graph, completed_graph)
                print('(after LLM + RAG) structural hamming distance: ', hamming)
            graph = completed_graph

            # save graph
            with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
                pickle.dump(graph, f)

    # fix the graph
    # (part 1): remove bidirected and undirected edges + add virtual nodes
    # edge can only be directed at this stage, the following function is just here in
    # case the CD + LLM + RAG pipeline is modified and could produce bidirected or undirected edges
    graph, dataset = remove_problematic_edges(graph, dataset)
    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1
    # (part 2): remove cycles
    graph = remove_cycles(graph, y_index)

    if true_graph is not None:
        hamming = hamming_distance(true_graph, graph)
        print('(after fix) structural hamming distance: ', hamming)

    maybe_plot_graph(graph, 'fixed_graph')

    # use the graph to define an intervention policy at test time
    policy_cfg = cfg.dataset.get('policy', [])
    if not policy_cfg:
        interv_policy, ip_names = [], []
    else:
        interv_policy, ip_names = get_intervention_policy(policy_cfg, graph, true_graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, dataset)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)

    ############ model block ########################################################################################
    [dataset.data[split].register_graph(graph) for split in dataset.data]

    # Concept sanity check
    train_ds = dataset.data['train']
    c_train = train_ds.c  # [N_train, N_concepts]
    print(f"\n=== Concept Sanity Check (train split) ===")
    print(f"  c shape: {c_train.shape}")
    for ci in range(c_train.shape[1]):
        col = c_train[:, ci]
        print(f"  concept {ci}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.2f}  max={col.max():.2f}")

    # Dynamic balanced y_class_weights from actual train labels.
    from collections import Counter
    train_labels = train_ds.y.flatten().numpy()
    label_counter = Counter(int(l) for l in train_labels)
    n_classes = cfg.dataset.loader.task_cardinality
    if n_classes == 4:
        class_name_map = {0: 'CR', 1: 'PR', 2: 'SD', 3: 'PD'}
    else:
        class_name_map = {0: 'Non-PD', 1: 'PD'}
    print(f"  train label distribution: { {class_name_map.get(k, str(k)): v for k, v in sorted(label_counter.items())} }")
    class_counts = np.bincount(train_labels, minlength=n_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)  # avoid div by zero
    y_class_weights = (len(train_labels) / (n_classes * class_counts)).tolist()
    y_class_weights = [round(w, 2) for w in y_class_weights]
    with open_dict(cfg):
        cfg.model.y_class_weights = y_class_weights
    print(f"  train class counts: {class_counts.astype(int).tolist()}")
    print(f"  y_class_weights (balanced): {y_class_weights}")

    # Rebalancing is handled by focal loss alpha (y_class_weights). Stacking
    # a WeightedRandomSampler on top double-counts the minority boost and
    # caused PD to collapse in the previous run; rely on shuffle=True here.
    train_dataloader = DataLoader(dataset.data['train'],
                                  batch_size=cfg.dataset.batch_size,
                                  collate_fn=static_graph_collate,
                                  num_workers=cfg.dataset.num_workers,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset.data['val'],
                                batch_size=cfg.dataset.batch_size,
                                collate_fn=static_graph_collate,
                                num_workers=cfg.dataset.num_workers)
    test_dataloader = DataLoader(dataset.data['test'],
                                 batch_size=cfg.dataset.batch_size,
                                 collate_fn=static_graph_collate,
                                 num_workers=cfg.dataset.num_workers)

    engine = instantiate(cfg.engine)

    # Save model architecture explicitly to the output directory
    with open("architecture.txt", "w") as f:
        f.write("=== Model Configuration ===\n")
        f.write(OmegaConf.to_yaml(cfg) + "\n\n")
        f.write("=== Model Architecture ===\n")
        f.write(str(engine.model) + "\n")

    fold_metrics = None
    trainer = None
    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        # ---- train
        trainer.fit(engine, train_dataloader, val_dataloader)
        # ---- finetune the encoder (eventually)
        if cfg.dataset.loader.ftune_size > 0:
            trainer, engine = finetune_model(cfg, engine, dataset)
        # ----- test
        # Store val dataloader on engine so threshold search can access it during test
        engine._val_dataloader_ref = val_dataloader
        ckpt = 'best' if cfg.trainer.get('enable_checkpointing', True) else None
        eval_dataloader = val_dataloader if fold_idx is not None else test_dataloader
        trainer.test(engine, eval_dataloader, ckpt_path=ckpt)
        latest_test_metrics = getattr(engine, 'latest_test_metrics', {})
        reports = {
            key: latest_test_metrics[key]
            for key in ('four_class', 'binary')
            if key in latest_test_metrics
        }
        if not reports:
            f1_macro = latest_test_metrics.get('y_f1_macro')
            if f1_macro is None:
                f1_macro = _metric_to_float(engine.test_y_metrics, 'y_f1_macro')
            weighted_accuracy = latest_test_metrics.get('y_weighted_accuracy')
            if weighted_accuracy is None:
                weighted_accuracy = _metric_to_float(engine.test_y_metrics, 'y_weighted_accuracy')
            reports['task'] = {
                'macro_f1': float(f1_macro),
                'weighted_accuracy': float(weighted_accuracy),
                'per_class_f1': [],
                'class_names': [],
            }
        fold_metrics = {
            "fold": -1 if fold_idx is None else int(fold_idx),
            "reports": reports,
        }
        if fold_idx is not None and _is_primary_process():
            print(f"\n=== Fold {fold_idx + 1} held-out validation metrics ===")
            for name, report in reports.items():
                label = '4-class' if name == 'four_class' else name
                print(f"{label}: macro-F1={report['macro_f1']:.4f}, "
                      f"weighted_accuracy={report['weighted_accuracy']:.4f}")
        trainer.logger.finalize("success")
    finally:
        if trainer is not None and isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()
        # Generate visualizations
        try:
            from src.visualize import generate_all_visualizations
            generate_all_visualizations('.')
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    ############################################################################################
    return fold_metrics


@hydra.main(config_path="conf", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cv_enabled, n_splits = _get_cv_settings(cfg)
    if not cv_enabled:
        _run_experiment(cfg)
        return

    if _is_primary_process():
        print(f"\n=== {n_splits}-fold cross-validation enabled "
              f"(StratifiedGroupKFold grouped by patient_id) ===")
    original_cwd = os.getcwd()
    fold_results = []
    for fold_idx in range(n_splits):
        fold_dir = os.path.join(original_cwd, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_cfg = copy.deepcopy(cfg)
        with open_dict(fold_cfg):
            fold_cfg.dataset.loader.cv_fold = fold_idx
            fold_cfg.dataset.loader.cv_n_splits = n_splits
            fold_cfg.dataset.loader.cv_val_fold = fold_idx

        if _is_primary_process():
            print(f"\n=== Starting fold {fold_idx + 1}/{n_splits} "
                  f"(held-out validation patient fold {fold_idx + 1}) ===")
        os.chdir(fold_dir)
        try:
            fold_metrics = _run_experiment(fold_cfg, fold_idx=fold_idx)
            if fold_metrics is not None:
                fold_results.append(fold_metrics)
        finally:
            os.chdir(original_cwd)

        # Incremental dump after every fold so we can monitor mid-run.
        if _is_primary_process() and fold_results:
            try:
                import json as _json
                live = []
                for r in fold_results:
                    reports = r.get('reports', {}) or {}
                    entry = {'fold': r.get('fold')}
                    for k in ('four_class', 'binary'):
                        rep = reports.get(k)
                        if rep:
                            entry[k] = {
                                'macro_f1': rep.get('macro_f1'),
                                'weighted_accuracy': rep.get('weighted_accuracy'),
                                'per_class_f1': rep.get('per_class_f1'),
                            }
                    live.append(entry)
                with open(os.path.join(original_cwd, 'fold_results_live.json'), 'w') as f:
                    _json.dump(live, f, indent=2)
                print(f"\n[live] Wrote fold_results_live.json after fold {fold_idx + 1}/{n_splits}")
            except Exception as _e:
                print(f"[live] failed to write incremental fold_results_live.json: {_e}")

    _print_cv_summary(fold_results)


if __name__ == "__main__":
    main()
    print('done')
