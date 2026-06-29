from env import CACHE

import os
import pickle
from hydra.utils import instantiate

from src.data.preprocessing import preprocess_dataset
from src.plots import maybe_plot_graph

def get_dataset(cfg):
    """
    1) instantiate the dataset, 
    2) split into train, val, test
    3) preprocess all of them 
    4) save the preprocessed dataset.
    Alternatively, if the 'cfg.dataset.load_embeddings' option is provided,
    load the stored dataset.
    Args:
        cfg: DictConfig
    Returns:
        dataset: the preprocessed dataset
    """
    dataset_directory = os.path.join(str(CACHE / cfg.dataset.name))
    os.makedirs(dataset_directory, exist_ok=True)

    cache_suffix = ""
    loader_cfg = cfg.dataset.get('loader', {})
    cv_fold = loader_cfg.get('cv_fold', None) if loader_cfg is not None else None
    task_cardinality = loader_cfg.get('task_cardinality', None) if loader_cfg is not None else None
    input_mode = loader_cfg.get('input_mode', None) if loader_cfg is not None else None
    if task_cardinality is not None:
        cache_suffix += f"_{int(task_cardinality)}class"
    if input_mode is not None:
        cache_suffix += f"_{str(input_mode)}"
    if cv_fold is not None:
        cache_suffix += f"_fold{int(cv_fold)}"
    destination_path = os.path.join(dataset_directory, f"preprocessed_dataset_{cfg.seed}{cache_suffix}.pkl")

    raw_volume_mode = input_mode in ['medicalnet']
    if raw_volume_mode:
        if cfg.dataset.get('load_embeddings') != False:
            print("  MedicalNet raw-volume mode: ignoring dataset.load_embeddings so "
                  "precomputed embeddings cannot shadow end-to-end training")
        dataset = instantiate(cfg.dataset.loader)
        dataset = preprocess_dataset(cfg,
                                     dataset,
                                     device=cfg.device,
                                     backbone=cfg.dataset.backbone)
        for split_name, split_data in dataset.data.items():
            if getattr(split_data, 'X', None) is not None:
                raise RuntimeError(
                    f"MedicalNet raw-volume mode expected {split_name}.X to be None, "
                    "but found precomputed features. Refusing to train on cached embeddings."
                )
        print("  MedicalNet raw-volume mode: skipping preprocessed_dataset pickle "
              "(raw volumes are rebuilt each run; cached embeddings are not used)")
    elif cfg.dataset.get('load_embeddings') == False:
        dataset = instantiate(cfg.dataset.loader)
        dataset = preprocess_dataset(cfg, 
                                     dataset, 
                                     device=cfg.device,
                                     backbone=cfg.dataset.backbone)
        with open(destination_path, 'wb') as f: 
            pickle.dump(dataset, f)
    else:
        with open(destination_path, 'rb') as f: 
            dataset = pickle.load(f)
    
    true_graph = dataset.load_ground_truth_graph()
    maybe_plot_graph(true_graph, 'true_graph')
    return dataset, true_graph, dataset_directory
