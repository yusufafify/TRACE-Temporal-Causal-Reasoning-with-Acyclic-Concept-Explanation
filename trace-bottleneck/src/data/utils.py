import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import numpy as np
import torch
import random
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

def static_graph_collate(batch):
    collated = {
        "x": torch.stack([item["x"] for item in batch]),
        "c": torch.stack([item["c"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "graph": batch[0]["graph"],  # Add the graph once
    }

    for optional_key in ("x_baseline", "x_seg_curr", "x_seg_base", "clinical_features"):
        if optional_key in batch[0]:
            collated[optional_key] = torch.stack([item[optional_key] for item in batch])

    return collated

def reduce_dataset(_dataset, index_to_keep):
    dataset = deepcopy(_dataset)
    if dataset.X is not None:
        dataset.X = dataset.X[index_to_keep]
    if dataset.c is not None:
        dataset.c = dataset.c[index_to_keep]
    if dataset.y is not None:
        dataset.y = dataset.y[index_to_keep]
    if hasattr(dataset, 'df'):
        dataset.df = dataset.df.iloc[index_to_keep]
        dataset.df = dataset.df.reset_index(drop = True)
    return dataset

def split_dataset(_dataset, split_size):
    len_dataset = len(_dataset)
    # get the number of samples to be split
    n_split = int(split_size * len_dataset)
    # get the indices of samples to be split
    index_split = np.random.choice(len_dataset, n_split, replace=False)
    # get the indices of the training (or test) samples
    index_original = np.setdiff1d(np.arange(len_dataset), index_split)

    dataset_split = reduce_dataset(_dataset, index_split)
    dataset_original = reduce_dataset(_dataset, index_original)

    return dataset_original, dataset_split

def random_coloring(coloring_kwargs):
    # Sample a number from 0 to two from a uniform distribution
    # If the number is 0, return red, else return green
    sample = random.randint(0, 2)
    if sample == 0:
        return "red"
    elif sample == 1:
        return "green"
    else:
        return "blue"

def custom_coloring(values, coloring_kwargs):
    colors_results = []
    for index in range(len(values)):
        value = values[index]
        col_dict = coloring_kwargs[index]
        if col_dict["mode"]=="single values":
                if value in col_dict["values"]:
                    colors_results.append('green')
                else:
                    s = random.randint(0, 1)
                    if s == 0:
                        colors_results.append('red')
                    else:
                        colors_results.append('blue')
        elif col_dict["mode"]=="interval":
            if col_dict["values"][0]<= value and value <= col_dict["values"][1]:
                colors_results.append('red')
            else:
                colors_results.append('green')
        else:
            raise ValueError(f"invalid coloring_kwargs.")
    
    if colors_results.count('red') == len(colors_results):
        return 'red'
    elif colors_results.count('green') == len(colors_results):
        return 'green'
    elif colors_results.count('blue') == len(colors_results):
        return 'blue'
        
def colorize(image: torch.Tensor, color: str) -> torch.Tensor:
    if image.dtype == "uint8":
        image = torch.tensor(image, dtype=torch.float32)
        colored_image = torch.zeros(3, 64, 64)
    else:
        colored_image = torch.zeros(3, 28, 28)  # Create an image with 3 channels (RGB)
    if color == 'red':
        colored_image[0] = image  # Red channel
    elif color == 'green':
        colored_image[1] = image  # Green channel
    return colored_image


def make_it_incomplete(dataset):
    """
    Make the dataset incomplete by removing some edges.
    Args:
        dataset: Dataset object
    Returns:
        dataset: Incomplete dataset object
    """
    pass
