from env import CACHE

import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST
from typing import Union
from torch_geometric.utils import to_dense_adj

from src.data.utils import split_dataset
from src.data.utils import random_coloring, custom_coloring, colorize, split_dataset

def update_concept_names_ColorMNIST(dataset):
    """
    Update the concept names of the ColorMNIST dataset.
    """
    dataset.c_info = {'names': ['number', 'color'], 
                      'cardinality': [10, 3]}
    dataset.y_info = {'names': ['parity'],
                      'cardinality': [2]}
    return dataset

def onehot_to_concepts_ColorMNIST(dataset):
    """
    Convert 0-1 hot encoded attributes and tasks from the ColorMNIST dataset into their corresponding values.
    """
    for split, data in dataset.data.items():
        digits = torch.argmax(data.c[:, :10], axis=1)
        colors = torch.argmax(data.c[:, 10:], axis=1)
        # add an attribute to the class data
        data.c = torch.stack([digits, colors], dim=1)
        data.y = data.y[:,0].reshape(-1,1)

        dataset.data[split] = data

    return dataset


class ColorMNISTDataset():
    """
    The color MNIST dataset is a modified version of the MNIST dataset where each digit is colored either red or green.
    The concept labels are the digit and the color of the digit.
    The task is to predict whether the digit is even or odd.

    Attributes:
        transform: The transformations to apply to the images. Default is None.
        target_transform: The transformations to apply to the target labels. Default is None.
        ftune_size: The proportion of the test set to include in the finetuning set. Default is 0.1.
        val_size: The proportion of the training set to include in the validation set. Default is 0.1.
        coloring: A dictionary with coloring options: 'train' and 'test'.
    """
    def __init__(self, 
                 transform: Union[Compose, torch.nn.Module] = None,
                 target_transform: Union[Compose, torch.nn.Module] = None, 
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
                 coloring: dict = {'train': {'mode': 'custom', 'kwargs': {'custom_digits': [1,3,5,7]}}, 
                                   'test' : {'mode': 'random', 'kwargs': {'random_prob': 0.5}}}):

        self.transform = transform
        self.target_transform = target_transform
        self.ftune_size = ftune_size
        self.val_size = val_size
        self.ftune_val_size = ftune_val_size
        
        self.coloring = coloring

        self.c_info = {'names': ['0', '1', '2', '3', '4', '5', \
                                 '6', '7', '8', '9', 'red', 'green', 'blue'], 
                       'cardinality': [2,2,2,2,2,2,2,2,2,2,2,2,2]}
        self.y_info = {'names': ['even', 'odd'],
                       'cardinality': [2,2]}
        self.data = {}

    def load_ground_truth_graph(self): 
        node_labels = self.c_info['names'] + self.y_info['names'] # number, color, parity
        values = to_dense_adj(torch.tensor([[0],[2]]))[0]
        self.adj = pd.DataFrame(values, index=node_labels, columns=node_labels, dtype=int)
        return self.adj

    def split(self):
        """ 
        Split the dataset into training, validation and test sets 
        """
        self.data['train'] = _ColorMNISTDataset(root = str(CACHE / "MNIST"), 
                                                train = True, 
                                                transform = self.transform,
                                                target_transform = self.target_transform, 
                                                download = True, 
                                                coloring_mode = self.coloring['train'].get('mode'),
                                                coloring_kwargs = self.coloring['train'].get('kwargs'))
        self.data['test'] = _ColorMNISTDataset(root = str(CACHE / "MNIST"), 
                                                train = False, 
                                                transform = self.transform,
                                                target_transform = self.target_transform, 
                                                download = True, 
                                                coloring_mode = self.coloring['test'].get('mode'),
                                                coloring_kwargs = self.coloring['test'].get('kwargs'))
        self.data['train'], self.data['val'] = split_dataset(self.data['train'], self.val_size)
        self.data['val'].split_type = 'val'
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'


class _ColorMNISTDataset(MNIST):
    def __init__(self, 
                 root: str, 
                 train: bool = False, 
                 transform: Union[Compose, torch.nn.Module] = None,
                 target_transform: Union[Compose, torch.nn.Module] = None, 
                 download: bool = True, 
                 coloring_mode: str = 'random',
                 coloring_kwargs: dict = {}):
        super(_ColorMNISTDataset, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform, download=download)
        self.coloring_mode = coloring_mode
        self.coloring_kwargs = coloring_kwargs
        self.split_type = 'train' if train else 'test'
        self.graph = {}

        output1, output2, output3 = self.colorize_data()
        self.X = torch.stack(output1)
        self.c = torch.stack(output2)
        self.y = torch.stack(output3)
        delattr(self, 'data')
        delattr(self, 'targets')

    def colorize_data(self):
        colored_image = []
        c = []
        y = []

        for index in range(len(self.data)):
            image, digit = self.data[index], int(self.targets[index])
            # Colorize the image
            if self.coloring_mode == False:
                color = 'red'
            elif self.coloring_mode == "random":
                color = random_coloring(self.coloring_kwargs)
            elif self.coloring_mode == "custom":
                color = custom_coloring([digit], [self.coloring_kwargs])
            else:
                raise ValueError("invalid coloring parameter.")
            colored_image.append(colorize(image.squeeze(), color))  # Remove channel dimension of the grayscale image

            # Create the concept label
            c_label = np.zeros(13)  # 10 digits + 3 colors
            c_label[digit] = 1
            c_label[10] = 1 if color == 'red' else 0
            c_label[11] = 1 if color == 'green' else 0
            c_label[12] = 1 if color == 'blue' else 0
            c.append(torch.tensor(c_label, dtype=torch.float32))

            # Create the target label
            target_label = 1 if digit % 2 == 0 else 0
            target_label = [target_label, 1 - target_label]
            y.append(torch.tensor(target_label, dtype=torch.float32))

        return (colored_image,
                c,
                y)

    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index]
        c = self.c[index]
        y = self.y[index]
        return {'x':img, 'c':c, 'y':y, 'graph':self.graph}
                