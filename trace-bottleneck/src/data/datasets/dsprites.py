from env import CACHE
import torch
import numpy as np
import pandas as pd
from src.data.utils import random_coloring, custom_coloring, colorize, split_dataset
from torch_geometric.utils import to_dense_adj

class DSpriteDataset():

    def __init__(self, 
                 path_to_rawdata, 
                 reduce_fraction: None,
                 val_size: float = 0.1, # proportion of the dataset to include in the validation set
                 test_size: float = 0.2, # proportion of the dataset to include in the test set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
                 coloring: dict = {'train': {'mode': 'custom', 'kwargs': {'custom_digits': [1,3,5,7]}}, 
                                   'val' : {'mode': 'custom', 'kwargs': {'custom_digits': [1,3,5,7]}},
                                   'test' : {'mode': 'random', 'kwargs': {'random_prob': 0.5}},
                                   'ftune': {'mode': 'random', 'kwargs': {'random_prob': 0.5}}}):
        
        data = np.load(str(CACHE / path_to_rawdata), allow_pickle=False)
        self.data_imgs = data['imgs']
        self.data_latent_values = data['latents_values']
        self.reduce_fraction = reduce_fraction
        self.test_size = test_size
        self.ftune_size = ftune_size
        self.val_size = val_size

        self.coloring = coloring

        self.c_info = {'names': ['color','scale', 'orientation', 'posX', 'posY'], 
                       'cardinality': [2,6,40,32,32]}
        self.y_info = {'names': ['shape'],
                       'cardinality': [3]}
        self.data = {}
        
        if self.reduce_fraction is not None:
            n_split = int(self.reduce_fraction * len(self.data_imgs))
            index_split = np.random.choice(len(self.data_imgs), n_split, replace=False)
            self.data_imgs = self.data_imgs[index_split]
            self.data_latent_values = self.data_latent_values[index_split]
                 
    def load_ground_truth_graph(self):
        node_labels = self.c_info['names'] + self.y_info['names']
        num_nodes = len(node_labels)
        self.adj = pd.DataFrame(torch.zeros((num_nodes, num_nodes)), dtype=int), 
        return self.adj
        
    def split(self):
        n_split = int(self.test_size * len(self.data_imgs))
        # get the indices of samples to be split
        index_split = np.random.choice(len(self.data_imgs), n_split, replace=False)
        # get the indices of the training (or test) samples
        index_original = np.setdiff1d(np.arange(len(self.data_imgs)), index_split)
                                    
        for indices, split_type in zip([index_original, index_split], 
                                       ['train', 'test']):
            self.data[split_type] = _DSpriteDataset(imgs = self.data_imgs[indices],
                                                    latent_variables = self.data_latent_values[indices], 
                                                    split_type = split_type,
                                                    coloring_mode = self.coloring[split_type].get('mode'),
                                                    coloring_kwargs = self.coloring[split_type].get('kwargs'))
            
        self.data['train'], self.data['val'] = split_dataset(self.data['train'], self.val_size)
        self.data['val'].split_type = 'val'
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'


class _DSpriteDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset class for loading and processing images from the dSprite dataset.

    Attributes:
    data (dict): Dictionary containing image data and latent variable information.

    images (np.ndarray): Array of images selected.

    concepts (np.ndarray): Array of latent variable attributes for the selected images ('color','scale', 'orientation', 'posX', 'posY').

    targets (np.ndarray): An array of another latent variable (shape) used as a target for classification models,
    adjusted to start from 0. The categories are as follows: 0 - square, 1 - ellipse, 2 - heart.

    random (bool): If set to True, the dataset is rearranged to ensure that each variable (concept, target) is independent of the others.
    Conversely, if set to False, the dataset is organized to introduce dependencies among the variables.

    concept_attr_names (list): List of names for concept attributes ['color','scale', 'orientation', 'posX', 'posY'].

    concept_cardinalities (list): A list of integers, representing the number of categories for each concept, e.g., [3, 6, 40, 32, 32].
    In the original dataset, the color has only one possible value (white).
    Here, we colorize the image using three possible values: "green," "red," and "blue."
    The color assignment is based on the value of the variable "random".

    task_attr_names (list): List of names for task attributes ["shape"].

    task_cardinality (list): Cardinality of task [3].
    """
    def __init__(self, 
                 imgs,
                 latent_variables,
                 split_type: str,
                 coloring_mode: str = 'random',
                 coloring_kwargs: dict = {'random_prob': 0.5}):
        super().__init__()

        self.c_dict = {'color': 0, 'scale': 1, 'orientation': 2, 'posX': 3, 'posY': 4}
        self.X = imgs
        self.c = torch.tensor(latent_variables[:, [0,2,3,4,5]])
        self.y = torch.tensor(latent_variables[:,1])-1 #to keep the indicization from 0
        self.y = self.y.unsqueeze(1)
        self.coloring_mode = coloring_mode
        self.coloring_kwargs = coloring_kwargs 
        self.split_type = split_type
        self.graph = []

        # eliminate redundancy in orientation
        unique_orientations = np.where(self.c[:,2] != 2*np.pi)[0]
        self.X = self.X[unique_orientations]
        self.c = self.c[unique_orientations]
        self.y = self.y[unique_orientations]

        new_X, new_c = zip(*[self._colorize_data(i) for i in range(len(self.X))])
        self.X = torch.stack(new_X)
        self.c[:,0] = torch.stack(new_c)

    def _colorize_data(self, index):
        img, c, y = self.X[index], self.c[index], self.y[index].item()

        # Colorize the image
        if self.coloring_mode == False:
            color = 'red'
        elif self.coloring_mode == "random":
            color = random_coloring(self.coloring_kwargs)
        elif self.coloring_mode == "custom":
            c_keys_str = [key for key in self.coloring_kwargs if key != "shape"]
            color_dict = [self.coloring_kwargs[key] for key in c_keys_str]
            c_keys = [self.c_dict[key] for key in c_keys_str]
            if "shape" in self.coloring_kwargs:
                color_dict.append(self.coloring_kwargs["shape"])
                color = custom_coloring(values= [c[tuple(c_keys)], y], coloring_kwargs = color_dict)
            else:
                color = custom_coloring(values = [c[tuple(c_keys)]], coloring_kwargs = color_dict)
        else:
            raise ValueError("invalid coloring parameter.")
        colored_image = colorize(img.squeeze(), color)  # Remove channel dimension of the grayscale image

        # Create the concept label
        c_label = 1 if color == 'red' else 0

        return (colored_image,
                torch.tensor(c_label, dtype=torch.float32))
    
    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        img = self.X[index]
        c = self.c[index]
        y = self.y[index]
        return {'x':img, 'c':c, 'y':y, 'graph':self.graph}

                



 