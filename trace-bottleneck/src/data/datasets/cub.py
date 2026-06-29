from env import CACHE

import torch
from torchvision import transforms
import os
import re
import numpy as np
import requests
import tarfile
from torch.utils.data import Dataset, random_split
from PIL import Image
from src.data.utils import split_dataset
import pandas as pd
import random

class _CUBDataset(Dataset):
    def __init__(self, root_dir, split='train', mean=None, std=None, selected_concepts=None, causal_struct=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (str): If it is equal to 'train' it will load the training set,
                if it is equal to 'val' it will load the validation set,
                if it is equal to 'test' it will load the test set. 
            mean (tuple): Mean for normalization.
            std (tuple): Standard deviation for normalization.
        """

        self.X = None
        self.causal_struct = causal_struct
        
        # Collect the list of selected attributes ids.

        # Read the attribute names
        self.attribute_names = {}
        dataset_dir = root_dir
        with open(os.path.join(dataset_dir, "./attributes/attributes.txt"), "r") as file:
            for line in file:
                matches = line.strip().split(" ")
                attribute_id, attribute_name = matches[0], matches[1]
                attribute_id = int(attribute_id)
                if attribute_name in selected_concepts:
                    self.attribute_names[attribute_name] = attribute_id

            SELECTED_CONCEPTS = self.attribute_names.values()

        # Read the attribute names
        self.attribute_names = {}
        with open(os.path.join(dataset_dir, "./attributes/attributes.txt"), "r") as file:
            for line in file:
                matches = line.strip().split(" ")
                attribute_id, attribute_name = matches[0], matches[1]
                attribute_id = int(attribute_id)
                if attribute_id in SELECTED_CONCEPTS:
                    self.attribute_names[attribute_id] = attribute_name

        # Create a mapping from attribute names to their indices in the array
        self.concept_array_order = {}
        for i, concept in enumerate(self.attribute_names.values()):
            self.concept_array_order[concept] = i

        self.root_dir = root_dir
        self.split = split

        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor()
            ]) 

        # Parse the dataset files
        dataset_dir = root_dir
        self.image_paths_train = []
        self.labels_train = []
        self.image_ids_train = []
        self.image_paths_val = []
        self.labels_val = []
        self.image_ids_val = []
        self.image_paths_test = []
        self.labels_test = []
        self.image_ids_test = []

        with open(os.path.join(dataset_dir, "images.txt"), "r") as img_file:
            image_lines = img_file.readlines()
        with open(os.path.join(dataset_dir, "image_class_labels.txt"), "r") as label_file:
            label_lines = label_file.readlines()
        with open(os.path.join(dataset_dir, "train_test_split.txt"), "r") as split_file:
            split_lines = split_file.readlines()

        # Initialize a dictionary to hold the boolean arrays for each image
        self.image_attributes = {}
        with open(os.path.join(dataset_dir, "./attributes/image_attribute_labels.txt"), "r") as file:
            for line in file:
                matches = re.findall(r"\d+\.\d+|\d+", line)
                image_id, attribute_id, is_present = matches[0], matches[1], matches[2] #line.strip().split(" ")
                image_id = int(image_id)
                attribute_id = int(attribute_id)
                is_present = int(is_present)
                if image_id not in self.image_attributes:
                    cnt = 0
                    self.image_attributes[image_id] = np.zeros(len(SELECTED_CONCEPTS), dtype=float)
                if attribute_id in SELECTED_CONCEPTS:
                    self.image_attributes[image_id][cnt] = float(is_present)
                    cnt += 1

        # Extract image paths and labels
        for img_line, label_line, split_line in zip(image_lines, label_lines, split_lines):
            img_id, img_path = img_line.strip().split(" ")
            label_id, label = label_line.strip().split(" ")
            img2_id, split_id = split_line.strip().split(" ")
            assert img_id == label_id == img2_id # Ensure consistent IDs
            if split_id == '1':
                self.image_ids_train.append(int(img_id))
                self.image_paths_train.append(os.path.join(dataset_dir, "images", img_path))
                self.labels_train.append(int(label) - 1)  # Convert to zero-based index
            else:
                self.image_ids_test.append(int(img_id))
                self.image_paths_test.append(os.path.join(dataset_dir, "images", img_path))
                self.labels_test.append(int(label) - 1)  # Convert to zero-based index

        # Ceate a list o shuffled indexes
        idxs = list(range(len(self.image_ids_train)))
        random.shuffle(idxs)
        self.image_ids_train = [self.image_ids_train[i] for i in idxs]
        self.image_paths_train = [self.image_paths_train[i] for i in idxs]
        self.labels_train = [self.labels_train[i] for i in idxs]

        # Randomly split the training in Train/Val
        # Select 10% of the training set for validation
        my_list = list(range(1, len(self.image_ids_train)))  # Example list with 100 elements
        sample_size = max(1, len(my_list) * 10 // 100)  # Ensure at least 1 element is selected
        val_idxs = random.sample(my_list, sample_size)
        self.image_paths_val = [self.image_paths_train[i] for i in val_idxs]
        self.labels_val = [self.labels_train[i] for i in val_idxs]
        self.image_ids_val = [self.image_ids_train[i] for i in val_idxs]

        # Remove the validation images from the training set
        self.image_paths_train = [self.image_paths_train[i] for i in range(len(self.image_ids_train)) if i not in val_idxs]
        self.labels_train = [self.labels_train[i] for i in range(len(self.image_ids_train)) if i not in val_idxs]
        self.image_ids_train = [self.image_ids_train[i] for i in range(len(self.image_ids_train)) if i not in val_idxs]

        # Initialize the graph
        self.graph = {}

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths_train)
        elif self.split == 'val':
            return len(self.image_paths_val)
        else:
            return len(self.image_paths_test)

    def _augment_concepts(self, concepts):
        camouflage = concepts[self.concept_array_order['has_tail_pattern::spotted']] \
            | concepts[self.concept_array_order['has_tail_pattern::striped']] \
            | concepts[self.concept_array_order['has_tail_pattern::multi-colored']] \
            | concepts[self.concept_array_order['has_back_pattern::spotted']] \
            | concepts[self.concept_array_order['has_back_pattern::striped']] \
            | concepts[self.concept_array_order['has_back_pattern::multi-colored']] 
        flight_adaptation = concepts[self.concept_array_order['has_tail_shape::rounded_tail']] \
            | concepts[self.concept_array_order['has_wing_shape::rounded-wings']] \
            | concepts[self.concept_array_order['has_size::medium_(9_-_16_in)']] 
        hunting_ability = concepts[self.concept_array_order['has_bill_shape::curved_(up_or_down)']] \
            | concepts[self.concept_array_order['has_bill_shape::needle']] \
            | concepts[self.concept_array_order['has_bill_shape::spatulate']] \
            | concepts[self.concept_array_order['has_bill_shape::all-purpose']] \
            | concepts[self.concept_array_order['has_bill_length::longer_than_head']] \
            | concepts[self.concept_array_order['has_bill_length::shorter_than_head']]

        concepts = torch.cat((concepts, torch.tensor([flight_adaptation, hunting_ability, camouflage])), dim=0)
        return concepts

    def _compute_task_label(self, concepts):
        # The task label is the result of the and operation between the following concepts:
        # 1. Camouflage
        # 2. Flight Adaptation
        # 3. Hunting Ability
        count = concepts[-1] + concepts[-2] + concepts[-3]
        if count==0:
            task_label = torch.tensor(0)
        if count==1:
            task_label = torch.tensor(1)
        if count>=2:
            task_label = torch.tensor(2)
        return task_label
    
    def register_graph(self, graph):
        self.graph = graph

    def update_lists(self):
        # Initialize the concepts and labels
        self.c = []
        self.y = []

        if self.split == 'train':
            for i, id in enumerate(self.image_ids_train):
                out = self.__getitem__(i)
                self.c.append(out['c'])
                self.y.append(out['y'])
        elif self.split == 'val':
            for i, id in enumerate(self.image_ids_val):
                out = self.__getitem__(i)
                self.c.append(out['c'])
                self.y.append(out['y'])
        else:
            for i, id in enumerate(self.image_ids_test):
                out = self.__getitem__(i)
                self.c.append(out['c'])
                self.y.append(out['y'])
        self.c = torch.cat([c.unsqueeze(0) for c in self.c], dim=0)
        self.y = torch.cat([y.unsqueeze(0) for y in self.y], dim=0).unsqueeze(-1)
        self.c = self.c.type(torch.int)
        self.y = self.y.type(torch.int)

    def __getitem__(self, idx):

        if self.split=='train':
            img_path = self.image_paths_train[idx]
            label = self.labels_train[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_train[idx]])
        elif self.split == 'val':
            img_path = self.image_paths_val[idx]
            label = self.labels_val[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_val[idx]])
        else:
            img_path = self.image_paths_test[idx]
            label = self.labels_test[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_test[idx]])
        
        if self.X is None:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image) 
        else:
            image = self.X[idx]

        # Make the concept ints
        concepts = concepts.type(torch.long)

        if self.causal_struct:
            # increase attribute tensor using the augment_concepts method
            concepts = self._augment_concepts(concepts)
            # compute the task label
            label = self._compute_task_label(concepts)
        else:
            label = torch.tensor(label)

        return {'x':image, 'c':concepts, 'y':label, 'graph':self.graph}

class CUBDataset():
    """
    Args:
        ftune_size: The fraction of the test set to use for fine-tuning. Default is 0.1.
        ftune_val_size: The fraction of the fine-tuning set to use for validation. Default is 0.1.
        task_label: The class attribute to use for the task. Default is "Should_be_Hired".
        task_cardinality: The cardinality of the task attribute. Default is 2.
    """
    def __init__(self, 
                 ftune_size: float = 0.1, 
                 ftune_val_size: float = 0.1, 
                 task_label: str = "", 
                 task_cardinality: int = 200,
                 to_keep: dict = None,
                 causal_struct = False):
        
        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size
        self.causal_struct = causal_struct

        # get the inner class
        self.get_dataset = _CUBDataset

        self.c_info = {'names': None, 
                       'cardinality': None} # to be updated in the split method
        
        self.y_info = {'names': [task_label],
                       'cardinality': [task_cardinality]}

        self.data = {}

        self.to_keep = to_keep

    def load_ground_truth_graph(self): 

        # Create the dictionary for the graph adj matrix 
        node_labels = self.c_info['names'] + self.y_info['names']

        # Number of nodes
        n = len(node_labels)

        # Initialize the adjacency matrix with zeros
        adj_matrix = np.zeros((n, n), dtype=int)

        if self.causal_struct:
            # Camouflage nodes indices (0-5)
            camouflage_indices = [0, 1, 2, 3, 4, 5]

            # Flight adaptation nodes indices (6-8)
            flight_adaptation_indices = [6, 7, 8]

            # Hunting ability nodes indices (9-15)
            hunting_ability_indices = [9, 10, 11, 12, 13, 14, 15]

            # Fill the adjacency matrix (relationships can be established based on your understanding)

            # Example relationships:
            # All camouflage nodes are related to each other (1s in their respective rows/columns)
            for i in camouflage_indices:
                adj_matrix[i, 16] = 1

            # All flight adaptation nodes are related to each other
            for i in flight_adaptation_indices:
                adj_matrix[i, 17] = 1

            # All hunting ability nodes are related to each other
            for i in hunting_ability_indices:
                adj_matrix[i, 18] = 1
        
            adj_matrix[16, 19] = 1  # Camouflage -> Sruvival
            adj_matrix[17, 19] = 1  # Flight Adaptation -> Survival
            adj_matrix[18, 19] = 1  # Hunting Ability -> Survival
            
        else:
            # It is a birtaprtite graph where the all the concepts are related to the task
            for i in range(n-1):
                adj_matrix[i, n-1] = 1

        # create a pandas dataframe from the adjacency matrix
        adj_pandas = pd.DataFrame(adj_matrix.astype(int), index=node_labels, columns=node_labels)
        self.adj = adj_pandas

        return self.adj
    
    def split(self):
        """ 
        Create training, validation and test partitions
        """

        attribute_names = []
        attribute_card = []
        for elem in list(self.to_keep.keys()):
            attribute_names.append(elem)
            attribute_card.append(2)
        self.c_info['names'] = attribute_names
        self.c_info['cardinality'] = attribute_card

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)

        train_dataset = _CUBDataset(root_dir=str(CACHE / "cub"), split='train', mean=mean, 
                                    std=std, selected_concepts=attribute_names,
                                    causal_struct=self.causal_struct)

        val_dataset = _CUBDataset(root_dir=str(CACHE / "cub"), split='val', mean=mean, 
                                  std=std, selected_concepts=attribute_names,
                                  causal_struct=self.causal_struct)

        self.data['train'] = train_dataset
        self.data["val"] = val_dataset
        self.data['val'].split_type = 'val'

        self.data['test'] = _CUBDataset(root_dir=str(CACHE / "cub"), split='test', mean=mean, 
                                        std=std, selected_concepts=attribute_names,
                                        causal_struct=self.causal_struct)

        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'

        if self.causal_struct:
            # Add the attributes: Flight Adaptation & Camouflage & Hunting Ability
            self.c_info['names'].append('Camouflage')
            self.c_info['cardinality'].append(2)            
            self.c_info['names'].append('Flight Adaptation')
            self.c_info['cardinality'].append(2)
            self.c_info['names'].append('Hunting Ability')
            self.c_info['cardinality'].append(2)
