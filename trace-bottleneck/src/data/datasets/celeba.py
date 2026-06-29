from env import CACHE

import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import CelebA
from typing import Union, List, Optional

from src.data.utils import split_dataset



class CelebADataset():
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
                 task_label: str = "Attractive", 
                 task_cardinality: int = 2,
                 to_keep: dict = None):
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size

        # get the inner class
        self.get_dataset = _CelebADataset

        self.c_info = {'names': None, 
                       'cardinality': None} # to be updated in the split method
        self.y_info = {'names': [task_label],
                       'cardinality': [task_cardinality]}
        self.to_keep = list(to_keep.keys()) if to_keep is not None else None
        self.data = {}

    def load_ground_truth_graph(self): 
        self.adj = None
        return self.adj

    def split(self):
        """ 
        Create training, validation and test partitions
        """
        self.data['train'] = self.get_dataset(root=str(CACHE / "CelebA"),
                                              split="train", 
                                              transform=self.transform,
                                              download=False, 
                                              task_label=self.y_info['names'],
                                              to_keep=self.to_keep)
        
        self.data["val"] = self.get_dataset(root=str(CACHE / "CelebA"),
                                            split="valid", 
                                            transform=self.transform,
                                            download=False, 
                                            task_label=self.y_info['names'],
                                            to_keep=self.to_keep)
        self.data['val'].split_type = 'val'

        self.data['test'] = self.get_dataset(root=str(CACHE / "CelebA"),
                                             split="test", 
                                             transform=self.transform,
                                             download=False, 
                                             task_label=self.y_info['names'],
                                             to_keep=self.to_keep)
        
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'

        # Update the concept info
        self.c_info['names'] = self.data['train'].c_names
        self.c_info['cardinality'] = [2] * len(self.c_info['names'])


class _CelebADataset(CelebA):
    """
    The CelebA dataset is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.
    This class extends the CelebA dataset to extract concept and task attributes based on class attributes.
    The dataset can be downloaded from the official website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Default is 'train'.
        transform: The transformations to apply to the images. Default is None.
        download: Whether to download the dataset if it does not exist. Default is False.
        task_label: The class attributes to use for the task. Default is 'Attractive'.
    """
    def __init__(self, 
                 root: str, 
                 split: str = 'train',
                 transform: Union[Compose, torch.nn.Module] = None,
                 download: bool = False,
                 task_label: Optional[List[str]] = 'Attractive',
                 to_keep: list = None):
        # Initialize and load a partition of the CelebA dataset
        super(_CelebADataset, self).__init__(root, 
                                             split=split, 
                                             target_type="attr", 
                                             transform=transform, 
                                             download=download)
        self.graph = {}
        self.split_type = split

        concepts = self.attr
        concept_names = [string for string in self.attr_names if string]  # Remove the '' at the end
    
        # select a few concepts
        concepts, concept_names = self._select_a_few_concepts(concepts, concept_names, to_keep)
        # # add custom concepts
        concepts, concept_names = self._add_custom_concepts(concepts, concept_names)

        self.X = None
        # split concepts into c and y
        self.y_names = task_label
        self.c_names = [name for name in concept_names if name not in self.y_names]
        self.y = concepts[:, [concept_names.index(concept) for concept in self.y_names]]
        self.c = concepts[:, [concept_names.index(concept) for concept in self.c_names]]
 
    def _select_a_few_concepts(self, concepts, concept_names, to_keep):
        """
        Select a few concepts to use for the task and remove the rest.
        Args:
            concepts: The concept attributes.
            concept_names: The concept attribute names.
        Returns:
            concepts: The selected concept attributes.
            to_keep: The selected concept attribute names.
        """
        concepts = concepts[:, [concept_names.index(name) for name in to_keep]]
        return concepts, to_keep
    
    def _add_custom_concepts(self, concepts, concept_names):
        """
        Add custom concepts to the dataset.
        Args:
            concepts: The concept attributes.
            concept_names: The concept attribute names.
        Returns:
            concepts: The updated concept attributes.
            concept_names: The updated concept attribute names.
        """

        return concepts, concept_names

        # in Dominici et al. (2024), they add the following custom concepts:

        # assert 'Wearing_Lipstick' in concept_names and 'Heavy_Makeup' in concept_names, \
        # "'Wearing_Lipstick' and 'Heavy_Makeup' must be in the concept names to define custom concepts"
        
        # ci1_id1 = concept_names.index('Wearing_Lipstick')
        # ci1_id2 = concept_names.index('Heavy_Makeup')
        # # ci1_id3 = ranked_label_names.index('Big_Lips')
        # ci1 = concepts[:, ci1_id1] & concepts[:, ci1_id2] #| concepts[:, ci1_id3]
        # label1 = 'Heavy_Makeup_and_Lipstick'

        # assert 'Male' in concept_names and 'Attractive' in concept_names, \
        # "'Attractive' and 'Male' must be in the concept names to define custom concepts"
        # ci2_id1 = concept_names.index('Attractive')
        # ci2_id2 = concept_names.index('Male')
        # ci2 = (concepts[:, ci2_id1] | ci1) & ~concepts[:, ci2_id2]
        # label2 = 'Fem_Model'

        # concepts = torch.cat((concepts, ci1.unsqueeze(1), ci2.unsqueeze(1)), dim=1)
        # concept_names = concept_names + [label1, label2]

        # return concepts, concept_names
    
    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.X is None:
            img, _ = super(_CelebADataset, self).__getitem__(index)
        else:
            img = self.X[index]
        y = self.y[index]
        c = self.c[index]
        return {'x':img, 'c':c, 'y':y, 'graph':self.graph}
    
    
    

    