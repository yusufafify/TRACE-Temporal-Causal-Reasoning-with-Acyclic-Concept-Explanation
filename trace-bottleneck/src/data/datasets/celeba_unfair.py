from env import CACHE

import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import CelebA
from typing import Union, List, Optional

from src.data.datasets.celeba import CelebADataset, _CelebADataset
from src.data.utils import split_dataset



class CelebAUnfairDataset(CelebADataset):
    """
    Extension of the CelebA dataset
    to include explicit unfairness.
    """
    def __init__(self, 
                 ftune_size: float = 0.1, 
                 ftune_val_size: float = 0.1, 
                 task_label: str = "Should_be_Hired", 
                 task_cardinality: int = 2,
                 to_keep: dict = None):
        super(CelebAUnfairDataset, self).__init__(ftune_size, 
                                                  ftune_val_size, 
                                                  task_label, 
                                                  task_cardinality,
                                                  to_keep)
        self.get_dataset = _CelebAUnfairDataset


class _CelebAUnfairDataset(_CelebADataset):
    """
    Extension of the CelebA inner dataset class 
    to include explicit unfairness.
    """
    def __init__(self, 
                 root: str, 
                 split: str = 'train',
                 transform: Union[Compose, torch.nn.Module] = None,
                 download: bool = False,
                 task_label: Optional[List[str]] = 'Should_be_Hired',
                 to_keep: Optional[List[str]] = None):
        to_keep = to_keep[:-2] if to_keep is not None else None
        super(_CelebAUnfairDataset, self).__init__(root, split, transform, download, task_label, to_keep)
    
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

        # add the attribute "qualified"
        # a person is unethically 'qualified' based on their appearance
        assert 'Attractive' in concept_names and 'Heavy_Makeup' in concept_names and 'Wearing_Lipstick' in concept_names, \
        "The dataset must contain the attributes 'Attractive', 'Heavy_Makeup' and 'Wearing_Lipstick' to add the custom concepts."
        index_a = concept_names.index('Attractive')
        index_m = concept_names.index('Heavy_Makeup')
        index_l = concept_names.index('Wearing_Lipstick')
        qualified = (concepts[:, index_m] & concepts[:, index_l]) | concepts[:, index_a]
        label1 = 'Qualified'

        assert 'Pointy_Nose' in concept_names, "The dataset must contain the attribute"
        index_p = concept_names.index('Pointy_Nose')
        hired = qualified & concepts[:, index_p]
        label2 = 'Should_be_Hired'

        concepts = torch.cat((concepts, qualified.unsqueeze(1), hired.unsqueeze(1)), dim=1)
        concept_names = concept_names + [label1, label2]

        return concepts, concept_names