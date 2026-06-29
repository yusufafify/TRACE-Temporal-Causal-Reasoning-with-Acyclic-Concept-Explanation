from env import CACHE
import ast
import os
import tqdm
from typing import Dict, List

import pandas as pd
import torch
import numpy as np
from PIL import Image
import albumentations
import albumentations.pytorch.transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Dict, Union
from src.data.utils import split_dataset

# to download the images
# 7z x archive.7z

def preprocess_csv_file(data_path):
    '''
    This function is designed to preprocess the siim_pneumothorax CSV file, 
    which should be downloaded from Kaggle at 
    https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data. 
    It then filters out images that are not present in the image directories downloaded from
    https://www.kaggle.com/datasets/abhishek/siim-png-images.

    Args:
        original_csv_path (str): Path to the original CSV file.
        images_paths (List[str]): List of directories containing the images.
        save_path (str): Path to save the processed CSV file.
    Returns:
        str: Path to the saved CSV file.
    '''
    original_csv_path = os.path.join(data_path,"stage_2_train.csv" )
    images_paths = [os.path.join(data_path, "train_png"), os.path.join(data_path, "test_png")]
    save_path = os.path.join(data_path, "siim_train.csv")

    if not os.path.exists(original_csv_path):
        raise FileNotFoundError(f"Error: The file {original_csv_path} was not found. Please download it from: "
                             "https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data")
            
    if not os.path.exists(images_paths[0]):
        raise FileNotFoundError(f"Error: Image directories {images_paths[0]} and {images_paths[1]} were not found. "
                                "Please download them from: https://www.kaggle.com/datasets/abhishek/siim-png-images")




    df = pd.read_csv(original_csv_path)
    # rename ImageId as image
    df = df.rename(columns={"ImageId": "image", "Unnamed: 0": "index"})
    # create label column
    df["label"] = df["EncodedPixels"].apply(lambda x: 1 if x=="-1" else 0)
    # eliminate column EncodePixels
    df = df.drop(columns=["EncodedPixels"])
    # create class column
    # TODO: check if this column is really necessary
    df["class"] = df["label"].apply(lambda x: "Pneumothorax" if x==1 else "No Pneumothorax") 

    # filter images not present in the dataset
    valid_indices = []
    not_valid_indices = []
    images_dir = []
    for idx in range(len(df)):
        image_id = df['image'][idx]
        image_found = False
        for directory in images_paths:
            image_path = os.path.join(directory, f"{image_id}.png")
            if os.path.exists(image_path):
                valid_indices.append(idx)
                images_dir.append(image_path)
                image_found = True
                break  # Stop searching once the image is found

        if not image_found:
            not_valid_indices.append(idx)
            continue
   
    df_filtered = df.iloc[valid_indices].copy()
    df_filtered["image"] = images_dir

    #return the final .csv
    df_filtered.to_csv(save_path, index=False)
    print(f"Processed CSV file available at: {save_path}")
    return None

def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}

    config = []
    if transform_config:
        if split in transform_config:
            config = transform_config[split]
    image_transforms = []

    for name in config:
        if hasattr(transforms, name):
            tr_ = getattr(transforms, name)
        else:
            tr_ = getattr(albumentations, name)
        tr = tr_(**config[name])
        image_transforms.append(tr)

    return image_transforms

def transform_image(image_transforms, image: Union[Image.Image, np.ndarray], normalize="huggingface"):
    for tr in image_transforms:
        if isinstance(tr, albumentations.BasicTransform):
            image = np.array(image) if not isinstance(image, np.ndarray) else image
            image = tr(image=image)["image"]
        else:
            image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
            image = tr(image)

    if normalize == "huggingface":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)

    elif normalize == "imagenet":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    else:
        raise KeyError(f"Not supported Normalize: {normalize}")

    return image

class ImageClassificationDataset(Dataset):
    def __init__(self,
                 test_size: float = 0.1, 
                 ftune_size: float = 0.1,
                 ftune_val_size: float = 0.): # proportion of the finetuning set to include in the finetuning validation set
        
        self.ftune_size = ftune_size
        self.test_size = test_size
        self.ftune_val_size = ftune_val_size

        self.c_info = {'names': None, 
                       'cardinality': None} # to be updated in the split method
        self.y_info = {'names': ["Pneumothorax"],
                       'cardinality': [2]}

        self.data = {}

    def load_ground_truth_graph(self): 
        self.adj = None
        return self.adj

    def split(self, ckpt_config):
        """ 
        Create training, validation and test partitions
        """
        
        data_file = os.path.join(str(CACHE / "siim_pneumothorax/siim_train.csv"))

        if not os.path.exists(data_file):
            preprocess_csv_file(os.path.join(str(CACHE / "siim_pneumothorax")))
        
        self.data['train'] = _ImageClassificationDataset(data_file = data_file,
                                                split="train", 
                                                transform_config = ckpt_config["transform"])
            
        self.data['train'], data_val_test = split_dataset(self.data['train'], 0.3)
        self.data["val"], data_test = split_dataset(data_val_test, self.test_size)
        self.data['val'].split_type = 'val'
        if self.ftune_size !=0:
            self.data['test'], self.data['ftune'] = split_dataset(data_test, self.ftune_size)
            self.data["test"].split_type = 'test'
            self.data['ftune'].split_type = 'ftune'
            self.data['ftune'], self.data['ftune_val'] = split_dataset(self.data['ftune'], self.ftune_val_size)
            self.data['ftune_val'].split_type = 'ftune_val'
        else:
            self.data['test'] = data_test
            self.data["test"].split_type = 'test'

class _ImageClassificationDataset(Dataset):
    def __init__(
            self,
            data_file: str,
            split: str,
            transform_config: Dict = None,
            normalize: str = "imagenet"
    ):
        super().__init__()

        self.split = split
        self.normalize = normalize
        self.image_transforms = load_transform(split="test", transform_config=transform_config)

        # load the data
        self.df = pd.read_csv(data_file)
        self.X = None
        self.c = None
        self.y = None
        self.graph = {}
        

    def __len__(self):
        return len(self.df)
    
    def register_graph(self, graph):
        self.graph = graph

    def __getitem__(self, index):
        if self.X is None:
            image_path = f'{CACHE}/siim_pneumothorax/{self.df["image"][index]}'
            image = Image.open(image_path).convert("RGB")
            image = transform_image(self.image_transforms, image, normalize=self.normalize)
            if "label" in self.df:
                label = self.df["label"][index]
                if type(label) is str:
                    label = ast.literal_eval(label)
                else:
                    label = [label]
                label = torch.Tensor(label)
            else:
                raise AttributeError("Cannot read the column for label")  
            c = torch.zeros_like(label) # default value, it will be filled in the preprocessing phase
        else:
            image = self.X[index]
            c = self.c[index]
            label = self.y[index]

        return {"x": image, "c": c, "y": label, "graph": self.graph}

    def collate_fn(self, instances: List):
        images = torch.stack([ins["x"] for ins in instances], dim=0)
        c = torch.stack([ins["c"] for ins in instances], dim=0)
        labels = torch.stack([ins["y"] for ins in instances], dim=0)
        graph = instances[0]["graph"]
        return {"x": images, "c": c, "y": labels, "graph" : graph}