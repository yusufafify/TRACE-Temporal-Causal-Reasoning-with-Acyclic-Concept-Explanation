from env import CACHE
import os
import torch
import json
import pandas as pd
from random import sample
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from typing import Dict, List, Union
from scipy.stats import pearsonr

# Add 'src' directory to sys.path
from src.models.clip import CXRClip

def load_pretrained_clip_model(model_name = "r50_mcc"):
    # load pretrained clip model and configurations
    ckpt = torch.load(f"{CACHE}/siim_pneumothorax/pretrained_models/{model_name}.tar", map_location="cpu")
    ckpt_config = ckpt["config"]
    ckpt_config_tokenizer = ckpt_config["tokenizer"]
    pretrained_model_name_or_path = ckpt_config_tokenizer["pretrained_model_name_or_path"]
    ckpt_config["cache_dir"] = CACHE
    cache_dir = ckpt_config["cache_dir"]

    # initialize tokenizer for clip model
    clip_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                cache_dir=cache_dir,
                local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')))
    
    if clip_tokenizer.bos_token_id is None:
            clip_tokenizer.bos_token_id = clip_tokenizer.cls_token_id

    # initialize clip model
    clip_model = CXRClip(ckpt_config["model"], 
                    ckpt_config["loss"], 
                    clip_tokenizer)


    # load pretrained weights into the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device)
    clip_model.load_state_dict(ckpt["model"], strict=False)
    clip_model.eval()

    return clip_model, clip_tokenizer, ckpt_config

def encode_image(clip_model, image: torch.Tensor, device: str = "cpu"):
        with torch.no_grad():
            img_emb = clip_model.encode_image(image.to(device))
            img_emb = clip_model.image_projection(img_emb) if clip_model.projection else img_emb
            img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        return img_emb.detach().cpu().numpy()

def encode_text(clip_model, clip_tokenizer, ckpt_config, text_token: Union[str, List[str], Dict, torch.Tensor], device: str = "cpu"):
        if isinstance(text_token, str) or isinstance(text_token, list):
            text_token = clip_tokenizer(
                text_token, padding="longest", truncation=True, return_tensors="pt", max_length=ckpt_config["base"]["text_max_length"]
            )

        with torch.no_grad():
            text_emb = clip_model.encode_text(text_token.to(device))
            text_emb = clip_model.text_projection(text_emb) if clip_model.projection else text_emb
            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        return text_emb.detach().cpu().numpy()

def transform_concepts_to_binary(c_embeddings):

    cluster_assignments = np.zeros_like(c_embeddings)
 
    # Iterate over each dimension
    for i in range(c_embeddings.shape[1]):
        #print('Concept ', i)
        # Reshape the dimension to be a 2D array (n, 1)
        concept = c_embeddings[:, i].reshape(-1, 1)
        
        # Apply k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(concept)
        
        # Get the cluster labels
        labels = kmeans.labels_
        
        # Calculate the mean of each cluster
        cluster_means = [concept[labels == j].mean() for j in range(2)]
        
        # Determine which cluster has the lower mean and which has the higher mean
        if cluster_means[0] < cluster_means[1]:
            cluster_assignments[:, i] = labels
        else:
            cluster_assignments[:, i] = 1 - labels
            
    c = torch.tensor(cluster_assignments)

    return c
 
def _generate_img_embeddings_and_assign_concepts(dataset, 
                                                 concepts, 
                                                 clip_model, 
                                                 clip_tokenizer, 
                                                 ckpt_config, 
                                                 input_encoder,
                                                 batch_size,
                                                 device):
    """
    Assign c values to dataset siim_pneumothorax based on the similarity between the images and a list concepts.
    Args:
        dataset: pneumothorax dataset object containing images.
        clip_model: generates embeddings for both images and concepts. 
                    These embeddings are then used to calculate similarities, which are subsequently used to determine the c values.
        clip_tokenizer: clip tokenizer object.
        ckpt_config: checkpoint configuration for clip model.
        input_encoder: input encoder model used to encode images for the complete pipeline
    """ 

    # create dataloader
    dataloader = DataLoader(dataset, 
                            #batch_size=cfg.dataset.batch_size, 
                            batch_size = batch_size,
                            collate_fn=getattr(dataset, "collate_fn", None), 
                            num_workers = 16,
                            pin_memory = True,
                            shuffle = False,
                            drop_last = False)

    # encode concepts with clip model
    concepts_embeddings = encode_text(clip_model, clip_tokenizer, ckpt_config, concepts, device)

    images_resnet_embeddings = []
    images_c_similarities = []
    y = []
    for batch in tqdm(dataloader):
        # encode images to calculate similarity with concepts
        img_emb = encode_image(clip_model, batch["x"], device)

        # encode images for the pipeline
        img_resnet_emb = input_encoder(batch["x"].to(device)).squeeze().detach().cpu().numpy()

        # calculate similarity between images and concepts
        img_c_similarities = metrics.pairwise.cosine_similarity(img_emb, concepts_embeddings)

        if len(batch["y"])==1:
            y.append(np.float32(batch["y"].item()))
            images_resnet_embeddings.extend(img_resnet_emb.reshape(1,-1))
            images_c_similarities.extend(img_c_similarities.reshape(1,-1))
        else:
            y.extend(batch["y"].squeeze().detach().cpu().numpy())
            images_resnet_embeddings.extend(img_resnet_emb)
            images_c_similarities.extend(img_c_similarities)

    
    # transform concepts into binary values through clustering
    c = transform_concepts_to_binary(torch.tensor(np.array(images_c_similarities)))
    dataset.c = c
    dataset.y = torch.tensor(y).unsqueeze(1)
    images_resnet_embeddings = torch.tensor(np.array(images_resnet_embeddings))

    # assign image embeddings to dataset
    dataset.X = images_resnet_embeddings
    return dataset

def concepts_analysis(c, y, concepts, images_c_similarities):

    sum_c = c.sum(dim=1)

    # boxplot of sum of concepts by Pneumothorax status
    data = np.column_stack((c.numpy(), y.numpy(), sum_c))
    column_names = concepts + ["label", "sum_c"]
    df = pd.DataFrame(data, columns = column_names)
    sns.boxplot(x='label', y='sum_c', data=df)
    plt.title('Distribution of concepts by Pneumothorax Status')
    plt.savefig('boxplot_concepts_pneumonia.png')
    plt.clf()

    
   # Plots of the distribution of sum_c for label == 0 and label == 1
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot for label == 0
    subset_0 = df[df['label'] == 0]
    axs[0].hist(subset_0['sum_c'], bins=20, alpha=0.7, color='blue', density=True)
    axs[0].set_xlabel('Sum of Concepts (sum_c)')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Distribution of sum_c for Label 0')
    axs[0].legend(['Label 0'])

    # Plot for label == 1
    subset_1 = df[df['label'] == 1]
    axs[1].hist(subset_1['sum_c'], bins=20, alpha=0.7, color='orange', density=True)
    axs[1].set_xlabel('Sum of Concepts (sum_c)')
    axs[1].set_title('Distribution of sum_c for Label 1')
    axs[1].legend(['Label 1'])

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot
    plt.savefig('distribution_sum_c_for_labels_0_and_1.png')

    # contingency tables and correlation between concepts and Pneumothorax status
    contingency_tables = {}
    for concept in concepts:
        contingency_table = pd.crosstab(df[concept], df['label'], rownames=[concept], colnames=['label'], margins=True)
        contingency_tables[concept] = contingency_table
        print(contingency_table)

    correlations = {}
    with open('contingency_and_correlations.txt', 'w') as file:
        for concept in concepts:
            contingency_table = pd.crosstab(df[concept], df['label'], rownames=[concept], colnames=['label'], margins=True)
            contingency_tables[concept] = contingency_table
            
            corr, _ = pearsonr(df[concept], df['label'])
            correlations[concept] = corr
            
            file.write(f"Contingency Table for {concept}:\n{contingency_table}\n\n")
            file.write(f"Correlation for {concept}: {corr:.4f}\n\n")

    plt.clf()



    return None

def generate_img_embeddings_and_assign_concepts(dataset_name: str,
                                                dataset: torch.utils.data.Dataset,
                                                concepts: List[str],
                                                clip_model: CXRClip,
                                                clip_tokenizer: AutoTokenizer,
                                                ckpt_config: Dict,
                                                batch_size: int = 128,
                                                device: str = 'cpu') -> None:
    
    # input encoder model to preprocess images for the pipeline
    input_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    modules = list(input_encoder.children())[:-1]
    input_encoder = nn.Sequential(*modules)
    input_encoder.to(device)
    input_encoder.eval()

    for split, data in dataset.data.items():
        data = _generate_img_embeddings_and_assign_concepts(data, 
                                                            concepts, 
                                                            clip_model, 
                                                            clip_tokenizer, 
                                                            ckpt_config, 
                                                            input_encoder,
                                                            batch_size,
                                                            device)
        dataset.data[split] = data

    # update c_info
    dataset.c_info = {'names': concepts, 
                      'cardinality': [2]* len(concepts)}
    return dataset
