from env import CACHE
from copy import deepcopy
import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
import numpy as np
# progress bar
from tqdm import tqdm

from src.models.layers.pretrained import InputImgEncoder
from src.data.utils import reduce_dataset
from src.data.datasets.colormnist import update_concept_names_ColorMNIST, onehot_to_concepts_ColorMNIST
from src.data.autoencoder import AutoencoderTrainer, scale_embeddings
from src.data.labelfree_preprocessing import load_pretrained_clip_model, generate_img_embeddings_and_assign_concepts
from src.completion.concepts_retrieval import concepts_generation, filtering_concepts_from_llm
from src.data.datasets.synthetic import get_synthetic_datasets, SyntheticDatasetContainer


def _adapt_resnet_conv1(model, in_channels):
    """Replace the first conv layer of a ResNet to accept `in_channels` instead of 3.
    Initializes new weights by repeating the pretrained 3-channel weights."""
    old_conv = model.conv1
    new_conv = torch.nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        # Repeat the 3-channel weights to fill in_channels, then average
        old_weight = old_conv.weight  # [64, 3, 7, 7]
        repeats = (in_channels // 3) + 1
        new_weight = old_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
        new_weight = new_weight * (3.0 / in_channels)  # scale to preserve magnitude
        new_conv.weight.copy_(new_weight)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv
    return model


def generate_img_embeddings_multichannel(dataset, batch_size=16, device='cpu',
                                         backbone='resnet18', in_channels=8):
    """Generate image embeddings for multi-channel (e.g. 8-ch MRI) datasets."""
    if backbone == 'resnet18':
        input_encoder = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet50':
        input_encoder = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    input_encoder = _adapt_resnet_conv1(input_encoder, in_channels)
    model = InputImgEncoder(input_encoder).to(device)
    model.eval()

    for split, data in dataset.data.items():
        data = _generate_img_embeddings(data, model, batch_size, device)
        dataset.data[split] = data
    return dataset


def generate_img_embeddings(dataset: torch.utils.data.Dataset,
                           batch_size: int = 32,
                           device: str = 'cpu',
                           backbone: str = 'resnet18') -> None:
    
    if backbone == 'resnet18':
        input_encoder = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet50':
        input_encoder = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = InputImgEncoder(input_encoder).to(device)
    model.eval()

    for split, data in dataset.data.items():
        data = _generate_img_embeddings(data, model, batch_size, device)
        dataset.data[split] = data
    return dataset

def _generate_img_embeddings(dataset, model, batch_size, device) -> None:
    """
    Preprocess an image dataset using a given input encoder.
    Args:
        dataset: dataset object.
        input_encoder: input encoder model.
        batch_size: batch size.
        device: device to run the model on.
    Returns:
        None
    """

    # Load dataset
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader)):
            images = batch['x'].to(device)
            # TODO: check this handles colors correctly
            emb = model(images)
            embeddings.append(emb)
                
    # Concatenate and save embeddings
    embeddings = torch.cat(embeddings, dim=0).cpu()
    dataset.X = embeddings
    return dataset

def maybe_reduce(reduce_fraction, dataset):
    # random sample a fraction of the dataset
    if reduce_fraction is not None:
        for split, data in dataset.data.items():
            # get the number of samples to be split
            n_split = int(reduce_fraction * len(data))
            # get the indices of samples to be split
            index_split = np.random.choice(len(data), n_split, replace=False)
            data = reduce_dataset(data, index_split)
            dataset.data[split] = data
    return dataset

def preprocess_dataset(cfg, _dataset, device, backbone) -> dict:
    """
    Preprocess the dataset.
    Args:
        cfg: Dictionary with the configuration.
        dataset: Dictionary with the dataset splits.
    Returns:
        processed_dataset: Dictionary with the preprocessed dataset splits.
    """
    dataset = deepcopy(_dataset)

    print('preprocessing data...')

    # colormnist
    dataset_name = cfg.dataset.get('name').replace('_ood', '')
    if dataset_name == 'colormnist':
        dataset.split()
        if cfg.dataset.get('onehot_to_concepts') == True: 
            dataset = update_concept_names_ColorMNIST(dataset)
        dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        dataset = generate_img_embeddings(dataset, 
                                          batch_size=256, 
                                          device=device,
                                          backbone=backbone)
        if cfg.dataset.get('onehot_to_concepts') == True:
            dataset = onehot_to_concepts_ColorMNIST(dataset)
    
    elif dataset_name in ['celeba', 'celeba_reduced', 'celeba_unfair', 'cub_causal_struct', 'cub']:
        dataset.split()
        if dataset_name in ['cub_causal_struct', 'cub']:
            dataset.data['train'].update_lists()
            dataset.data['val'].update_lists()
            dataset.data['test'].update_lists()

        dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        dataset = generate_img_embeddings(dataset, 
                                          batch_size=256,
                                          device=device,
                                          backbone=backbone)         

    elif dataset_name in ['asia', 'asia_reduced', 'alarm', 'alarm_reduced', \
                          'sachs', 'sachs_reduced', 'hailfinder', 'insurance']:
        dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        
        all_var = dataset.c_info_complete['names'] + dataset.y_info['names'] # variables have been reordered
                                                                    # when the dataset was created
        # for most datasets, encode only the concepts variables, exclude the task
        #selected_var = dataset.c_info['names']
        selected_var = dataset.c_info_complete['names']
        if dataset_name=='asia':
            pass
            # selected_var = ['asia', 'smoke']
        elif dataset_name=='alarm':
            pass
            # selected_var = ['MINVOLSET', 'DISCONNECT', 'PULMEMBOLUS', \
            #                 'INTUBATION', 'KINKEDTUBE', 'ANAPHYLAXIS', \
            #                 'FIO2', 'INSUFFANESTH', 'LVFAILURE', 'HYPOVOLEMIA', \
            #                 'ERRLOWOUTPUT', 'ERRCAUTER']
        elif dataset_name=='sachs_ood':
            pass
        selected_var_index = [all_var.index(var) for var in selected_var]

        autoencoder_trainer = AutoencoderTrainer(autoencoder_cfg=cfg.dataset.autoencoder,
                                                 input_shape=len(selected_var), 
                                                 device=device)
        dataset.split()
        dataset = autoencoder_trainer.train(dataset=dataset, 
                                            selected_var_index=selected_var_index)
        dataset = scale_embeddings(dataset)
    elif cfg.dataset.get('name') == 'siim_pneumothorax':
        clip_model, clip_tokenizer, ckpt_config = load_pretrained_clip_model("r50_mcc")
        dataset.split(ckpt_config)
        	   
        # if we already generated the concepts we simply read them form the respective json file,
        # otherwise we generate them using the llm.
        concepts_path = os.path.join(CACHE, "siim_pneumothorax")
        if not os.path.exists(os.path.join(concepts_path, 'generated_concepts.json')):
            # generate concepts with llm
            concepts = concepts_generation()
            concepts = filtering_concepts_from_llm(concepts,
                                                    class_labels = dataset.y_info['names'],
                                                    training_data = dataset.data["train"],
                                                    clip_model = clip_model,
                                                    clip_tokenizer = clip_tokenizer,
                                                    ckpt_config = ckpt_config,
                                                    device = device) 
            with open(os.path.join(concepts_path, 'generated_concepts.json'), 'w') as f:
                json.dump({'concepts': concepts}, f)
        else:
            with open(os.path.join(concepts_path, 'generated_concepts.json')) as f:
                concepts = json.load(f)['concepts']          
        dataset = generate_img_embeddings_and_assign_concepts(dataset_name = cfg.dataset.get('name'),
                                                                dataset = dataset,
                                                                concepts = concepts,
                                                                clip_model = clip_model,
                                                                clip_tokenizer = clip_tokenizer,
                                                                ckpt_config = ckpt_config,
                                                                batch_size=256, 
                                                                device=device)
        # avoid empty spaces in the concepts names
        dataset.c_info['names'] = [concept.replace(' ', '_') for concept in dataset.c_info['names']]


        #dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        #dataset = generate_img_embeddings(dataset, batch_size=cfg.dataset.get('batch_size'), device=device)

    elif dataset_name == 'lumiere':
        dataset.split()
        dataset.data['train'].update_lists()
        dataset.data['val'].update_lists()
        dataset.data['test'].update_lists()
        dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)
        # Determine number of input channels (4 modalities * 5 slices = 20 base channels)
        in_channels = 20
        if cfg.dataset.loader.get('use_8_channels', True):
            in_channels = 40
        if cfg.dataset.loader.get('use_delta_only', False):
            in_channels = 20
        dataset = generate_img_embeddings_multichannel(
            dataset, batch_size=16, device=device,
            backbone=backbone, in_channels=in_channels
        )

    elif cfg.dataset.get('name') == 'synthetic':
        c_info = {}
        c_info['names'] = [f'concept_{i}' for i in range(cfg.dataset.loader.get('num_predicates'))]
        c_info['cardinality'] = [2 for _ in range(cfg.dataset.loader.get('num_predicates'))]
        y_info = {}
        y_info['names'] = ['y']
        y_info['cardinality'] = [2]
        dataset = SyntheticDatasetContainer(data=dataset,
                                   c_info=c_info,
                                   y_info=y_info,)
    else:
        raise ValueError(f"Preprocessing is missing for dataset: {cfg.dataset.get('name')}")
    
    print('done')

    print(f"Concepts: {dataset.c_info['names']}")
    print(f"Task: {dataset.y_info['names']}")
    return dataset
