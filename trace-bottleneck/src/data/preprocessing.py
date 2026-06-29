from env import CACHE
from copy import deepcopy
import os
import json
import torch
import torch.nn as nn
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


class TemporalDifferenceEncoder(nn.Module):
    """Encodes current/baseline MRI scans via shared ResNet (up to layer3),
    computes feature-space spatial diffs, then pools. Output: [B, 512]."""

    def __init__(self, backbone):
        super().__init__()
        children = list(backbone.children())
        self.spatial_features = nn.Sequential(*children[:7])  # through layer3
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_current, x_baseline):
        feat_curr = self.spatial_features(x_current)
        feat_base = self.spatial_features(x_baseline)
        feat_diff = feat_curr - feat_base

        curr_embed = self.pool(feat_curr).flatten(1)
        diff_embed = self.pool(feat_diff).flatten(1)
        return torch.cat([curr_embed, diff_embed], dim=1)


def _adapt_conv1(backbone, in_channels):
    """Tile pretrained conv1 weights to handle non-3-channel input."""
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        old_w = old_conv.weight  # [64, 3, 7, 7]
        reps = (in_channels + 2) // 3
        new_w = old_w.repeat(1, reps, 1, 1)[:, :in_channels, :, :]
        new_w = new_w * (3.0 / in_channels)
        new_conv.weight.copy_(new_w)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    backbone.conv1 = new_conv
    return backbone


def generate_temporal_diff_embeddings(dataset, batch_size=16, device='cpu',
                                      backbone='resnet18', in_channels=20):
    """Generate [B, 512] embeddings using temporal difference encoding."""
    if backbone == 'resnet18':
        base_model = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet50':
        base_model = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if in_channels != 3:
        base_model = _adapt_conv1(base_model, in_channels)
        print(f"Adapted {backbone} conv1 for {in_channels} input channels.")

    encoder = TemporalDifferenceEncoder(base_model).to(device)
    encoder.eval()
    print(f"TemporalDifferenceEncoder ready  (backbone={backbone}, in_ch={in_channels})")

    for split_name, data in dataset.data.items():
        print(f"  Encoding split '{split_name}' ({len(data)} samples) ...")
        data = _generate_temporal_diff_embeddings(data, encoder, batch_size, device)
        dataset.data[split_name] = data
    return dataset


def _generate_temporal_diff_embeddings(dataset, encoder, batch_size, device):
    """Run the temporal-diff encoder over one split and store embeddings."""
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="TemporalDiff"):
            x_curr = batch['x'].to(device)
            x_base = batch['x_baseline'].to(device)
            emb = encoder(x_curr, x_base)
            embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0).cpu()
    dataset.X = embeddings
    return dataset



class SegDifferenceEncoder(nn.Module):
    """Temporal difference encoder for 3-ch segmentation maps.
    Output: [B, 1536] = concat(f_current, f_baseline, f_diff)."""

    def __init__(self, backbone):
        super().__init__()
        self.shared = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, seg_current, seg_baseline):
        f_curr = self.shared(seg_current).flatten(1)
        f_base = self.shared(seg_baseline).flatten(1)
        f_diff = f_curr - f_base
        return torch.cat([f_curr, f_base, f_diff], dim=1)


def generate_seg_diff_embeddings(dataset, batch_size=16, device='cpu',
                                  backbone='resnet18'):
    """Generate [B, 1536] embeddings from segmentation temporal differences."""
    if backbone == 'resnet18':
        base_model = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet50':
        base_model = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    encoder = SegDifferenceEncoder(base_model).to(device)
    encoder.eval()
    print(f"SegDifferenceEncoder ready  (backbone={backbone})")

    for split_name, data in dataset.data.items():
        print(f"  Encoding seg split '{split_name}' ({len(data)} samples) ...")
        data_loader = DataLoader(data, batch_size=batch_size, num_workers=0)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="SegDiff"):
                x_curr = batch['x'].to(device)
                x_base = batch['x_baseline'].to(device)
                emb = encoder(x_curr, x_base)
                embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0).cpu()
        data.X = embeddings
        dataset.data[split_name] = data
    return dataset



def generate_seg_guided_embeddings(dataset, batch_size=8, device='cpu',
                                   backbone='resnet18', mri_in_channels=20,
                                   seg_in_channels=3, embed_dim=256,
                                   pretrain_epochs=50, pretrain_lr=1e-4,
                                   use_temporal=True):
    """Generate embeddings using the Seg-Guided Dual-Stream Encoder.
    Optionally pretrains on concept prediction before encoding.
    Returns (dataset, encoder)."""
    from src.models.seg_guided_encoder import SegGuidedDualStreamEncoder, SegSupervisedPretrainer

    # Create encoder
    encoder = SegGuidedDualStreamEncoder(
        mri_in_channels=mri_in_channels,
        seg_in_channels=seg_in_channels,
        embed_dim=embed_dim,
        backbone=backbone,
        freeze_backbone=False,
        use_temporal=use_temporal,
    ).to(device)
    print(f"SegGuidedDualStreamEncoder ready  (backbone={backbone}, "
          f"mri_ch={mri_in_channels}, seg_ch={seg_in_channels}, "
          f"embed_dim={embed_dim}, output_dim={encoder.output_dim}, "
          f"use_temporal={use_temporal})")

    # Seg-supervised concept pretraining
    if pretrain_epochs > 0:
        # Get concept info from the training set
        train_ds = dataset.data['train']
        c_info = getattr(dataset, 'c_info', None)
        if c_info is not None:
            concept_names = c_info['names']
            concept_cardinality = c_info['cardinality']
        else:
            # Fallback
            concept_names = [f'concept_{i}' for i in range(train_ds.c.shape[1])]
            concept_cardinality = [1] * train_ds.c.shape[1]

        pretrainer = SegSupervisedPretrainer(
            encoder=encoder,
            concept_names=concept_names,
            concept_cardinality=concept_cardinality,
        ).to(device)

        # Freeze encoder backbones: prevent overfitting on small dataset.
        # Only projection heads + concept heads are updated during pretraining.
        encoder.mri_backbone.requires_grad_(False)
        encoder.seg_backbone.requires_grad_(False)
        print("  Encoder backbones frozen — training projection + concept heads only.")

        trainable_params = [p for p in pretrainer.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params, lr=pretrain_lr, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=pretrain_epochs
        )

        def _seg_guided_collate(batch):
            return {
                'x': torch.stack([b['x'] for b in batch]),
                'x_baseline': torch.stack([b['x_baseline'] for b in batch]),
                'x_seg_curr': torch.stack([b['x_seg_curr'] for b in batch]),
                'x_seg_base': torch.stack([b['x_seg_base'] for b in batch]),
                'c': torch.stack([b['c'] for b in batch]),
                'y': torch.stack([b['y'] for b in batch]),
                'graph': batch[0].get('graph', {}),
            }

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_seg_guided_collate,
        )

        print(f"\n=== Seg-Supervised Concept Pretraining ({pretrain_epochs} epochs) ===")

        # --- Validation loader for early stopping ---
        val_ds = dataset.data.get('val')
        val_loader = None
        if val_ds is not None and len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=0, collate_fn=_seg_guided_collate,
            )

        best_val_loss = float('inf')
        patience_counter = 0
        pretrain_patience = 40
        best_state = None

        pretrainer.train()
        for epoch in range(pretrain_epochs):
            epoch_loss = 0.0
            epoch_concept_losses = {n: 0.0 for n in concept_names}
            n_batches = 0
            for batch in train_loader:
                x_curr_mri = batch['x'].to(device)
                x_base_mri = batch['x_baseline'].to(device)
                x_curr_seg = batch['x_seg_curr'].to(device)
                x_base_seg = batch['x_seg_base'].to(device)
                c_targets = batch['c'].to(device)

                preds, _ = pretrainer(x_curr_mri, x_base_mri, x_curr_seg, x_base_seg)
                loss, loss_dict = pretrainer.compute_loss(preds, c_targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                for key, val in loss_dict.items():
                    if key != 'total':
                        cname = key.split('/')[-1]
                        if cname in epoch_concept_losses:
                            epoch_concept_losses[cname] += val
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            for k in epoch_concept_losses:
                epoch_concept_losses[k] /= max(n_batches, 1)

            # Validation + early stopping
            val_loss_str = ""
            if val_loader is not None:
                pretrainer.eval()
                val_total = 0.0
                val_n = 0
                val_concept = {n: 0.0 for n in concept_names}
                with torch.no_grad():
                    for vbatch in val_loader:
                        vx_c = vbatch['x'].to(device)
                        vx_b = vbatch['x_baseline'].to(device)
                        vxs_c = vbatch['x_seg_curr'].to(device)
                        vxs_b = vbatch['x_seg_base'].to(device)
                        vc = vbatch['c'].to(device)
                        vpreds, _ = pretrainer(vx_c, vx_b, vxs_c, vxs_b)
                        vloss, vld = pretrainer.compute_loss(vpreds, vc)
                        val_total += vloss.item()
                        val_n += 1
                        for key, val in vld.items():
                            if key != 'total':
                                cname = key.split('/')[-1]
                                if cname in val_concept:
                                    val_concept[cname] += val
                avg_val = val_total / max(val_n, 1)
                for k in val_concept:
                    val_concept[k] /= max(val_n, 1)
                val_loss_str = f"  val={avg_val:.4f}"
                pretrainer.train()

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in pretrainer.state_dict().items()}
                else:
                    patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                concept_str = "  ".join(f"{n[:8]}={v:.3f}" for n, v in epoch_concept_losses.items())
                print(f"  Epoch {epoch+1}/{pretrain_epochs}  loss={avg_loss:.4f}{val_loss_str}  "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")
                print(f"    per-concept: {concept_str}")

            if patience_counter >= pretrain_patience:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={pretrain_patience})")
                break

        # Restore best weights
        if best_state is not None:
            pretrainer.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            print(f"  Restored best pretrainer (val_loss={best_val_loss:.4f})")
        print("Pretraining complete.\n")

    # Generate embeddings for all splits
    encoder.eval()

    def _seg_guided_collate(batch):
        result = {
            'x': torch.stack([b['x'] for b in batch]),
            'x_baseline': torch.stack([b['x_baseline'] for b in batch]),
            'c': torch.stack([b['c'] for b in batch]),
            'y': torch.stack([b['y'] for b in batch]),
            'graph': batch[0].get('graph', {}),
        }
        if 'x_seg_curr' in batch[0]:
            result['x_seg_curr'] = torch.stack([b['x_seg_curr'] for b in batch])
            result['x_seg_base'] = torch.stack([b['x_seg_base'] for b in batch])
        return result

    for split_name, data in dataset.data.items():
        print(f"  Encoding seg-guided split '{split_name}' ({len(data)} samples) ...")
        data_loader = DataLoader(
            data, batch_size=batch_size, num_workers=0,
            collate_fn=_seg_guided_collate,
        )
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"SegGuided-{split_name}"):
                x_curr_mri = batch['x'].to(device)
                x_base_mri = batch['x_baseline'].to(device)
                x_curr_seg = batch['x_seg_curr'].to(device)
                x_base_seg = batch['x_seg_base'].to(device)
                emb = encoder(x_curr_mri, x_base_mri, x_curr_seg, x_base_seg)
                embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0).cpu()
        data.X = embeddings
        dataset.data[split_name] = data
        print(f"    → {split_name}: X shape = {data.X.shape}")

    return dataset, encoder


def generate_img_embeddings(dataset: torch.utils.data.Dataset,
                           batch_size: int = 32,
                           device: str = 'cpu',
                           backbone: str = 'resnet18') -> None:
    
    if backbone == 'resnet18':
        input_encoder = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet50':
        input_encoder = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
    try:
        sample_x = dataset.data['train'][0]['x']
        in_channels = sample_x.shape[0]
        if in_channels != 3:
            original_conv1 = input_encoder.conv1
            new_conv1 = torch.nn.Conv2d(in_channels, original_conv1.out_channels, 
                                        kernel_size=original_conv1.kernel_size, 
                                        stride=original_conv1.stride, 
                                        padding=original_conv1.padding, 
                                        bias=original_conv1.bias is not None)
            with torch.no_grad():
                new_weights = original_conv1.weight.clone()
                if in_channels > 3:
                    repeats = (in_channels + 2) // 3
                    new_weights = new_weights.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                    new_weights = new_weights * (3.0 / in_channels)
                else:
                    new_weights = new_weights[:, :in_channels, :, :]
                new_conv1.weight.copy_(new_weights)
                if original_conv1.bias is not None:
                    new_conv1.bias.copy_(original_conv1.bias)
            input_encoder.conv1 = new_conv1
            print(f"Adjusted {backbone} input channels to {in_channels}.")
    except Exception as e:
        print(f"Could not adjust input channels: {e}")

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

    elif dataset_name == 'lumiere':
        input_mode = cfg.dataset.loader.get('input_mode', 'mri')
        # Store the mode so LumiereDataset.split() can propagate it to inner datasets
        dataset._input_mode = input_mode
        dataset.split()
        dataset.data['train'].update_lists()
        dataset.data['val'].update_lists()
        dataset.data['test'].update_lists()
        dataset = maybe_reduce(cfg.dataset.get('reduce_fraction', None), dataset)

        print(f"Lumiere input mode: {input_mode}")

        if input_mode == 'radiomic':
            # Bypass images entirely: use z-scored raw radiomic feature vectors.
            # Sanity check to confirm whether concepts are predictive of task.
            for split_name, ds in dataset.data.items():
                raw = np.array([ds.all_concepts[i] for i in ds._indices], dtype=np.float32)
                means = ds.concept_means
                stds  = ds.concept_stds
                z_scored = (raw - means) / stds
                ds.X = torch.from_numpy(z_scored).float()  # [N, N_CONCEPTS]
                print(f"  {split_name}: X shape = {ds.X.shape}")
        elif input_mode == 'seg':
            # Seg-based Temporal Difference: 3-ch seg maps through standard ResNet18
            import gc; gc.collect()
            dataset = generate_seg_diff_embeddings(
                dataset, batch_size=16, device=device, backbone=backbone
            )
        elif input_mode == 'seg_guided':
            # Seg-Guided Dual-Stream: MRI + seg through shared backbone
            import gc; gc.collect()
            pretrain_epochs = cfg.dataset.get('seg_guided_pretrain_epochs', 50)
            pretrain_lr = cfg.dataset.get('seg_guided_pretrain_lr', 1e-4)
            embed_dim = cfg.dataset.get('seg_guided_embed_dim', 256)
            dataset, _ = generate_seg_guided_embeddings(
                dataset, batch_size=8, device=device,
                backbone=backbone, mri_in_channels=20,
                seg_in_channels=3, embed_dim=embed_dim,
                pretrain_epochs=pretrain_epochs,
                pretrain_lr=pretrain_lr,
            )
        elif input_mode == 'seg_guided_no_temporal':
            # Seg-Guided without temporal difference (current only)
            import gc; gc.collect()
            pretrain_epochs = cfg.dataset.get('seg_guided_pretrain_epochs', 50)
            pretrain_lr = cfg.dataset.get('seg_guided_pretrain_lr', 1e-4)
            embed_dim = cfg.dataset.get('seg_guided_embed_dim', 256)
            dataset, _ = generate_seg_guided_embeddings(
                dataset, batch_size=8, device=device,
                backbone=backbone, mri_in_channels=20,
                seg_in_channels=3, embed_dim=embed_dim,
                pretrain_epochs=pretrain_epochs,
                pretrain_lr=pretrain_lr,
                use_temporal=False,
            )
        elif input_mode in ['m3d_lamed', 'medicalnet']:
            # Raw full 3D volume mode. The 3D encoder now lives
            # inside C2BM, replacing the C2BM MLP encoder instead of producing
            # a second-stage precomputed embedding.
            print("  using raw full 3D volumes; C2BM will encode them end-to-end")
        elif input_mode == 'medicalnet_cached':
            # Load precomputed MedicalNet embeddings from a per-fold cache.
            # cfg.dataset.embeddings_path is a prefix; the per-fold file is
            # f"{prefix}_fold{cv_fold}.pt" and contains a dict
            # {global_idx: tensor[D]}.
            import torch as _torch
            prefix = cfg.dataset.get('embeddings_path')
            cv_fold = cfg.dataset.loader.get('cv_fold', 0)
            cache_path = f"{prefix}_fold{cv_fold}.pt"
            print(f"  loading cached MedicalNet embeddings: {cache_path}")
            blob = _torch.load(cache_path, map_location='cpu', weights_only=False)
            emb_by_global = blob['embeddings'] if isinstance(blob, dict) and 'embeddings' in blob else blob
            for split_name, ds in dataset.data.items():
                stack = []
                for g in ds._indices:
                    g_int = int(g)
                    if g_int not in emb_by_global:
                        raise KeyError(
                            f"Missing embedding for global_idx={g_int} (split={split_name}) "
                            f"in {cache_path}"
                        )
                    stack.append(emb_by_global[g_int])
                ds.X = _torch.stack(stack, dim=0).float()
                print(f"  {split_name}: X shape = {tuple(ds.X.shape)}")
        else:  # 'mri'
            import gc; gc.collect()
            dataset = generate_temporal_diff_embeddings(
                dataset, batch_size=8, device=device,
                backbone=backbone, in_channels=20
            )         

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
