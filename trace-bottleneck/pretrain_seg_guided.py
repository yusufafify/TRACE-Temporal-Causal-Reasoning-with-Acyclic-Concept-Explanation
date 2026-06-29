"""
Standalone pretraining script for the Seg-Guided Dual-Stream Encoder.

Usage:
    python pretrain_seg_guided.py \
        --data_root /home/group2/dataset/Lumiere/ \
        --img_dir /home/group2/dataset/processed_images/ \
        --epochs 50 \
        --lr 1e-4 \
        --batch_size 8 \
        --embed_dim 256 \
        --backbone resnet18 \
        --save_path pretrained_seg_guided_encoder.pt

This pretrains the encoder on a seg-supervised concept prediction task:
  - Continuous concepts (vol, delta%, baseline vol) → MSE loss
  - Binary concepts (new_lesion_flag, prog_threshold_flag) → CE loss

After pretraining, the encoder weights can be loaded into the main pipeline
by setting `pretrained_encoder_path` in the config or by using the
`input_mode: seg_guided` with `pretrain_epochs > 0`.
"""

import argparse
import os
import sys
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.seg_guided_encoder import SegGuidedDualStreamEncoder, SegSupervisedPretrainer
from src.data.datasets.lumiere import (
    _LumiereDataset, LumiereDataset, CONCEPT_NAMES, BINARY_CONCEPTS, RANO_CONCEPTS
)


def seg_guided_collate(batch):
    """Collate function for seg_guided mode with 4 input streams."""
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


def create_dataset(data_root, img_dir):
    """Create train/val/test splits using the existing LumiereDataset."""
    dataset = LumiereDataset(
        root_dir=data_root,
        img_dir=img_dir,
        input_mode='seg_guided',
        task_cardinality=4,
        use_delta_only=False,
        use_8_channels=True,
    )
    dataset.split()
    dataset.data['train'].update_lists()
    dataset.data['val'].update_lists()
    dataset.data['test'].update_lists()
    return dataset


def evaluate(pretrainer, data_loader, device, concept_names, concept_cardinality):
    """Run evaluation on a data loader and return metrics."""
    pretrainer.eval()
    total_loss = 0.0
    n_batches = 0
    per_concept_loss = {name: 0.0 for name in concept_names}

    with torch.no_grad():
        for batch in data_loader:
            x_curr_mri = batch['x'].to(device)
            x_base_mri = batch['x_baseline'].to(device)
            x_curr_seg = batch['x_seg_curr'].to(device)
            x_base_seg = batch['x_seg_base'].to(device)
            c_targets = batch['c'].to(device)

            preds, _ = pretrainer(x_curr_mri, x_base_mri, x_curr_seg, x_base_seg)
            loss, loss_dict = pretrainer.compute_loss(preds, c_targets)

            total_loss += loss.item()
            n_batches += 1

            for key, val in loss_dict.items():
                if key != 'total':
                    cname = key.split('/')[-1]
                    if cname in per_concept_loss:
                        per_concept_loss[cname] += val

    avg_loss = total_loss / max(n_batches, 1)
    for k in per_concept_loss:
        per_concept_loss[k] /= max(n_batches, 1)

    pretrainer.train()
    return avg_loss, per_concept_loss


def main():
    parser = argparse.ArgumentParser(description='Pretrain Seg-Guided Encoder')
    parser.add_argument('--data_root', type=str,
                        default='/home/group2/dataset/Lumiere/')
    parser.add_argument('--img_dir', type=str,
                        default='/home/group2/dataset/processed_images/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--save_path', type=str,
                        default='pretrained_seg_guided_encoder.pt')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not set)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation during pretraining')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration: {vars(args)}")

    # 1. Create dataset
    print("\n=== Creating dataset ===")
    dataset = create_dataset(args.data_root, args.img_dir)

    concept_names = dataset.c_info['names']
    concept_cardinality = dataset.c_info['cardinality']
    print(f"Concepts ({len(concept_names)}): {concept_names}")
    print(f"Cardinality: {concept_cardinality}")
    print(f"Train: {len(dataset.data['train'])} samples")
    print(f"Val:   {len(dataset.data['val'])} samples")
    print(f"Test:  {len(dataset.data['test'])} samples")

    # 2. Create model
    print("\n=== Creating Seg-Guided Dual-Stream Encoder ===")
    encoder = SegGuidedDualStreamEncoder(
        mri_in_channels=20,
        seg_in_channels=3,
        embed_dim=args.embed_dim,
        backbone=args.backbone,
        freeze_backbone=False,
    ).to(device)

    pretrainer = SegSupervisedPretrainer(
        encoder=encoder,
        concept_names=concept_names,
        concept_cardinality=concept_cardinality,
    ).to(device)

    n_params = sum(p.numel() for p in pretrainer.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Encoder output dim: {encoder.output_dim}")

    # 3. Data loaders
    augment_fn = None
    if args.augment:
        try:
            from src.data.augmentation import get_mri_augmentation
            augment_fn = get_mri_augmentation(use_torchio=True)
            print("Data augmentation: ENABLED (torchio)")
        except ImportError:
            from src.data.augmentation import get_mri_augmentation
            augment_fn = get_mri_augmentation(use_torchio=False)
            print("Data augmentation: ENABLED (basic fallback)")

    train_loader = DataLoader(
        dataset.data['train'], batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=seg_guided_collate, drop_last=False,
    )
    val_loader = DataLoader(
        dataset.data['val'], batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=seg_guided_collate,
    )
    test_loader = DataLoader(
        dataset.data['test'], batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=seg_guided_collate,
    )

    # 4. Training
    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n=== Pretraining ({args.epochs} epochs, patience={args.patience}) ===")
    start_time = time.time()

    for epoch in range(args.epochs):
        pretrainer.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x_curr_mri = batch['x'].to(device)
            x_base_mri = batch['x_baseline'].to(device)
            x_curr_seg = batch['x_seg_curr'].to(device)
            x_base_seg = batch['x_seg_base'].to(device)
            c_targets = batch['c'].to(device)

            # Optional augmentation
            if augment_fn is not None:
                for b in range(x_curr_mri.size(0)):
                    x_curr_mri[b], x_curr_seg[b] = augment_fn(
                        x_curr_mri[b], x_curr_seg[b]
                    )
                    x_base_mri[b], x_base_seg[b] = augment_fn(
                        x_base_mri[b], x_base_seg[b]
                    )

            preds, _ = pretrainer(x_curr_mri, x_base_mri, x_curr_seg, x_base_seg)
            loss, loss_dict = pretrainer.compute_loss(preds, c_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss, val_per_concept = evaluate(
            pretrainer, val_loader, device, concept_names, concept_cardinality
        )

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'pretrainer_state_dict': pretrainer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'embed_dim': args.embed_dim,
                'backbone': args.backbone,
                'concept_names': concept_names,
                'concept_cardinality': concept_cardinality,
            }, args.save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                  f"train={avg_train_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}  patience={patience_counter}/{args.patience}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"time={elapsed:.1f}s")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 5. Final evaluation
    print("\n=== Loading best model and evaluating on test set ===")
    ckpt = torch.load(args.save_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    pretrainer_eval = SegSupervisedPretrainer(
        encoder=encoder,
        concept_names=concept_names,
        concept_cardinality=concept_cardinality,
    ).to(device)
    pretrainer_eval.load_state_dict(ckpt['pretrainer_state_dict'])

    test_loss, test_per_concept = evaluate(
        pretrainer_eval, test_loader, device, concept_names, concept_cardinality
    )
    print(f"\nTest loss: {test_loss:.4f}")
    print("Per-concept test losses:")
    for name, val in test_per_concept.items():
        print(f"  {name}: {val:.4f}")

    # Save training history
    history_path = args.save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best encoder saved to {args.save_path}")
    print(f"  Best val loss: {best_val_loss:.4f}  at epoch {ckpt['epoch']+1}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
