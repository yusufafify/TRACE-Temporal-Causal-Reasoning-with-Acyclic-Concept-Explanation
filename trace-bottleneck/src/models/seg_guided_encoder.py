"""Seg-Guided Dual-Stream Encoder for longitudinal brain tumor MRI.

Processes current/baseline MRI + segmentation through shared ResNet backbones,
computes temporal feature differences, and outputs a 768-dim embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SegGuidedDualStreamEncoder(nn.Module):
    """Dual-stream encoder with shared MRI and segmentation backbones.

    Output dim: 3 * embed_dim (with temporal) or embed_dim (without).
    """

    def __init__(self, mri_in_channels=20, seg_in_channels=3, embed_dim=256,
                 backbone='resnet18', freeze_backbone=False, use_temporal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_temporal = use_temporal
        self.output_dim = 3 * embed_dim if use_temporal else embed_dim

        mri_base = self._create_backbone(backbone)
        mri_base = self._adapt_conv1(mri_base, mri_in_channels)
        self.mri_backbone = nn.Sequential(*list(mri_base.children())[:-1])
        feat_dim = self._get_feat_dim(backbone)

        seg_base = self._create_backbone(backbone)
        if seg_in_channels != 3:
            seg_base = self._adapt_conv1(seg_base, seg_in_channels)
        self.seg_backbone = nn.Sequential(*list(seg_base.children())[:-1])

        fused_dim = feat_dim * 2
        # Separate projection heads per timepoint (backbone is shared)
        self.proj_curr = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.1),
        )
        self.proj_base = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.1),
        )

        if freeze_backbone:
            self._freeze_backbones()

    @staticmethod
    def _create_backbone(name):
        if name == 'resnet18':
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif name == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    @staticmethod
    def _get_feat_dim(name):
        return 512 if name == 'resnet18' else 2048

    @staticmethod
    def _adapt_conv1(backbone, in_channels):
        """Tile pretrained conv1 weights for multi-channel input."""
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

    def _freeze_backbones(self):
        for param in self.mri_backbone.parameters():
            param.requires_grad = False
        for param in self.seg_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        for param in self.mri_backbone.parameters():
            param.requires_grad = True
        for param in self.seg_backbone.parameters():
            param.requires_grad = True

    def _encode_pair(self, x_mri, x_seg):
        """Encode a single timepoint through both backbones → [B, feat_dim*2]."""
        f_mri = self.mri_backbone(x_mri).flatten(1)
        f_seg = self.seg_backbone(x_seg).flatten(1)
        return torch.cat([f_mri, f_seg], dim=-1)

    def forward(self, x_curr_mri, x_base_mri, x_curr_seg, x_base_seg):
        """Returns [B, 3*embed_dim] (temporal) or [B, embed_dim] (no temporal)."""
        f_curr = self._encode_pair(x_curr_mri, x_curr_seg)

        if self.use_temporal:
            f_base = self._encode_pair(x_base_mri, x_base_seg)

            # Separate projection heads allow specialization per timepoint
            f_curr_proj = self.proj_curr(f_curr)
            f_base_proj = self.proj_base(f_base)
            # Difference computed in projected space
            f_diff_proj = f_curr_proj - f_base_proj

            embedding = torch.cat([f_curr_proj, f_base_proj, f_diff_proj], dim=-1)
        else:
            embedding = self.proj_curr(f_curr)
        return embedding


class SegSupervisedPretrainer(nn.Module):
    """Pretrain the encoder on a concept prediction task (MSE for continuous, CE for binary)."""

    def __init__(self, encoder, concept_names, concept_cardinality):
        super().__init__()
        self.encoder = encoder
        self.concept_names = concept_names
        self.concept_cardinality = concept_cardinality

        embed_dim = encoder.output_dim
        self.concept_heads = nn.ModuleDict()
        for name, card in zip(concept_names, concept_cardinality):
            out_dim = 1 if card == 1 else card
            self.concept_heads[name] = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.4),          # aggressive: kills 40% of neurons
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(128, out_dim),
            )

    def forward(self, x_curr_mri, x_base_mri, x_curr_seg, x_base_seg):
        """Return per-concept predictions and the raw embedding."""
        emb = self.encoder(x_curr_mri, x_base_mri, x_curr_seg, x_base_seg)
        # Gaussian noise injection: cheap train-time augmentation in embedding space
        if self.training:
            emb = emb + torch.randn_like(emb) * 0.1
        preds = {}
        for name, head in self.concept_heads.items():
            preds[name] = head(emb)
        return preds, emb

    def compute_loss(self, preds, c_targets):
        """Sum of per-concept losses (MSE or CE). Returns (total_loss, loss_dict)."""
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=c_targets.device)

        for i, (name, card) in enumerate(
            zip(self.concept_names, self.concept_cardinality)
        ):
            pred = preds[name]
            target = c_targets[:, i]

            if card == 1:
                loss = F.mse_loss(pred.squeeze(-1), target)
                loss_dict[f'mse/{name}'] = loss.item()
            else:
                loss = F.cross_entropy(pred, target.long())
                loss_dict[f'ce/{name}'] = loss.item()

            total_loss = total_loss + loss

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
