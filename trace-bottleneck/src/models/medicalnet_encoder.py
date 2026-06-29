import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


DEFAULT_MEDICALNET_ROOT = (
    "/home/group2/youssef/XAI_Disease_progression_GP/Label-free-CBM/"
    "option4-siamese-explicit-delta/MedicalNet"
)
DEFAULT_PRETRAINED_PATH = os.path.join(
    DEFAULT_MEDICALNET_ROOT, "pretrain", "resnet_18_23dataset.pth"
)


class MedicalNetC2BMEncoder(nn.Module):
    """C2BM encoder backed by Youssef's MedicalNet ResNet-18 setup.

    Baseline and follow-up MRI volumes are encoded with one MedicalNet
    backbone. Optional segmentation volumes are encoded with a parallel
    MedicalNet backbone. Each stream is fused as
    [baseline, follow-up, follow-up - baseline]. Optional clinical scalars
    can be appended before projection into the C2BM latent space.
    """

    def __init__(
        self,
        output_dim: int,
        medicalnet_root: str = DEFAULT_MEDICALNET_ROOT,
        pretrained_path: str = DEFAULT_PRETRAINED_PATH,
        in_channels: int = 4,
        seg_in_channels: int = 3,
        use_segmentation: bool = True,
        clinical_dim: int = 0,
        freeze_backbone: bool = False,
        sample_size: int = 128,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.in_channels = int(in_channels)
        self.seg_in_channels = int(seg_in_channels)
        self.use_segmentation = bool(use_segmentation)
        self.clinical_dim = int(clinical_dim)
        self.sample_size = int(sample_size)
        self.target_shape = (self.sample_size, self.sample_size, self.sample_size)
        self.encoder = self._build_encoder(
            medicalnet_root=medicalnet_root,
            pretrained_path=pretrained_path,
            in_channels=self.in_channels,
            freeze_backbone=freeze_backbone,
            stream_name="MRI",
        )
        if self.use_segmentation:
            self.seg_encoder = self._build_encoder(
                medicalnet_root=medicalnet_root,
                pretrained_path=pretrained_path,
                in_channels=self.seg_in_channels,
                freeze_backbone=freeze_backbone,
                stream_name="SEG",
            )
        else:
            self.seg_encoder = None

        self.encoder_dim = 512
        self.mri_fusion_dim = 3 * self.encoder_dim
        self.seg_fusion_dim = 3 * self.encoder_dim if self.use_segmentation else 0
        self.fusion_dim = self.mri_fusion_dim + self.seg_fusion_dim + self.clinical_dim
        self.projector = nn.Sequential(
            nn.Linear(self.fusion_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.LeakyReLU(0.1),
        )
        print(
            f"  MedicalNet C2BM projection: {self.fusion_dim} -> {self.output_dim} "
            f"(MRI {self.mri_fusion_dim}"
            f"{' + SEG ' + str(self.seg_fusion_dim) if self.use_segmentation else ''}"
            f"{' + clinical ' + str(self.clinical_dim) if self.clinical_dim else ''})"
        )

    def _build_encoder(self, medicalnet_root, pretrained_path, in_channels,
                       freeze_backbone, stream_name):
        if medicalnet_root not in sys.path:
            sys.path.insert(0, medicalnet_root)
        from models.resnet import resnet18

        model = resnet18(
            sample_input_W=self.sample_size,
            sample_input_H=self.sample_size,
            sample_input_D=self.sample_size,
            shortcut_type="B",
            no_cuda=False,
            num_seg_classes=1,
        )

        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading MedicalNet ResNet-18 weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print("  loaded pretrained MedicalNet ResNet-18 weights")
        else:
            print(f"Warning: MedicalNet pretrained weights not found: {pretrained_path}")

        self._initialize_multimodal_conv1(model, in_channels, stream_name)
        model.conv_seg = nn.Identity()

        if freeze_backbone:
            model.requires_grad_(False)
            print(f"  MedicalNet {stream_name} backbone frozen")

        return model

    def _initialize_multimodal_conv1(self, model, in_channels, stream_name):
        original_conv1 = model.conv1
        old_weight = original_conv1.weight.data
        if old_weight.shape[1] == in_channels:
            return

        new_conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )
        with torch.no_grad():
            if old_weight.shape[1] == 1:
                new_conv1.weight.copy_(
                    old_weight.repeat(1, in_channels, 1, 1, 1) / float(in_channels)
                )
            else:
                keep = min(old_weight.shape[1], in_channels)
                new_conv1.weight[:, :keep].copy_(old_weight[:, :keep])
                if in_channels > keep:
                    mean_weight = old_weight[:, :keep].mean(dim=1, keepdim=True)
                    new_conv1.weight[:, keep:].copy_(
                        mean_weight.repeat(1, in_channels - keep, 1, 1, 1)
                    )
            if original_conv1.bias is not None:
                new_conv1.bias.copy_(original_conv1.bias.data)

        model.conv1 = new_conv1
        print(f"  initialized MedicalNet {stream_name} conv1 for {in_channels} channels")

    def encode_volume(self, volume, encoder, expected_channels, stream_name):
        if volume.ndim != 5:
            raise ValueError(
                f"MedicalNet {stream_name} expected [B, C, D, H, W] input, "
                f"got {tuple(volume.shape)}"
            )
        if volume.shape[1] != expected_channels:
            raise ValueError(
                f"MedicalNet {stream_name} expected {expected_channels} channels, "
                f"got {volume.shape[1]}"
            )

        volume = volume.float()
        if tuple(volume.shape[-3:]) != self.target_shape:
            volume = F.interpolate(
                volume,
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            )

        if self.training:
            features = grad_checkpoint(encoder, volume, use_reentrant=False)
        else:
            features = encoder(volume)
        if isinstance(features, tuple):
            features = features[0]
        features = F.adaptive_avg_pool3d(features, 1)
        return features.view(features.size(0), -1)

    @staticmethod
    def _temporal_triplet(baseline_features, followup_features):
        return torch.cat(
            [
                baseline_features,
                followup_features,
                followup_features - baseline_features,
            ],
            dim=1,
        )

    def forward(self, followup, baseline=None, x_seg_curr=None, x_seg_base=None,
                clinical_features=None):
        if baseline is None:
            baseline = followup

        baseline_features = self.encode_volume(
            baseline, self.encoder, self.in_channels, "MRI"
        )
        followup_features = self.encode_volume(
            followup, self.encoder, self.in_channels, "MRI"
        )
        fused_parts = [self._temporal_triplet(baseline_features, followup_features)]

        if self.use_segmentation:
            if x_seg_curr is None or x_seg_base is None:
                raise ValueError(
                    "MedicalNet segmentation stream is enabled but x_seg_curr/x_seg_base "
                    "were not provided by the dataloader."
                )
            seg_base_features = self.encode_volume(
                x_seg_base, self.seg_encoder, self.seg_in_channels, "SEG"
            )
            seg_curr_features = self.encode_volume(
                x_seg_curr, self.seg_encoder, self.seg_in_channels, "SEG"
            )
            fused_parts.append(self._temporal_triplet(seg_base_features, seg_curr_features))

        if self.clinical_dim > 0:
            if clinical_features is None:
                raise ValueError(
                    "MedicalNet clinical features are enabled but clinical_features "
                    "was not provided by the dataloader."
                )
            clinical_features = clinical_features.float().view(followup.shape[0], -1)
            if clinical_features.shape[1] != self.clinical_dim:
                raise ValueError(
                    f"Expected {self.clinical_dim} clinical features, "
                    f"got {clinical_features.shape[1]}"
                )
            fused_parts.append(clinical_features)

        fused = torch.cat(fused_parts, dim=1)
        return self.projector(fused)
