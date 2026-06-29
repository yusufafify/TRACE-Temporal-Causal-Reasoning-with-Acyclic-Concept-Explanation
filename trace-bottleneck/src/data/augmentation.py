"""
MRI-safe data augmentation for the Seg-Guided Dual-Stream Encoder.

Uses torchio-compatible transforms that preserve segmentation-image
correspondence. Segmentation masks use nearest-neighbour interpolation
automatically (torchio LabelMap), so label integrity is maintained.

For environments without torchio, provides a fallback using basic
torch/torchvision transforms.
"""

import torch
import numpy as np

try:
    import torchio as tio
    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False


def get_mri_augmentation(use_torchio=True, intensity_p=0.5, spatial_p=0.5):
    """Get an augmentation pipeline for paired MRI + segmentation data.

    Parameters
    ----------
    use_torchio : bool
        Prefer torchio transforms (handles seg automatically). Falls back
        to basic transforms if torchio not installed.
    intensity_p : float
        Probability of intensity transforms.
    spatial_p : float
        Probability of spatial transforms.

    Returns
    -------
    augment_fn : callable
        Takes (mri_tensor, seg_tensor) → (aug_mri, aug_seg).
        Both tensors are [C, H, W] (2D multi-slice).
    """
    if use_torchio and HAS_TORCHIO:
        return TorchIOAugmentation(intensity_p=intensity_p, spatial_p=spatial_p)
    else:
        return BasicAugmentation(p=spatial_p)


class TorchIOAugmentation:
    """Augment paired MRI + segmentation using torchio.

    torchio automatically handles LabelMap with nearest-neighbour
    interpolation, preserving label integrity.

    Expects 2D multi-slice inputs [C, H, W]. Internally adds a dummy depth
    dimension for torchio compatibility, then squeezes it back.
    """

    def __init__(self, intensity_p=0.5, spatial_p=0.5):
        # Spatial transforms (applied to both image and label)
        self.spatial = tio.Compose([
            tio.RandomFlip(axes=['LR'], p=spatial_p),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=15,
                translation=5,
                p=spatial_p,
            ),
        ])

        # Intensity transforms (applied to image only, not labels)
        self.intensity = tio.Compose([
            tio.RandomNoise(std=(0, 0.05), p=intensity_p),
            tio.RandomBiasField(coefficients=0.3, p=intensity_p),
        ])

    def __call__(self, mri_tensor, seg_tensor):
        """Augment a paired MRI + seg sample.

        Parameters
        ----------
        mri_tensor : Tensor [C_mri, H, W] — e.g. [20, 224, 224]
        seg_tensor : Tensor [C_seg, H, W] — e.g. [3, 224, 224]

        Returns
        -------
        aug_mri : Tensor [C_mri, H, W]
        aug_seg : Tensor [C_seg, H, W]
        """
        # torchio expects [C, D, H, W] — add singleton depth
        mri_4d = mri_tensor.unsqueeze(1).float()  # [C, 1, H, W]
        seg_4d = seg_tensor.unsqueeze(1).float()   # [C, 1, H, W]

        # Create a torchio Subject
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=mri_4d),
            seg=tio.LabelMap(tensor=seg_4d),
        )

        # Apply spatial transforms (both image and seg)
        subject = self.spatial(subject)

        # Apply intensity transforms (image only)
        # torchio applies intensity transforms only to ScalarImage, not LabelMap
        subject = self.intensity(subject)

        # Extract and squeeze depth
        aug_mri = subject.mri.data.squeeze(1)  # [C, H, W]
        aug_seg = subject.seg.data.squeeze(1)   # [C, H, W]

        return aug_mri, aug_seg


class BasicAugmentation:
    """Fallback augmentation using basic torch operations.

    Only spatial (flip + small rotation). No intensity transforms.
    Segmentation is transformed identically to the MRI.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, mri_tensor, seg_tensor):
        """Augment paired MRI + seg.

        Parameters
        ----------
        mri_tensor : Tensor [C, H, W]
        seg_tensor : Tensor [C, H, W]

        Returns
        -------
        aug_mri, aug_seg : Tensor [C, H, W] each
        """
        if torch.rand(1).item() < self.p:
            # Random horizontal flip
            mri_tensor = torch.flip(mri_tensor, dims=[-1])
            seg_tensor = torch.flip(seg_tensor, dims=[-1])

        if torch.rand(1).item() < self.p:
            # Random vertical flip
            mri_tensor = torch.flip(mri_tensor, dims=[-2])
            seg_tensor = torch.flip(seg_tensor, dims=[-2])

        if torch.rand(1).item() < self.p:
            # Random 90° rotation (k = 1, 2, or 3)
            k = torch.randint(1, 4, (1,)).item()
            mri_tensor = torch.rot90(mri_tensor, k, dims=[-2, -1])
            seg_tensor = torch.rot90(seg_tensor, k, dims=[-2, -1])

        # Add small Gaussian noise to MRI only
        if torch.rand(1).item() < self.p * 0.5:
            noise = torch.randn_like(mri_tensor) * 0.03
            mri_tensor = mri_tensor + noise

        return mri_tensor, seg_tensor
