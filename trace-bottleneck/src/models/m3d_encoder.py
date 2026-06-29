"""M3D-LaMed 3D medical vision encoder for longitudinal MRI analysis.

Loads the 3D vision backbone from M3D-LaMed-Llama-2-7B, discards the LLM,
and wraps it with a temporal difference projection head compatible with the
seg_guided pipeline.

Model: GoodBaiBai88/M3D-LaMed-Llama-2-7B (HuggingFace)
Required VRAM: ~6–8 GB for the vision encoder alone in fp16
Input: 20-channel MRI [B, 20, H, W] per timepoint (treated as 3D volume)
Output: [B, 3 * embed_dim] — concat(f_curr, f_base, f_curr − f_base)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# M3D-LaMed processes 3D volumes of this canonical size
_M3D_DEPTH  = 32
_M3D_HEIGHT = 256
_M3D_WIDTH  = 256


def _try_extract_vision_encoder(model):
    """Try common attribute paths used by LLaVA-style 3D medical models."""
    candidates = [
        # M3D-LaMed / LLaVA-Med style
        lambda m: m.model.vision_tower,
        lambda m: m.model.vision_tower.vision_tower,
        lambda m: m.vision_tower,
        # Alternative naming
        lambda m: m.model.visual_encoder,
        lambda m: m.visual_encoder,
        lambda m: m.model.image_encoder,
        lambda m: m.model.mm_encoder,
    ]
    for fn in candidates:
        try:
            enc = fn(model)
            if enc is not None:
                return enc
        except AttributeError:
            continue

    # Last resort: print available children and fail with a helpful message
    top_level = [n for n, _ in model.named_children()]
    if hasattr(model, 'model'):
        second_level = [n for n, _ in model.model.named_children()]
    else:
        second_level = []
    raise RuntimeError(
        "Could not locate vision encoder in M3D-LaMed.\n"
        f"  Top-level children:    {top_level}\n"
        f"  model.* children:      {second_level}\n"
        "Please update _try_extract_vision_encoder() with the correct path."
    )


class M3DLamedTemporalEncoder(nn.Module):
    """Temporal difference encoder backed by M3D-LaMed's 3D vision tower.

    The vision tower is loaded frozen (no grad). Only the lightweight
    projection heads are trainable, which can optionally be fine-tuned
    via seg-supervised pretraining.
    """

    MODEL_ID = "GoodBaiBai88/M3D-LaMed-Llama-2-7B"

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Load 3D vision backbone ──────────────────────────────────────────
        vision_enc, feat_dim = self._load_and_probe()
        self.vision_enc = vision_enc          # frozen
        self._feat_dim  = feat_dim

        # ── Lightweight trainable projection heads ───────────────────────────
        # Separate heads for current and baseline allow specialisation
        self.proj_curr = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.1),
        )
        self.proj_base = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.1),
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_and_probe(self):
        """Load M3D-LaMed, extract vision encoder, probe output dimension."""
        from transformers import AutoModelForCausalLM

        print(f"\nLoading M3D-LaMed from HuggingFace: {self.MODEL_ID}")
        print("  (this may take several minutes on first run — model is ~13 GB)\n")

        full_model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",            # load to CPU; caller moves to GPU
            low_cpu_mem_usage=True,
        )

        vision_enc = _try_extract_vision_encoder(full_model)
        print(f"  Vision encoder extracted: {type(vision_enc).__name__}")

        # Convert to float32 for numerical stability in downstream heads
        vision_enc = vision_enc.float()

        # Freeze — we use the pretrained 3D features as-is
        vision_enc.requires_grad_(False)

        # Probe output dimensionality with a dummy forward
        dummy = torch.zeros(1, 1, _M3D_DEPTH, _M3D_HEIGHT, _M3D_WIDTH)
        with torch.no_grad():
            try:
                out = vision_enc(dummy)
            except Exception as e:
                raise RuntimeError(
                    f"M3D vision encoder forward failed with dummy input "
                    f"[1,1,{_M3D_DEPTH},{_M3D_HEIGHT},{_M3D_WIDTH}]: {e}"
                )
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Pool spatial dims if the encoder returns a feature map / token seq
        feat = out.flatten(1)
        feat_dim = feat.shape[-1]
        print(f"  Vision encoder output dim: {feat_dim}")

        # Free full model (LLM weights) immediately to reclaim VRAM
        del full_model
        torch.cuda.empty_cache()

        return vision_enc, feat_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a multi-slice MRI volume.

        Args:
            x: [B, n_slices, H, W]  (n_slices treated as depth dimension)
        Returns:
            features: [B, feat_dim]
        """
        B, C, H, W = x.shape   # C == number of MRI slices

        # Reshape: [B, C, H, W] → [B, 1, C, H, W]  (1 = grayscale channel)
        vol = x.unsqueeze(1).float()

        # Resize to M3D's expected input size using trilinear interpolation
        vol = F.interpolate(
            vol,
            size=(_M3D_DEPTH, _M3D_HEIGHT, _M3D_WIDTH),
            mode="trilinear",
            align_corners=False,
        )

        with torch.no_grad():
            out = self.vision_enc(vol)

        if isinstance(out, (tuple, list)):
            out = out[0]

        return out.flatten(1).float()   # [B, feat_dim]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x_curr: torch.Tensor, x_base: torch.Tensor) -> torch.Tensor:
        """Compute temporal-difference embedding.

        Args:
            x_curr: [B, n_slices, H, W]  current timepoint MRI
            x_base: [B, n_slices, H, W]  baseline timepoint MRI
        Returns:
            embedding: [B, 3 * embed_dim]  (curr ‖ base ‖ diff)
        """
        f_curr = self._encode(x_curr)       # [B, feat_dim]
        f_base = self._encode(x_base)

        e_curr = self.proj_curr(f_curr)     # [B, embed_dim]
        e_base = self.proj_base(f_base)
        e_diff = e_curr - e_base

        return torch.cat([e_curr, e_base, e_diff], dim=-1)  # [B, 3*embed_dim]

    @property
    def output_dim(self) -> int:
        return 3 * self.embed_dim


# ── Standalone embedding generation ─────────────────────────────────────────

def generate_m3d_lamed_embeddings(
    dataset,
    batch_size: int = 4,
    device: str = "cpu",
    embed_dim: int = 512,
):
    """Generate longitudinal MRI embeddings using M3D-LaMed's vision backbone.

    No concept pretraining step needed — the 3D encoder is already
    pretrained on diverse medical imaging data.

    Args:
        dataset:    LumiereDataset container with .data dict of split datasets
        batch_size: images per forward pass (reduce if OOM)
        device:     'cuda' or 'cpu'
        embed_dim:  projection head output dimension per stream
    Returns:
        dataset with .X set on each split
    """
    encoder = M3DLamedTemporalEncoder(embed_dim=embed_dim).to(device)
    encoder.eval()
    print(f"M3DLamedTemporalEncoder ready  (output_dim={encoder.output_dim})")

    def _collate(batch):
        result = {
            "x":          torch.stack([b["x"]          for b in batch]),
            "x_baseline": torch.stack([b["x_baseline"] for b in batch]),
            "c":          torch.stack([b["c"]          for b in batch]),
            "y":          torch.stack([b["y"]          for b in batch]),
            "graph":      batch[0].get("graph", {}),
        }
        return result

    for split_name, data in dataset.data.items():
        print(f"  Encoding M3D split '{split_name}' ({len(data)} samples) ...")
        loader = DataLoader(
            data, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_collate,
        )
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"M3D-{split_name}"):
                x_curr = batch["x"].to(device)
                x_base = batch["x_baseline"].to(device)
                emb = encoder(x_curr, x_base)
                embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        data.X = embeddings
        dataset.data[split_name] = data
        print(f"    → {split_name}: X shape = {data.X.shape}")

    return dataset
