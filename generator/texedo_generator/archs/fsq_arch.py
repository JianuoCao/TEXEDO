# FSQ (Finite Scalar Quantization) Motion Tokenizer
#
# Adapted from vqvae_arch_v3.py:
#   - QuantizeEMAReset replaced with FSQ from vector_quantize_pytorch
#   - No commitment loss, no EMA codebook updates, no codebook reset
#   - Identical encoder/decoder architecture
#   - Same 36-dim input: root_pos(3) + root_quat(4) + joint_pos(29)
#   - Same root position delta encoding and quaternion normalization
#
# FSQ reference: Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple" (2023)
#
# Install dependency: pip install vector-quantize-pytorch

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict

from vector_quantize_pytorch import FSQ


# ============================================================================
# Utility functions  (unchanged from vqvae_arch_v3.py)
# ============================================================================

def convert_to_csv_format(data: np.ndarray) -> np.ndarray:
    """
    Convert data to CSV format (36 dims):
    1. Quaternion: w,x,y,z -> x,y,z,w
    2. Joint positions: reorder by NPZ_TO_CSV mapping

    Args:
        data: (seq_len, 36) array in NPZ ordering
              [root_pos(3), quat_wxyz(4), joint_pos(29)]

    Returns:
        (seq_len, 36) array in CSV ordering
              [root_pos(3), quat_xyzw(4), joint_pos_reordered(29)]
    """
    NPZ_TO_CSV = [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
        2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
        12, 16, 20, 22, 24, 26, 28
    ]

    converted_data = np.zeros_like(data)

    # Root position (unchanged)
    converted_data[:, :3] = data[:, :3]

    # Quaternion: w,x,y,z -> x,y,z,w
    converted_data[:, 3] = data[:, 4]   # x
    converted_data[:, 4] = data[:, 5]   # y
    converted_data[:, 5] = data[:, 6]   # z
    converted_data[:, 6] = data[:, 3]   # w

    # Joint positions: reorder
    original_joint_pos = data[:, 7:36]
    for i, csv_idx in enumerate(NPZ_TO_CSV):
        converted_data[:, 7 + i] = original_joint_pos[:, csv_idx]

    return converted_data


def normalize_quaternions(x: Tensor) -> Tensor:
    """
    Apply L2 normalization to quaternions (dims 3-6) and ensure w >= 0.

    Args:
        x: (..., 36) tensor where dims 3:7 are quaternion (w, x, y, z)

    Returns:
        Tensor with normalized quaternions
    """
    x_normalized = x.clone()

    quaternions = x[..., 3:7]  # (*, 4)
    quat_norm = torch.norm(quaternions, p=2, dim=-1, keepdim=True)
    quat_norm = torch.clamp(quat_norm, min=1e-8)
    normalized_quaternions = quaternions / quat_norm

    # Ensure w >= 0 (q and -q represent the same rotation)
    w_component = normalized_quaternions[..., 0:1]
    sign = torch.where(
        w_component < 0,
        torch.tensor(-1.0, device=x.device, dtype=x.dtype),
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
    )
    normalized_quaternions = normalized_quaternions * sign

    x_normalized[..., 3:7] = normalized_quaternions
    return x_normalized


def root_pos_to_delta(x: Tensor) -> Tensor:
    """
    First-order differencing on root position dims 0:3 along time axis.

    delta[0]   = x[0]               (keep absolute first frame)
    delta[t]   = x[t] - x[t-1]     for t >= 1

    Args:
        x: (B, T, F) where F >= 3

    Returns:
        (B, T, F) with dims 0:3 replaced by deltas
    """
    delta = x.clone()
    delta[:, 1:, :3] = x[:, 1:, :3] - x[:, :-1, :3]
    # delta[:, 0, :3] keeps the absolute first-frame value
    return delta


def root_pos_from_delta(delta: Tensor) -> Tensor:
    """
    Reconstruct absolute root positions from first-order deltas via cumsum.

    x[t] = sum(delta[0..t])

    Args:
        delta: (B, T, F) where dims 0:3 are in delta space

    Returns:
        (B, T, F) with dims 0:3 restored to absolute positions
    """
    x = delta.clone()
    x[:, :, :3] = torch.cumsum(delta[:, :, :3], dim=1)
    return x


# ============================================================================
# Building blocks  (unchanged from vqvae_arch_v3.py)
# ============================================================================

class Swish(nn.Module):
    """Swish (SiLU) activation: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """Residual 1-D convolution block with optional normalization."""

    def __init__(self, n_in, n_state, dilation=1, activation='relu', norm=None):
        super().__init__()
        padding = dilation
        self.norm_type = norm

        # Normalization layers
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Activations
        act_map = {"relu": nn.ReLU, "silu": Swish, "gelu": nn.GELU}
        act_cls = act_map.get(activation, nn.ReLU)
        self.activation1 = act_cls()
        self.activation2 = act_cls()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x):
        x_orig = x
        if self.norm_type == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        if self.norm_type == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return x + x_orig


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1,
                 reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [
            ResConv1DBlock(
                n_in, n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Encoder / Decoder  (unchanged from vqvae_arch_v3.py)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, input_emb_width=36, output_emb_width=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                         activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width=36, output_emb_width=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None,
                 upsample_mode: str = 'nearest'):
        super().__init__()
        blocks = []
        if upsample_mode not in {'nearest', 'linear'}:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")
        align_corners = False if upsample_mode == 'linear' else None
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# FSQ Motion Tokenizer
# ============================================================================

class FSQVae(nn.Module):
    """
    FSQ-based motion tokenizer for 36-dim motion data.

    Replaces the EMA vector quantizer in VQVaeV3 with FSQ (Finite Scalar
    Quantization). FSQ eliminates codebook collapse, commitment loss, and
    EMA update machinery while achieving competitive reconstruction quality.

    Input layout (36 dims):
        0-2 : root position (world)  → internally delta-encoded
        3-6 : root quaternion (w, x, y, z)
        7-35: joint positions (29 dims)

    Codebook size = product of fsq_levels.
    Default [3,3,3,3,3,2,2,2,2,2] → 3^5 * 2^5 = 7,776 codes.

    FSQ paper: Mentzer et al., https://arxiv.org/abs/2309.15505
    """

    def __init__(
        self,
        nfeats: int = 36,
        fsq_levels: List[int] = [3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: Optional[str] = "BN",
        activation: str = "relu",
        upsample_mode: str = "nearest",
        normalization_stats_file: Optional[str] = None,
        normalize_root_delta: bool = False,
        normalize_joint_pos: bool = False,
        normalization_epsilon: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.output_emb_width = output_emb_width
        self.fsq_levels = fsq_levels

        # Codebook size = product of all levels
        self.codebook_size = 1
        for lvl in fsq_levels:
            self.codebook_size *= lvl

        self.encoder = Encoder(
            input_emb_width=nfeats,
            output_emb_width=output_emb_width,
            down_t=down_t, stride_t=stride_t,
            width=width, depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation, norm=norm,
        )

        self.decoder = Decoder(
            input_emb_width=nfeats,
            output_emb_width=output_emb_width,
            down_t=down_t, stride_t=stride_t,
            width=width, depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation, norm=norm,
            upsample_mode=upsample_mode,
        )

        # FSQ quantizer: project_in (output_emb_width → len(levels))
        #                quantize per-scalar
        #                project_out (len(levels) → output_emb_width)
        # preserve_symmetry=True is required when any level == 2
        # (ensures symmetric quantization grid around 0)
        needs_symmetry = any(lvl == 2 for lvl in fsq_levels)
        self.quantizer = FSQ(
            levels=fsq_levels,
            dim=output_emb_width,
            preserve_symmetry=needs_symmetry,
        )

        self.normalize_root_delta = normalize_root_delta
        self.normalize_joint_pos = normalize_joint_pos
        self.normalization_stats_file = normalization_stats_file
        self.normalization_epsilon = normalization_epsilon

        feature_mean = torch.zeros(nfeats, dtype=torch.float32)
        feature_std = torch.ones(nfeats, dtype=torch.float32)
        if normalize_root_delta or normalize_joint_pos:
            if not normalization_stats_file:
                raise ValueError(
                    "normalization_stats_file must be set when normalization is enabled"
                )
            stats_path = Path(normalization_stats_file)
            if not stats_path.exists():
                raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")
            stats = np.load(stats_path)
            loaded_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            loaded_std = torch.tensor(stats["std"], dtype=torch.float32)

            if loaded_mean.numel() != nfeats or loaded_std.numel() != nfeats:
                raise ValueError(
                    f"Normalization stats shape mismatch: expected {nfeats}, "
                    f"got mean={loaded_mean.numel()}, std={loaded_std.numel()}"
                )

            if normalize_root_delta:
                feature_mean[:3] = loaded_mean[:3]
                feature_std[:3] = loaded_std[:3]
            if normalize_joint_pos:
                feature_mean[7:36] = loaded_mean[7:36]
                feature_std[7:36] = loaded_std[7:36]

        feature_std = torch.clamp(feature_std, min=normalization_epsilon)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)

    # ------------------------------------------------------------------
    # Perplexity monitoring  (no equivalent in plain FSQ — compute here)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_perplexity(self, indices: Tensor) -> Tensor:
        """Compute codebook perplexity from flat integer indices.

        Uses torch.unique instead of F.one_hot to avoid allocating a
        (n_tokens × codebook_size) tensor every step (~188MB at batch=256).
        Memory cost is O(n_unique_codes) ≈ a few KB.
        """
        flat = indices.flatten().long()
        _, counts = torch.unique(flat, return_counts=True)
        probs = counts.float() / flat.numel()
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        return perplexity

    # ------------------------------------------------------------------
    # Preprocessing helpers  (unchanged)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_delta(features: Tensor) -> Tensor:
        return root_pos_to_delta(features)

    @staticmethod
    def _from_delta(features: Tensor) -> Tensor:
        return root_pos_from_delta(features)

    @staticmethod
    def _normalize_quat(features: Tensor) -> Tensor:
        return normalize_quaternions(features)

    def _normalize_features(self, features: Tensor) -> Tensor:
        return (features - self.feature_mean.view(1, 1, -1)) / self.feature_std.view(1, 1, -1)

    def _denormalize_features(self, features: Tensor) -> Tensor:
        return features * self.feature_std.view(1, 1, -1) + self.feature_mean.view(1, 1, -1)

    def _preprocess(self, x: Tensor) -> Tensor:
        """(B, T, F) -> delta root pos -> (B, F, T)"""
        x = self._to_delta(x)
        x = self._normalize_features(x)
        return x.permute(0, 2, 1)

    def _postprocess(self, x: Tensor) -> Tensor:
        """(B, F, T) -> (B, T, F) -> cumsum root pos -> normalize quat"""
        x = x.permute(0, 2, 1)
        x = self._denormalize_features(x)
        x = self._from_delta(x)
        x = self._normalize_quat(x)
        return x

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Training forward pass.

        Args:
            features: (B, T, 36) raw motion features

        Returns:
            x_out:      (B, T, 36) reconstructed features (absolute root pos, normalized quat)
            commit_loss: scalar 0.0  (FSQ has no commitment loss)
            perplexity:  scalar — codebook usage entropy metric
        """
        x_in = self._preprocess(features)           # (B, 36, T) delta space

        x_enc = self.encoder(x_in)                  # (B, output_emb_width, T')
        x_enc_seq = x_enc.permute(0, 2, 1)          # (B, T', output_emb_width)

        # FSQ: project → bound → round; straight-through gradient built-in
        x_quant_seq, indices = self.quantizer(x_enc_seq)  # (B,T',dim), (B,T')

        x_quant = x_quant_seq.permute(0, 2, 1)      # (B, output_emb_width, T')
        x_dec = self.decoder(x_quant)               # (B, 36, T)
        x_out = self._postprocess(x_dec)            # (B, T, 36) absolute space

        perplexity = self._compute_perplexity(indices)
        commit_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)

        return x_out, commit_loss, perplexity

    # ------------------------------------------------------------------
    # Encode to discrete codes
    # ------------------------------------------------------------------
    def encode(self, features: Tensor) -> Tuple[Tensor, None]:
        """
        Encode motion features to discrete FSQ codes.

        Args:
            features: (B, T, 36)

        Returns:
            code_idx: (B, T') integer indices, range [0, codebook_size-1]
            None (placeholder for TEXEDO generator compat)
        """
        x_in = self._preprocess(features)           # (B, 36, T) delta space
        x_enc = self.encoder(x_in)                  # (B, output_emb_width, T')
        x_enc_seq = x_enc.permute(0, 2, 1)          # (B, T', output_emb_width)
        _, code_idx = self.quantizer(x_enc_seq)     # code_idx: (B, T')
        return code_idx, None

    # ------------------------------------------------------------------
    # Decode from discrete codes
    # ------------------------------------------------------------------
    def decode(self, z: Tensor, initial_root_pos: Optional[Tensor] = None) -> Tensor:
        """
        Decode from FSQ discrete codes to motion features.

        Args:
            z: (B, T') integer codes, range [0, codebook_size-1]
            initial_root_pos: optional (B, 3) real-world root position for frame 0

        Returns:
            x_out: (B, T, 36) reconstructed motion in absolute space
        """
        # FSQ indices_to_codes: (B, T') -> (B, T', output_emb_width)
        x_d_seq = self.quantizer.indices_to_codes(z)  # (B, T', output_emb_width)
        x_d = x_d_seq.permute(0, 2, 1)               # (B, output_emb_width, T')

        x_dec = self.decoder(x_d)                    # (B, 36, T)
        x_out = x_dec.permute(0, 2, 1)               # (B, T, 36)
        x_out = self._denormalize_features(x_out)

        # Recover absolute root position from deltas
        x_out = root_pos_from_delta(x_out)

        # Optionally override frame-0 root position
        if initial_root_pos is not None:
            offset = initial_root_pos.unsqueeze(1) - x_out[:, 0:1, :3]
            x_out[:, :, :3] = x_out[:, :, :3] + offset

        # Normalize quaternions
        x_out = normalize_quaternions(x_out)
        return x_out
