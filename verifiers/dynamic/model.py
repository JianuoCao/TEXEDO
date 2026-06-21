"""
model.py — DynamicVerifier architecture.

Components:
  - InputProjection: semantic split of 94-dim input → 256-dim
  - CausalTransformerEncoder: 4-layer causal Transformer (d_model=256)
  - MeanAttentionPooling: masked mean + attention concat pooling
  - Three heads: success (logit), dynamics (sigmoid), progress (sigmoid)
  - Hierarchical reward fusion (Q* formula):
        r = p_s * (1 + α·p_d) / (1 + α) + (1 - p_s) * β · p_g · p_d
    with α=0.4, β=0.6 (β < 1/(1+α) guarantees any success > any failure).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import INPUT_DIM


# Hierarchical reward fusion constants. β < 1/(1+α) keeps the formula
# hierarchically consistent: any success-prob=1 sample dominates any
# success-prob=0 sample regardless of progress/dynamics values.
ALPHA: float = 0.4
BETA:  float = 0.6


def fuse_reward(
    success_prob: torch.Tensor,
    dynamics_hat: torch.Tensor,
    progress_hat: torch.Tensor,
    alpha: float = ALPHA,
    beta: float  = BETA,
) -> torch.Tensor:
    """Q* hierarchical reward formula."""
    return (
        success_prob * (1.0 + alpha * dynamics_hat) / (1.0 + alpha)
        + (1.0 - success_prob) * beta * progress_hat * dynamics_hat
    )


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """
    Projects 94-dim input into d_model by separately encoding each semantic group,
    then fusing with a Linear + GELU + LayerNorm.

    Groups (same as PLAN.md):
      root_dyn  = delta_xy(2) + root_z(1) + root_quat(4) = 7 dims
      joint_pos = 29 dims
      joint_vel = 29 dims
      joint_acc = 29 dims
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.root_proj      = nn.Linear(7,  128)
        self.joint_pos_proj = nn.Linear(29, 128)
        self.joint_vel_proj = nn.Linear(29, 128)
        self.joint_acc_proj = nn.Linear(29, 128)

        self.fusion = nn.Sequential(
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 94)
        root = self.root_proj(x[..., :7])          # (B, T, 128)
        pos  = self.joint_pos_proj(x[..., 7:36])   # (B, T, 128)
        vel  = self.joint_vel_proj(x[..., 36:65])  # (B, T, 128)
        acc  = self.joint_acc_proj(x[..., 65:94])  # (B, T, 128)
        cat  = torch.cat([root, pos, vel, acc], dim=-1)  # (B, T, 512)
        return self.fusion(cat)                     # (B, T, d_model)


class CausalTransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers with causal (autoregressive) attention.
    Uses Pre-LayerNorm for training stability.
    """

    def __init__(
        self,
        d_model: int   = 256,
        n_heads: int   = 4,
        d_ff: int      = 1024,
        n_layers: int  = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, d_model)
        # padding_mask: (B, T) bool, True = padding (ignored)
        T = x.shape[1]
        # Bool causal mask: True = ignore (future positions).
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        return self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )


class MeanAttentionPooling(nn.Module):
    """
    Combines masked mean pooling and attention pooling, then projects.

    h_mean = mean over valid (non-padded) positions
    h_attn = weighted sum using learned attention scores
    output = Linear(concat(h_mean, h_attn)) + GELU + LayerNorm
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.attn_score = nn.Linear(d_model, 1)
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        H: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # H: (B, T, d_model)
        # padding_mask: (B, T) bool, True = padding

        valid   = (~padding_mask).float()                          # (B, T)
        n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1)     # (B, 1)

        # Masked mean: average over valid positions only
        h_mean = (H * valid.unsqueeze(-1)).sum(dim=1) / n_valid   # (B, d_model)

        # Attention pooling: mask padding with -inf before softmax
        scores = self.attn_score(H).squeeze(-1)                   # (B, T)
        scores = scores.masked_fill(padding_mask, float("-inf"))
        attn   = torch.softmax(scores, dim=-1)                    # (B, T)
        h_attn = (attn.unsqueeze(-1) * H).sum(dim=1)             # (B, d_model)

        return self.proj(torch.cat([h_mean, h_attn], dim=-1))     # (B, d_model)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DynamicVerifier(nn.Module):
    """
    Transformer-based reward model for motion quality estimation.

    Input:
      feats:        (B, T, 94) normalized motion features
      padding_mask: (B, T) bool, True = padding

    Output dict:
      success_logit: (B,) raw logit for BCE loss
      success_prob:  (B,) sigmoid of success_logit
      dynamics_hat:  (B,) dynamics quality ∈ (0,1)
      progress_hat:  (B,) progress quality ∈ (0,1)
      reward_hat:    (B,) fused reward score

    Reward fusion (Q* formula, see module-level docstring).
    """

    def __init__(
        self,
        d_model: int   = 256,
        n_heads: int   = 4,
        d_ff: int      = 1024,
        n_layers: int  = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = InputProjection(d_model)
        self.encoder    = CausalTransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.pooling    = MeanAttentionPooling(d_model)

        self.success_head  = _mlp(d_model, 64, 1)
        self.dynamics_head = _mlp(d_model, 64, 1)
        self.progress_head = _mlp(d_model, 64, 1)

    def forward(
        self,
        feats: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict:
        h      = self.input_proj(feats)                   # (B, T, 256)
        H      = self.encoder(h, padding_mask)            # (B, T, 256)
        H      = torch.nan_to_num(H, nan=0.0)            # guard against attn overflow
        h_pool = self.pooling(H, padding_mask)            # (B, 256)

        success_logit = self.success_head(h_pool).squeeze(-1)    # (B,)
        dynamics_hat  = torch.sigmoid(
            self.dynamics_head(h_pool).squeeze(-1)
        )                                                          # (B,)
        progress_hat  = torch.sigmoid(
            self.progress_head(h_pool).squeeze(-1)
        )                                                          # (B,)

        success_prob = torch.sigmoid(success_logit)

        reward_hat = fuse_reward(success_prob, dynamics_hat, progress_hat)

        return {
            "success_logit": success_logit,
            "success_prob":  success_prob,
            "dynamics_hat":  dynamics_hat,
            "progress_hat":  progress_hat,
            "reward_hat":    reward_hat,
        }
