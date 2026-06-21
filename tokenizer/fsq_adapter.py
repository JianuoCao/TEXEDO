#!/usr/bin/env python3
"""
FSQ Motion Tokenizer Adapter: Wraps FSQVae for MotionGPT compatibility.

Interface (MotionGPT-compatible):
    encode(features) -> (motion_tokens, lengths)
    decode(motion_tokens) -> features

Codes are integers in range [0, codebook_size-1] where
codebook_size = product of fsq_levels (default 7,776 for [3,3,3,3,3,2,2,2,2,2]).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, List

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fsq_arch import FSQVae


class FSQAdapter(nn.Module):
    """
    Adapter: FSQVae → MotionGPT-compatible interface.

    Single codebook, codes in [0, codebook_size-1].
    """

    def __init__(self, fsqvae: FSQVae):
        super().__init__()
        self.fsqvae = fsqvae
        self.codebook_size = fsqvae.codebook_size

    def encode(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode motion features to discrete FSQ codes.

        Args:
            features: (B, T, 36) raw motion features

        Returns:
            motion_tokens: (B, T') discrete codes, range [0, codebook_size-1]
            lengths: (B,) actual token lengths
        """
        code_idx, _ = self.fsqvae.encode(features)
        lengths = torch.full(
            (features.shape[0],), code_idx.shape[1],
            dtype=torch.long, device=features.device,
        )
        return code_idx, lengths

    def decode(
        self,
        motion_tokens: Tensor,
        initial_root_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode from discrete FSQ codes to motion features.

        Args:
            motion_tokens: (B, T') discrete codes
            initial_root_pos: optional (B, 3) root anchor for frame 0

        Returns:
            features: (B, T, 36) reconstructed motion
        """
        return self.fsqvae.decode(motion_tokens, initial_root_pos=initial_root_pos)

    def forward(self, features: Tensor):
        """Training forward (reconstruction)."""
        return self.fsqvae(features)

    @property
    def device(self):
        return next(self.fsqvae.parameters()).device

    def eval(self):
        self.fsqvae.eval()
        return super().eval()

    def train(self, mode=True):
        self.fsqvae.train(mode)
        return super().train(mode)


def load_fsq_adapter(
    checkpoint_path: str,
    device: str = "cuda",
) -> FSQAdapter:
    """
    Load FSQVae from checkpoint and wrap with adapter.

    Args:
        checkpoint_path: path to .pt checkpoint
        device: target device

    Returns:
        FSQAdapter ready for inference
    """
    print(f"Loading FSQVae checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "config" in ckpt:
        model_config = ckpt["config"]["model"]
    else:
        model_config = {
            "nfeats": 36,
            "fsq_levels": [3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
            "output_emb_width": 512,
            "down_t": 2,
            "stride_t": 2,
            "width": 512,
            "depth": 3,
            "dilation_growth_rate": 3,
            "norm": "BN",
            "activation": "relu",
        }

    fsqvae = FSQVae(**model_config).to(device)

    state_dict = ckpt["model_state_dict"]
    # Handle DDP checkpoints
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    fsqvae.load_state_dict(state_dict)
    fsqvae.eval()

    epoch = ckpt.get("epoch", "unknown")
    fsq_levels = model_config.get("fsq_levels", [3, 3, 3, 3, 3, 2, 2, 2, 2, 2])
    codebook_size = fsqvae.codebook_size
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"FSQ levels: {fsq_levels} → codebook_size={codebook_size}")

    adapter = FSQAdapter(fsqvae)
    return adapter


if __name__ == "__main__":
    """Quick self-test."""
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a random model for testing
    fsqvae = FSQVae(
        nfeats=36,
        fsq_levels=[3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
        norm="BN",
    ).to(device)
    adapter = FSQAdapter(fsqvae)
    adapter.eval()

    # Test data
    B, T = 2, 100
    features = torch.randn(B, T, 36, device=device)

    with torch.no_grad():
        tokens, lengths = adapter.encode(features)

    print(f"Input:  {features.shape}")
    print(f"Tokens: {tokens.shape}, range=[{tokens.min()}, {tokens.max()}]")
    print(f"Lengths: {lengths.tolist()}")
    print(f"Codebook size: {adapter.codebook_size}")

    with torch.no_grad():
        recon = adapter.decode(tokens)

    print(f"Recon:  {recon.shape}")
    mse = torch.mean((features - recon) ** 2).item()
    print(f"MSE (random model): {mse:.6f}")
    print("Adapter self-test passed!")
