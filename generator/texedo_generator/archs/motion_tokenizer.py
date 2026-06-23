"""
Unified Motion Tokenizer Interface
===================================

Provides a common encode/decode API for the motion tokenizer so the
TEXEDO generator model and LM are agnostic to the underlying tokenizer type.

Concrete implementation:
  - FSQTokenizer : wraps FSQVae with a single-codebook direct-code scheme

Config-based instantiation:
  # configs/fsq/default.yaml
  target: texedo_generator.archs.motion_tokenizer.FSQTokenizer
  params: ...
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Abstract base
# ============================================================================

class MotionTokenizerBase(nn.Module, abc.ABC):
    """Abstract interface that all motion tokenizers must implement."""

    @property
    @abc.abstractmethod
    def codebook_size(self) -> int:
        """Total number of discrete codes (vocabulary size for the LM)."""
        ...

    # ---- Encode / Decode (inference) ------------------------------------

    @abc.abstractmethod
    def encode(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode raw motion features to a single stream of discrete tokens.

        Args:
            features: (B, T, 36) raw motion features.

        Returns:
            motion_tokens: (B, T') integer codes in [0, codebook_size-1].
            lengths:       (B,)   valid token count per sample.
        """
        ...

    @abc.abstractmethod
    def decode(self, motion_tokens: Tensor) -> Tensor:
        """
        Decode a single stream of discrete tokens back to motion features.

        Args:
            motion_tokens: (B, T') integer codes in [0, codebook_size-1].

        Returns:
            features: (B, T, 36) reconstructed motion features.
        """
        ...

    # ---- Forward (training: reconstruction) -----------------------------

    @abc.abstractmethod
    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, object]:
        """
        Training forward pass (encode → quantize → decode).

        Args:
            features: (B, T, 36)

        Returns:
            reconstructed: (B, T, 36)
            commit_loss:   scalar
            perplexity:    scalar or tuple
        """
        ...

    # ---- Checkpoint loading ---------------------------------------------

    @abc.abstractmethod
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load tokenizer weights from a standalone checkpoint file."""
        ...


# ============================================================================
# FSQ Tokenizer Wrapper
# ============================================================================

class FSQTokenizer(MotionTokenizerBase):
    """
    Wraps FSQVae as a TEXEDO generator-compatible tokenizer.

    Single codebook — no interleave/deinterleave.
    codebook_size = product(fsq_levels)  (default 7776).
    """

    def __init__(self, **fsq_kwargs):
        super().__init__()
        from texedo_generator.archs.fsq_arch import FSQVae
        self.fsqvae = FSQVae(**fsq_kwargs)

    # -- properties -------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        return self.fsqvae.codebook_size

    # -- interface --------------------------------------------------------

    def encode(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        code_idx, _ = self.fsqvae.encode(features)          # (B, T')
        lengths = torch.full(
            (features.shape[0],), code_idx.shape[1],
            dtype=torch.long, device=features.device,
        )
        return code_idx, lengths

    def decode(self, motion_tokens: Tensor) -> Tensor:
        return self.fsqvae.decode(motion_tokens)             # (B, T, 36)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, object]:
        return self.fsqvae(features)

    def load_pretrained(self, checkpoint_path: str) -> None:
        print(f"[FSQTokenizer] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Restore config validation (optional)
        if "config" in ckpt:
            ckpt_levels = ckpt["config"].get("model", {}).get("fsq_levels")
            if ckpt_levels and list(ckpt_levels) != list(self.fsqvae.fsq_levels):
                print(
                    f"[FSQTokenizer] WARNING: checkpoint fsq_levels={ckpt_levels} "
                    f"vs model fsq_levels={self.fsqvae.fsq_levels}"
                )

        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip DDP prefix
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.fsqvae.load_state_dict(state_dict, strict=True)

        epoch = ckpt.get("epoch", "unknown")
        print(
            f"[FSQTokenizer] Loaded epoch {epoch}, "
            f"codebook_size={self.codebook_size}"
        )
