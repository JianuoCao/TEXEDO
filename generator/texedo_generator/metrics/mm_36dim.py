"""
MMMetrics for 36-dim custom motion data (no foot contacts).
Overrides _get_t2m_evaluator to load from custom36 path,
and get_motion_embeddings to NOT strip the last 4 dims.
"""
import os
import torch
import numpy as np
from typing import List
from torch import Tensor
from .mm import MMMetrics


class MMMetrics36Dim(MMMetrics):
    """MMMetrics adapted for 36-dim motion data without foot contacts."""

    def _get_t2m_evaluator(self, cfg):
        """Load from custom36 evaluator path instead of hardcoded 't2m'."""
        from texedo_generator.config import instantiate_from_config

        self.t2m_textencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_textencoder)
        self.t2m_moveencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_moveencoder)
        self.t2m_motionencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_motionencoder)

        dataname = "custom36"
        ckpt_path = os.path.join(
            cfg.METRIC.TM2T.t2m_path, dataname, "t2m/text_mot_match/model/finest.tar")

        if not os.path.exists(ckpt_path):
            print(f"WARNING: MMMetrics evaluator checkpoint not found at {ckpt_path}")
            return

        t2m_checkpoint = torch.load(ckpt_path, map_location="cpu")
        print(f"✓ Load MMMetrics 36-dim evaluator from {ckpt_path}")

        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        """Use ALL 36 dims (no foot contact stripping) and correct UNIT_LEN."""
        m_lens = torch.tensor(lengths)
        unit_len = 4
        m_lens = torch.div(m_lens, unit_len, rounding_mode="floor")

        # Use ALL dims - no :-4 stripping
        mov = self.t2m_moveencoder(feats).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        return torch.flatten(emb, start_dim=1).detach()
