"""
TM2T Metrics for 36-dim custom motion data (no foot contacts).
This overrides get_motion_embeddings to NOT strip the last 4 dims, 
and uses the custom36 evaluator checkpoint.
"""
from typing import List
import os
import torch
import numpy as np
from torch import Tensor
from .t2m import TM2TMetrics


class TM2TMetrics36Dim(TM2TMetrics):
    """TM2T metrics adapted for 36-dim motion data without foot contacts."""
    
    def _get_t2m_evaluator(self, cfg):
        """
        Load T2M text encoder and motion encoder for evaluating.
        Uses 'custom36' as dataname instead of 't2m'.
        """
        from mgpt.config import instantiate_from_config
        
        # init module
        self.t2m_textencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_textencoder)
        self.t2m_moveencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_moveencoder)
        self.t2m_motionencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_motionencoder)

        # Load from custom36 evaluator
        dataname = "custom36"
        ckpt_path = os.path.join(
            cfg.METRIC.TM2T.t2m_path, dataname, "t2m/text_mot_match/model/finest.tar")
        
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Evaluator checkpoint not found at {ckpt_path}")
            print("You need to train the evaluator first. Metrics will not work correctly.")
            return

        t2m_checkpoint = torch.load(ckpt_path, map_location="cpu")
        print(f"✓ Load T2M 36-dim pretrained from {ckpt_path}")
        
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def _target_motion_length(self, fallback: int) -> int:
        dataset_key = {
            "customdata_combined": "CUSTOM_COMBINED",
            "customdata_all": "CUSTOM_ALL",
            "customdata_long": "CUSTOM_LONG",
        }.get(self.dataname)

        if dataset_key is None:
            return fallback

        try:
            return int(getattr(self.cfg.DATASET, dataset_key).MAX_MOTION_LEN)
        except (AttributeError, KeyError, TypeError, ValueError):
            return fallback

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        """
        Override: use ALL 36 dims (no foot contact stripping).
        Also fix the double-division bug for UNIT_LEN.

        The 36-dim evaluator was trained with motions padded to a fixed
        max_motion_length. MotionGPT batches are padded only to the batch max,
        and root-position-to-velocity conversion can leave a large invalid
        velocity at the first padded frame. Since the evaluator motion GRU does
        not pack by length, normalize the padding contract here before encoding.
        """
        lengths = torch.as_tensor(lengths, device=feats.device, dtype=torch.long)
        feats = feats.clone()

        valid_lens = torch.clamp(lengths, min=0, max=feats.shape[1])
        frame_ids = torch.arange(feats.shape[1], device=feats.device)[None, :]
        invalid = frame_ids >= valid_lens[:, None]
        feats = feats.masked_fill(invalid[:, :, None], 0.0)

        target_len = self._target_motion_length(feats.shape[1])
        if feats.shape[1] < target_len:
            pad = feats.new_zeros(
                feats.shape[0], target_len - feats.shape[1], feats.shape[2])
            feats = torch.cat([feats, pad], dim=1)
        elif feats.shape[1] > target_len:
            feats = feats[:, :target_len]
            lengths = torch.clamp(lengths, max=target_len)

        # Divide by unit_length ONCE (the original code divides twice).
        unit_len = 4  # standard unit_length
        m_lens = torch.div(lengths, unit_len, rounding_mode="floor")
        m_lens = torch.clamp(m_lens, min=1)
        
        # Use ALL dims - no :-4 stripping since 36-dim has no foot contacts
        mov = self.t2m_moveencoder(feats).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        return torch.flatten(emb, start_dim=1).detach()
