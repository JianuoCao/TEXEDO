"""
CustomCombined data module for MotionGPT.

Combined dataset: AMASS + TextSeeDo motions (32 655 + 14 971 files).
Data at 50fps, 36-dim features (root xyz 3 + root quat 4 + joint angles 29).
Tokens from FSQ (single-stream, codebook size 7776).

Design choices are identical to CustomDataAll:
  - Training mean=0, std=1 (identity, VQ-VAE/FSQ decodes to raw motion)
  - Eval mean/std from evaluator for renorm4t2m()
  - unit_length=4 (FSQ 4x temporal downsample, no interleave)
"""

import numpy as np
import torch
import os
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from . import BASEDataModule
from .humanml import Text2MotionDatasetCB, Text2MotionDatasetEval
from .utils import humanml3d_collate


def root_pos_to_vel_torch(features, root_dims=3):
    """Convert root absolute position to root velocity (first-order diff)."""
    result = features.clone()
    root_vel = torch.zeros_like(features[:, :, :root_dims])
    root_vel[:, 1:] = features[:, 1:, :root_dims] - features[:, :-1, :root_dims]
    if features.shape[1] > 1:
        root_vel[:, 0] = root_vel[:, 1]
    result[:, :, :root_dims] = root_vel
    return result


class CustomDataModule_Combined(BASEDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        cfg.DATASET.JOINT_TYPE = 'customdata_combined'
        self.name = "customdata_combined"
        self.njoints = 36

        # Dataset root
        data_root = cfg.DATASET.CUSTOM_COMBINED.ROOT
        self.hparams.data_root = data_root
        text_dir_name = getattr(cfg.DATASET.CUSTOM_COMBINED, "TEXT_DIR", "texts")
        self.hparams.text_dir = pjoin(data_root, text_dir_name)
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs')

        # ---------- Normalization ----------
        self.hparams.mean = np.zeros(36)
        self.hparams.std = np.ones(36)

        # Evaluator mean/std for renorm4t2m()
        eval_meta_dir = pjoin(
            'deps', 't2m_custom36_combined', 'custom36', 't2m', 'Comp_v6_KLD01', 'meta')
        try:
            t2m_path = cfg.METRIC.TM2T.t2m_path
            eval_meta_dir = pjoin(t2m_path, 'custom36', 't2m', 'Comp_v6_KLD01', 'meta')
        except (AttributeError, KeyError):
            pass

        eval_mean_path = pjoin(eval_meta_dir, 'mean.npy')
        eval_std_path = pjoin(eval_meta_dir, 'std.npy')
        if os.path.exists(eval_mean_path) and os.path.exists(eval_std_path):
            self.hparams.mean_eval = np.load(eval_mean_path)
            self.hparams.std_eval = np.load(eval_std_path)
            print(f"✓ [CustomCombined] Loaded evaluator mean/std from {eval_meta_dir}")
            print(f"  mean range: [{self.hparams.mean_eval.min():.4f}, {self.hparams.mean_eval.max():.4f}]")
            print(f"  std  range: [{self.hparams.std_eval.min():.4f}, {self.hparams.std_eval.max():.4f}]")
        else:
            print(f"⚠ [CustomCombined] Evaluator mean/std not found at {eval_meta_dir}")
            print(f"  Using identity (renorm4t2m will be a no-op)")
            self.hparams.mean_eval = np.zeros(36)
            self.hparams.std_eval = np.ones(36)

        # ---------- Length / fps parameters ----------
        self.hparams.max_motion_length = cfg.DATASET.CUSTOM_COMBINED.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.CUSTOM_COMBINED.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.CUSTOM_COMBINED.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.CUSTOM_COMBINED.UNIT_LEN
        self.hparams.fps = cfg.DATASET.CUSTOM_COMBINED.FPS  # 50fps

        # ---------- Additional parameters ----------
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        self.hparams.code_path = cfg.DATASET.CODE_PATH
        print(f"✓ [CustomCombined] code_path: {self.hparams.code_path}")
        self.hparams.task_path = cfg.DATASET.TASK_PATH
        self.hparams.std_text = cfg.DATASET.CUSTOM_COMBINED.STD_TEXT
        self.Dataset = Text2MotionDatasetCB
        self.DatasetEval = Text2MotionDatasetEval
        self.nfeats = 36

    def feats2joints(self, features):
        return features

    def joints2feats(self, features):
        return features

    def normalize(self, features):
        return features

    def denormalize(self, features):
        return features

    def renorm4t2m(self, features):
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = root_pos_to_vel_torch(features, root_dims=3)
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
