"""
Train the text-motion matching (semantic) evaluator for 36-dim custom motion data.

Same evaluator architecture used by MotionGPT for computing FID, R-Precision,
Diversity, and Matching Score metrics.

36-dim motion: root xyz (3) + root quat (4) + joint angles (29).
No foot contacts, so we use ALL 36 dims (no stripping of last 4).

Training pipeline (same as text-to-motion paper):
  Step 1: Compute mean/std of the training data
  Step 2: Train MovementConvEncoder + MovementConvDecoder (decomposition)
  Step 3: Train TextEncoderBiGRUCo + MotionEncoderBiGRUCo (text-motion matching)
          with frozen MovementConvEncoder from step 2

Configuration comes from ``configs/evaluator.yaml`` (OmegaConf), not a hardcoded
python class. See ``verifiers/semantic/README.md`` for the expected directory layout.

Usage:
  python train_evaluator.py --step all       # Run all steps
  python train_evaluator.py --step mean_std  # Only compute mean/std
  python train_evaluator.py --step decomp    # Only train decomp
  python train_evaluator.py --step match     # Only train text-motion match
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import time
import math
import random
import numpy as np
import pickle
import codecs as cs
from collections import OrderedDict
from os.path import join as pjoin
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:  # works both as a package import and as a script run from this directory
    from .models import (  # noqa: E402
        MovementConvEncoder,
        MovementConvDecoder,
        MotionEncoderBiGRUCo,
        TextEncoderBiGRUCo,
    )
except ImportError:
    from models import (  # noqa: E402
        MovementConvEncoder,
        MovementConvDecoder,
        MotionEncoderBiGRUCo,
        TextEncoderBiGRUCo,
    )

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "evaluator.yaml"


# =============================================================================
# Root Position -> Root Velocity Conversion
# =============================================================================
def root_pos_to_vel_np(motion, root_dims=3):
    """Convert root absolute position to root velocity (first-order diff).

    Args:
        motion: [T, D] numpy array, where dims 0:root_dims are root position.
        root_dims: number of root position dimensions (default 3 for xyz).

    Returns:
        [T, D] array with dims 0:root_dims replaced by frame-to-frame velocity.
        First frame velocity is copied from second frame.
    """
    result = motion.copy()
    root_vel = np.zeros_like(motion[:, :root_dims])
    root_vel[1:] = motion[1:, :root_dims] - motion[:-1, :root_dims]
    if len(root_vel) > 1:
        root_vel[0] = root_vel[1]  # first frame copies second frame's velocity
    result[:, :root_dims] = root_vel
    return result


# =============================================================================
# Configuration (loaded from YAML, replaces the old hardcoded Config class)
# =============================================================================
class Config:
    """Flat view over the OmegaConf evaluator config + derived directory layout."""

    def __init__(self, cfg=None):
        cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
        self._cfg = cfg

        # Data paths
        self.data_root = cfg.data.data_root
        self.glove_dir = cfg.data.glove_dir
        self.code_path = cfg.data.code_path
        self.save_root = cfg.save_root

        # Motion params
        self.dim_pose = cfg.motion.dim_pose
        self.foot_contact_dims = cfg.motion.foot_contact_dims
        self.motion_input_dim = cfg.motion.motion_input_dim
        self.root_pos_dims = cfg.motion.root_pos_dims
        self.use_root_vel = cfg.motion.use_root_vel

        # Length params
        self.max_motion_length = cfg.length.max_motion_length
        self.min_motion_len = cfg.length.min_motion_len
        self.max_text_len = cfg.length.max_text_len
        self.unit_length = cfg.length.unit_length

        # Architecture
        self.dim_movement_enc_hidden = cfg.arch.dim_movement_enc_hidden
        self.dim_movement_latent = cfg.arch.dim_movement_latent
        self.dim_text_hidden = cfg.arch.dim_text_hidden
        self.dim_motion_hidden = cfg.arch.dim_motion_hidden
        self.dim_coemb_hidden = cfg.arch.dim_coemb_hidden
        self.dim_word = cfg.arch.dim_word
        self.dim_pos_ohot = cfg.arch.dim_pos_ohot

        # Decomp training
        self.decomp_lr = cfg.decomp.lr
        self.decomp_epochs = cfg.decomp.epochs
        self.decomp_batch_size = cfg.decomp.batch_size
        self.decomp_window = cfg.decomp.window
        self.lambda_sparsity = cfg.decomp.lambda_sparsity
        self.lambda_smooth = cfg.decomp.lambda_smooth

        # Match training
        self.match_lr = cfg.match.lr
        self.match_epochs = cfg.match.epochs
        self.match_batch_size = cfg.match.batch_size
        self.negative_margin = cfg.match.negative_margin
        self.warmup_iters = cfg.match.warmup_iters
        self.match_lr_min = cfg.match.lr_min

        # General
        self.gpu_id = cfg.general.gpu_id
        self.log_every = cfg.general.log_every
        self.save_every_e = cfg.general.save_every_e
        self.eval_every_e = cfg.general.eval_every_e
        self.seed = cfg.general.seed

        # W&B
        self.use_wandb = cfg.wandb.use_wandb
        self.wandb_project = cfg.wandb.project
        self.wandb_run_name = cfg.wandb.run_name

        self.refresh_paths()

    def refresh_paths(self):
        self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
        self.text_dir = pjoin(self.data_root, "texts")

        # Match MotionGPT expected structure:
        # {save_root}/{dataname}/t2m/{subdir}/model/
        base = pjoin(self.save_root, "custom36", "t2m")

        self.decomp_model_dir = pjoin(base, "Decomp_SP001_SM001_H512", "model")
        self.decomp_eval_dir = pjoin(base, "Decomp_SP001_SM001_H512", "eval")
        self.decomp_log_dir = pjoin(base, "Decomp_SP001_SM001_H512", "log")

        self.match_model_dir = pjoin(base, "text_mot_match", "model")
        self.match_eval_dir = pjoin(base, "text_mot_match", "eval")
        self.match_log_dir = pjoin(base, "text_mot_match", "log")

        # Meta dir for mean/std (also matches MotionGPT's MEAN_STD_PATH structure)
        self.meta_dir = pjoin(base, "Comp_v6_KLD01", "meta")


def validate_paths(cfg):
    """Validate the CustomCombined evaluator inputs.

    The evaluator consumes continuous 36-dim motions from new_joint_vecs and text
    files from texts. The glove vocab is required to build word embeddings.
    """
    required_dirs = {
        "data_root": cfg.data_root,
        "motion_dir": cfg.motion_dir,
        "text_dir": cfg.text_dir,
        "glove_dir": cfg.glove_dir,
    }
    for label, path in required_dirs.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    for split in ["train.txt", "val.txt"]:
        path = pjoin(cfg.data_root, split)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"split file not found: {path}")

    for glove_file in ["our_vab_data.npy", "our_vab_words.pkl", "our_vab_idx.pkl"]:
        path = pjoin(cfg.glove_dir, glove_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"GloVe vocabulary file not found: {path}")


def write_run_metadata(cfg):
    os.makedirs(cfg.meta_dir, exist_ok=True)
    metadata = {
        "data_root": cfg.data_root,
        "motion_dir": cfg.motion_dir,
        "text_dir": cfg.text_dir,
        "glove_dir": cfg.glove_dir,
        "code_path": cfg.code_path,
        "save_root": cfg.save_root,
        "motion_input_dim": cfg.motion_input_dim,
        "use_root_vel": cfg.use_root_vel,
        "root_pos_dims": cfg.root_pos_dims,
        "max_text_len": cfg.max_text_len,
    }
    with open(pjoin(cfg.meta_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Word Vectorizer
# =============================================================================
POS_enumerator = {
    "VERB": 0, "NOUN": 1, "DET": 2, "ADP": 3, "NUM": 4,
    "AUX": 5, "PRON": 6, "ADJ": 7, "ADV": 8,
    "Loc_VIP": 9, "Body_VIP": 10, "Obj_VIP": 11,
    "Act_VIP": 12, "Desc_VIP": 13, "OTHER": 14,
}

Loc_list = ("left", "right", "clockwise", "counterclockwise", "anticlockwise",
            "forward", "back", "backward", "backwards", "up", "down", "straight", "curve",
            "ahead", "sideways")
Body_list = ("arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg",
             "waist", "eye", "knee", "shoulder", "thigh",
             "head", "hip", "elbow", "wrist", "fist")
Obj_List = ("stair", "dumbbell", "chair", "window", "floor", "car", "ball",
            "handrail", "baseball", "basketball")
Act_list = ("walk", "run", "swing", "pick", "bring", "kick", "put", "squat",
            "throw", "hop", "dance", "jump", "turn", "stumble", "dance", "stop",
            "sit", "lift", "lower", "raise", "wash", "stand", "kneel", "stroll",
            "rub", "bend", "balance", "flap", "jog", "shuffle", "lean", "rotate",
            "spin", "spread", "climb",
            "crawl", "sneak", "creep", "jab", "punch", "box", "crouch", "duck",
            "lunge", "strafe", "dodge", "parry", "block", "strike", "thrust",
            "step", "pivot", "wade", "tiptoe", "stalk",
            "veer", "advance", "shamble", "scurry", "stride", "tread", "limp",
            "strut", "flee", "lurch", "hobble", "sway", "march", "stagger",
            "sidestep", "clap", "wave", "grab", "hook", "carry", "point", "leave", "gesture", "take", "hold", "push")
Desc_list = ("slowly", "carefully", "fast", "careful", "slow", "quickly", "happy",
             "angry", "sad", "happily", "angrily", "sadly",
             "steadily", "powerfully", "cautiously", "covertly", "aggressively",
             "stealthily", "silently", "fearfully", "joyfully", "cheerfully",
             "deliberately", "rapidly", "frantically", "briskly", "energetically")

VIP_dict = {
    "Loc_VIP": Loc_list, "Body_VIP": Body_list,
    "Obj_VIP": Obj_List, "Act_VIP": Act_list, "Desc_VIP": Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        words = pickle.load(open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb"))
        word2idx = pickle.load(open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb"))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator["OTHER"]] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        # HumanML3D evaluator tokens are "word/POS". Some custom text files can
        # contain plain normalized words, so fall back to OTHER instead of crashing.
        if "/" in item:
            word, pos = item.rsplit("/", 1)
            if not word or not pos:
                word, pos = item.replace("/", ""), "OTHER"
        else:
            word, pos = item, "OTHER"
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._get_pos_ohot("OTHER")
        return word_vec, pos_vec


# =============================================================================
# Losses
# =============================================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=3.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


class AllPairsContrastiveLoss(nn.Module):
    """All-pairs contrastive loss.

    Instead of picking ONE random negative per sample, compute contrastive loss
    between every (text_i, motion_j) pair in the batch:
      - (i == j) -> positive pair: pull together
      - (i != j) -> negative pair: push apart beyond margin

    This gives B*(B-1) negatives per batch instead of just B, producing much
    stabler gradients and richer negative mining.
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, text_emb, motion_emb):
        """
        Args:
            text_emb:   [B, D] text embeddings
            motion_emb: [B, D] motion embeddings
        Returns:
            loss, loss_pos, loss_neg (scalars)
        """
        B = text_emb.shape[0]
        # [B, B] pairwise distance matrix: dist[i,j] = ||text_i - motion_j||
        dist_mat = torch.cdist(text_emb, motion_emb, p=2)  # [B, B]

        # Positive pairs: diagonal (i == j)
        pos_dist = torch.diag(dist_mat)                       # [B]
        loss_pos = torch.mean(pos_dist ** 2)

        # Negative pairs: off-diagonal (i != j)
        mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)  # [B, B]
        neg_dist = dist_mat[mask]                              # [B*(B-1)]
        loss_neg = torch.mean(
            torch.clamp(self.margin - neg_dist, min=0.0) ** 2
        )

        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg


# =============================================================================
# Datasets
# =============================================================================
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class DecompDataset(Dataset):
    """Dataset for decomposition training (full-length motions, cropped to fixed window).

    The MovementConvEncoder downsamples by 4x temporally, so we need inputs
    of at least 8+ frames to get >=2 latent frames (needed for smooth loss).
    We use a fixed window of decomp_window frames, cropped randomly from each motion.
    """

    def __init__(self, cfg, mean, std, split_file):
        self.cfg = cfg
        self.mean = mean
        self.std = std
        # Window size for decomp: must be multiple of unit_length and large enough
        # that after 4x downsample we get enough frames for smooth loss
        self.window_size = getattr(cfg, "decomp_window", 64)  # 64 frames -> 16 latent frames

        self.data_dict = {}
        self.name_list = []

        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    id_list.append(line)

        min_len = max(cfg.unit_length * 2, 8)  # need at least 8 frames

        for name in tqdm(id_list, desc="Loading decomp data"):
            motion_path = pjoin(cfg.motion_dir, name + ".npy")
            if not os.path.exists(motion_path):
                continue
            try:
                motion = np.load(motion_path)
                if len(motion) < min_len:
                    continue
                # Convert root pos -> vel on full motion BEFORE cropping
                if cfg.use_root_vel:
                    motion = root_pos_to_vel_np(motion, cfg.root_pos_dims)
                self.data_dict[name] = motion
                self.name_list.append(name)
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue

        print(f"Decomp dataset: {len(self.name_list)} motions, window_size={self.window_size}"
              f", use_root_vel={cfg.use_root_vel}")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        motion = self.data_dict[self.name_list[idx]]

        # Crop to window_size (or use full motion if shorter)
        win = self.window_size
        if len(motion) >= win:
            start = random.randint(0, len(motion) - win)
            motion = motion[start:start + win]
        else:
            # Truncate to multiple of unit_length
            trunc = (len(motion) // self.cfg.unit_length) * self.cfg.unit_length
            if trunc < self.cfg.unit_length * 2:
                trunc = self.cfg.unit_length * 2
            motion = motion[:trunc]
            # Pad to window_size
            pad = np.zeros((win - len(motion), motion.shape[1]), dtype=motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)

        # Z-normalize
        motion = (motion - self.mean) / self.std
        return torch.FloatTensor(motion)


class TextMotionDataset(Dataset):
    """Dataset for text-motion matching training."""

    def __init__(self, cfg, mean, std, split_file, w_vectorizer, expand_flag0=False):
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = cfg.max_motion_length
        self._expand_flag0 = expand_flag0

        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    id_list.append(line)

        self.data_dict = {}
        new_name_list = []
        length_list = []

        for name in tqdm(id_list, desc="Loading text-motion data"):
            motion_path = pjoin(cfg.motion_dir, name + ".npy")
            text_path = pjoin(cfg.text_dir, name + ".txt")

            if not os.path.exists(motion_path) or not os.path.exists(text_path):
                continue

            try:
                motion = np.load(motion_path)
                if len(motion) < cfg.min_motion_len:
                    continue
                # Convert root pos -> vel on full motion BEFORE any sub-cropping
                if cfg.use_root_vel:
                    motion = root_pos_to_vel_np(motion, cfg.root_pos_dims)

                text_data = []
                flag = False
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        if len(line_split) < 4:
                            continue
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                # Our motion data is 50fps (not HumanML3D's 20fps)
                                n_motion = motion[int(f_tag * 50):int(to_tag * 50)]
                                if len(n_motion) < cfg.min_motion_len:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in self.data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                self.data_dict[new_name] = {
                                    "motion": n_motion, "length": len(n_motion),
                                    "text": [text_dict], "motion_id": name,
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except Exception:
                                pass

                if flag:
                    if self._expand_flag0:
                        for cap_idx, td in enumerate(text_data):
                            key = f"cap{cap_idx}_{name}" if cap_idx > 0 else name
                            self.data_dict[key] = {
                                "motion": motion, "length": len(motion),
                                "text": [td], "motion_id": name,
                            }
                            new_name_list.append(key)
                            length_list.append(len(motion))
                    else:
                        self.data_dict[name] = {
                            "motion": motion, "length": len(motion),
                            "text": text_data, "motion_id": name,
                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue

        if len(new_name_list) == 0:
            raise RuntimeError("No valid data found!")

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.name_list = name_list
        self.length_arr = np.array(length_list)
        self.pointer = 0
        self.max_length = 20 * cfg.unit_length  # default
        self.reset_max_len(self.max_length)

        print(f"Text-motion dataset: {len(self.data_dict)} entries (expand_flag0={expand_flag0}), use_root_vel={cfg.use_root_vel}")

    @property
    def motion_ids(self):
        """Return motion_id for each accessible entry (after pointer filtering), in dataset index order."""
        return [
            self.data_dict[self.name_list[self.pointer + i]]["motion_id"]
            for i in range(len(self))
        ]

    def reset_max_len(self, length):
        # Clamp the max_length to not exceed max_motion_length
        length = min(length, self.max_motion_length)
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.cfg.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.cfg.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.cfg.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop motion to multiple of unit_length, capped at max_motion_length
        if self.cfg.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.cfg.unit_length - 1) * self.cfg.unit_length
        else:
            m_length = (m_length // self.cfg.unit_length) * self.cfg.unit_length

        # Cap at max_motion_length
        m_length = min(m_length, self.max_motion_length)

        if m_length <= 0:
            m_length = self.cfg.unit_length

        idx_start = random.randint(0, max(0, len(motion) - m_length))
        motion = motion[idx_start:idx_start + m_length]

        # Ensure motion length matches
        if len(motion) < m_length:
            m_length = (len(motion) // self.cfg.unit_length) * self.cfg.unit_length
            motion = motion[:m_length]

        # Z-normalize
        motion = (motion - self.mean) / self.std

        # Pad to max_motion_length
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens)


# =============================================================================
# Step 1: Compute Mean/Std
# =============================================================================
def compute_mean_std(cfg):
    print("=" * 60)
    print("Step 1: Computing mean/std of training data")
    print("=" * 60)

    os.makedirs(cfg.meta_dir, exist_ok=True)

    train_split = pjoin(cfg.data_root, "train.txt")
    id_list = []
    with cs.open(train_split, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                id_list.append(line)

    all_motions = []
    for name in tqdm(id_list, desc="Loading motions for mean/std"):
        motion_path = pjoin(cfg.motion_dir, name + ".npy")
        if not os.path.exists(motion_path):
            continue
        try:
            motion = np.load(motion_path)
            if cfg.use_root_vel:
                motion = root_pos_to_vel_np(motion, cfg.root_pos_dims)
            all_motions.append(motion)
        except Exception:
            continue

    all_data = np.concatenate(all_motions, axis=0)
    print(f"Total frames for mean/std: {all_data.shape[0]}, dim: {all_data.shape[1]}")

    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)

    # Avoid division by zero
    std[std < 1e-6] = 1.0

    np.save(pjoin(cfg.meta_dir, "mean.npy"), mean)
    np.save(pjoin(cfg.meta_dir, "std.npy"), std)

    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"Saved to {cfg.meta_dir}")

    return mean, std


# =============================================================================
# Step 2: Train Decomposition (MovementConvEncoder + Decoder)
# =============================================================================
def train_decomp(cfg, mean, std, device, use_wandb=False):
    print("=" * 60)
    print("Step 2: Training Movement Decomposition (Encoder + Decoder)")
    print("=" * 60)

    os.makedirs(cfg.decomp_model_dir, exist_ok=True)
    os.makedirs(cfg.decomp_eval_dir, exist_ok=True)

    # Build models - use full 36 dims (no foot contacts to strip)
    movement_enc = MovementConvEncoder(
        cfg.motion_input_dim, cfg.dim_movement_enc_hidden, cfg.dim_movement_latent
    ).to(device)
    movement_dec = MovementConvDecoder(
        cfg.dim_movement_latent, cfg.dim_movement_enc_hidden, cfg.motion_input_dim
    ).to(device)

    print(f"MovementConvEncoder input_size: {cfg.motion_input_dim}")
    print(f"Encoder params: {sum(p.numel() for p in movement_enc.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in movement_dec.parameters()):,}")

    # Dataset
    train_split = pjoin(cfg.data_root, "train.txt")
    val_split = pjoin(cfg.data_root, "val.txt")

    train_dataset = DecompDataset(cfg, mean, std, train_split)
    val_dataset = DecompDataset(cfg, mean, std, val_split)

    train_loader = DataLoader(train_dataset, batch_size=cfg.decomp_batch_size,
                               shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.decomp_batch_size,
                             shuffle=False, num_workers=4, drop_last=True)

    opt_enc = optim.Adam(movement_enc.parameters(), lr=cfg.decomp_lr)
    opt_dec = optim.Adam(movement_dec.parameters(), lr=cfg.decomp_lr)
    l1_criterion = nn.L1Loss()

    start_time = time.time()
    total_iters = cfg.decomp_epochs * len(train_loader)
    it = 0
    min_val_loss = float("inf")

    if use_wandb:
        wandb.config.update({
            "decomp_epochs": cfg.decomp_epochs,
            "decomp_lr": cfg.decomp_lr,
            "decomp_batch_size": cfg.decomp_batch_size,
            "decomp_window": cfg.decomp_window,
            "lambda_sparsity": cfg.lambda_sparsity,
            "lambda_smooth": cfg.lambda_smooth,
            "dim_movement_enc_hidden": cfg.dim_movement_enc_hidden,
            "dim_movement_latent": cfg.dim_movement_latent,
        }, allow_val_change=True)

    print(f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")
    print(f"Iters per epoch: {len(train_loader)}")

    for epoch in range(cfg.decomp_epochs):
        movement_enc.train()
        movement_dec.train()
        logs = OrderedDict()

        for i, batch_data in enumerate(train_loader):
            motions = batch_data.to(device).float()

            # Forward - use all dims (no foot contact stripping)
            latents = movement_enc(motions)
            recon_motions = movement_dec(latents)

            # Loss
            loss_rec = l1_criterion(recon_motions, motions)
            loss_sparsity = torch.mean(torch.abs(latents))
            loss_smooth = l1_criterion(latents[:, 1:], latents[:, :-1])
            loss = loss_rec + loss_sparsity * cfg.lambda_sparsity + loss_smooth * cfg.lambda_smooth

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss.backward()
            opt_enc.step()
            opt_dec.step()

            for k, v in [("loss", loss), ("rec", loss_rec), ("sparse", loss_sparsity), ("smooth", loss_smooth)]:
                logs[k] = logs.get(k, 0) + v.item()

            it += 1
            if it % cfg.log_every == 0:
                elapsed = time.time() - start_time
                msg = f"[Decomp] Ep {epoch:03d} It {it:06d}/{total_iters} ({it*100//total_iters}%) {elapsed/60:.1f}m"
                avg_logs = {k: v / cfg.log_every for k, v in logs.items()}
                for k, v in avg_logs.items():
                    msg += f" {k}: {v:.4f}"
                print(msg)
                if use_wandb:
                    wandb.log({
                        f"decomp/train/{k}": v for k, v in avg_logs.items()
                    } | {"decomp/iter": it, "decomp/epoch": epoch})
                logs = OrderedDict()

        # Save periodically
        state = {
            "movement_enc": movement_enc.state_dict(),
            "movement_dec": movement_dec.state_dict(),
            "opt_enc": opt_enc.state_dict(),
            "opt_dec": opt_dec.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, pjoin(cfg.decomp_model_dir, "latest.tar"))

        if (epoch + 1) % cfg.save_every_e == 0:
            torch.save(state, pjoin(cfg.decomp_model_dir, f"E{epoch+1:04d}.tar"))

        # Validation
        movement_enc.eval()
        movement_dec.eval()
        val_loss = 0
        val_rec = 0
        with torch.no_grad():
            for batch_data in val_loader:
                motions = batch_data.to(device).float()
                latents = movement_enc(motions)
                recon_motions = movement_dec(latents)
                loss_rec = l1_criterion(recon_motions, motions)
                loss_sparsity = torch.mean(torch.abs(latents))
                loss_smooth = l1_criterion(latents[:, 1:], latents[:, :-1])
                val_loss += (loss_rec + loss_sparsity * cfg.lambda_sparsity + loss_smooth * cfg.lambda_smooth).item()
                val_rec += loss_rec.item()

        n = max(len(val_loader), 1)
        val_loss /= n
        val_rec /= n
        print(f"  Validation Loss: {val_loss:.5f}, Rec Loss: {val_rec:.5f}")

        if use_wandb:
            wandb.log({
                "decomp/val/loss": val_loss,
                "decomp/val/rec": val_rec,
                "decomp/epoch": epoch,
            })

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(state, pjoin(cfg.decomp_model_dir, "finest.tar"))
            print(f"  ** New best model saved (val_loss={val_loss:.5f})")
            if use_wandb:
                wandb.run.summary["decomp/best_val_loss"] = min_val_loss
                wandb.run.summary["decomp/best_epoch"] = epoch

    print(f"Decomp training complete. Best val loss: {min_val_loss:.5f}")
    return movement_enc


# =============================================================================
# Step 3: Train Text-Motion Matching
# =============================================================================
def train_text_motion_match(cfg, mean, std, device, use_wandb=False):
    print("=" * 60)
    print("Step 3: Training Text-Motion Matching Evaluator")
    print("=" * 60)

    os.makedirs(cfg.match_model_dir, exist_ok=True)
    os.makedirs(cfg.match_eval_dir, exist_ok=True)

    # Build models
    movement_enc = MovementConvEncoder(
        cfg.motion_input_dim, cfg.dim_movement_enc_hidden, cfg.dim_movement_latent
    ).to(device)

    text_enc = TextEncoderBiGRUCo(
        word_size=cfg.dim_word, pos_size=cfg.dim_pos_ohot,
        hidden_size=cfg.dim_text_hidden, output_size=cfg.dim_coemb_hidden
    ).to(device)

    motion_enc = MotionEncoderBiGRUCo(
        input_size=cfg.dim_movement_latent,
        hidden_size=cfg.dim_motion_hidden,
        output_size=cfg.dim_coemb_hidden
    ).to(device)

    # Load pretrained decomp movement encoder
    decomp_ckpt = pjoin(cfg.decomp_model_dir, "finest.tar")
    if not os.path.exists(decomp_ckpt):
        decomp_ckpt = pjoin(cfg.decomp_model_dir, "latest.tar")

    print(f"Loading decomp checkpoint: {decomp_ckpt}")
    checkpoint = torch.load(decomp_ckpt, map_location=device)
    movement_enc.load_state_dict(checkpoint["movement_enc"])
    movement_enc.eval()  # Freeze movement encoder

    print(f"TextEncoder params:   {sum(p.numel() for p in text_enc.parameters()):,}")
    print(f"MotionEncoder params: {sum(p.numel() for p in motion_enc.parameters()):,}")
    print(f"MovementEncoder params (frozen): {sum(p.numel() for p in movement_enc.parameters()):,}")

    # Word vectorizer
    w_vectorizer = WordVectorizer(cfg.glove_dir, "our_vab")

    # Datasets
    train_split = pjoin(cfg.data_root, "train.txt")
    val_split = pjoin(cfg.data_root, "val.txt")

    train_dataset = TextMotionDataset(cfg, mean, std, train_split, w_vectorizer)
    val_dataset = TextMotionDataset(cfg, mean, std, val_split, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg.match_batch_size,
                               shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.match_batch_size,
                             shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn)

    # Optimizers - only train text & motion encoders, NOT movement encoder
    opt_text = optim.Adam(text_enc.parameters(), lr=cfg.match_lr)
    opt_motion = optim.Adam(motion_enc.parameters(), lr=cfg.match_lr)

    # All-pairs contrastive loss instead of single-shift.
    # This gives B*(B-1) negative pairs per step instead of just B,
    # dramatically reducing gradient noise and improving convergence.
    allpairs_loss = AllPairsContrastiveLoss(margin=cfg.negative_margin)

    start_time = time.time()
    total_iters = cfg.match_epochs * len(train_loader)
    it = 0
    min_val_loss = float("inf")

    # Cosine annealing LR with linear warmup.
    # Warmup: linearly increase LR from ~0 to match_lr over warmup_iters.
    # Cosine: smoothly decay LR to match_lr_min over remaining training.
    warmup_iters = getattr(cfg, "warmup_iters", 200)
    lr_min = getattr(cfg, "match_lr_min", 1e-6)

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Linear warmup: 0 -> 1 over warmup_iters
            return float(current_iter) / float(max(1, warmup_iters))
        # Cosine decay: 1 -> lr_min/match_lr over remaining iters
        progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(lr_min / cfg.match_lr, cosine_decay)

    scheduler_text = optim.lr_scheduler.LambdaLR(opt_text, lr_lambda)
    scheduler_motion = optim.lr_scheduler.LambdaLR(opt_motion, lr_lambda)

    if use_wandb:
        wandb.config.update({
            "match_epochs": cfg.match_epochs,
            "match_lr": cfg.match_lr,
            "match_lr_min": lr_min,
            "warmup_iters": warmup_iters,
            "match_batch_size": cfg.match_batch_size,
            "negative_margin": cfg.negative_margin,
            "loss_type": "all_pairs_contrastive",
            "dim_text_hidden": cfg.dim_text_hidden,
            "dim_motion_hidden": cfg.dim_motion_hidden,
            "dim_coemb_hidden": cfg.dim_coemb_hidden,
        }, allow_val_change=True)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"Iters per epoch: {len(train_loader)}")

    for epoch in range(cfg.match_epochs):
        text_enc.train()
        motion_enc.train()
        movement_enc.eval()
        logs = OrderedDict()

        for i, batch_data in enumerate(train_loader):
            word_emb, pos_ohot, caption, cap_lens, motions, m_lens, _ = batch_data
            word_emb = word_emb.to(device).float()
            pos_ohot = pos_ohot.to(device).float()
            motions = motions.to(device).float()

            # Sort by motion length (descending)
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            # Movement encoding (frozen) - use ALL 36 dims
            with torch.no_grad():
                movements = movement_enc(motions).detach()
            m_lens_enc = m_lens // cfg.unit_length

            # Motion embedding
            motion_embedding = motion_enc(movements, m_lens_enc)

            # Text embedding
            text_embedding = text_enc(word_emb, pos_ohot, cap_lens)
            text_embedding = text_embedding.clone()[align_idx]

            # All-pairs contrastive loss:
            # For each text_i, compute distance to ALL motions in the batch.
            # Diagonal pairs (i==j) are positives, off-diagonal are negatives.
            loss, loss_pos, loss_neg = allpairs_loss(text_embedding, motion_embedding)

            opt_text.zero_grad()
            opt_motion.zero_grad()
            loss.backward()
            clip_grad_norm_(text_enc.parameters(), 0.5)
            clip_grad_norm_(motion_enc.parameters(), 0.5)
            opt_text.step()
            opt_motion.step()

            # Step LR schedulers after each optimizer step
            scheduler_text.step()
            scheduler_motion.step()

            for k, v in [("loss", loss), ("pos", loss_pos), ("neg", loss_neg)]:
                logs[k] = logs.get(k, 0) + v.item()

            it += 1
            if it % cfg.log_every == 0:
                elapsed = time.time() - start_time
                msg = f"[Match] Ep {epoch:03d} It {it:06d}/{total_iters} ({it*100//total_iters}%) {elapsed/60:.1f}m"
                avg_logs = {k: v / cfg.log_every for k, v in logs.items()}
                for k, v in avg_logs.items():
                    msg += f" {k}: {v:.4f}"
                print(msg)
                if use_wandb:
                    current_lr = scheduler_text.get_last_lr()[0]
                    wandb.log({
                        f"match/train/{k}": v for k, v in avg_logs.items()
                    } | {"match/iter": it, "match/epoch": epoch,
                         "match/lr": current_lr})
                logs = OrderedDict()

        # Save
        state = {
            "text_encoder": text_enc.state_dict(),
            "motion_encoder": motion_enc.state_dict(),
            "movement_encoder": movement_enc.state_dict(),
            "opt_text_encoder": opt_text.state_dict(),
            "opt_motion_encoder": opt_motion.state_dict(),
            "epoch": epoch,
            "iter": it,
        }
        torch.save(state, pjoin(cfg.match_model_dir, "latest.tar"))

        if (epoch + 1) % cfg.save_every_e == 0:
            torch.save(state, pjoin(cfg.match_model_dir, f"E{epoch+1:04d}.tar"))

        # Validation
        text_enc.eval()
        motion_enc.eval()
        val_loss = 0
        val_pos = 0
        val_neg = 0
        with torch.no_grad():
            for batch_data in val_loader:
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, _ = batch_data
                word_emb = word_emb.to(device).float()
                pos_ohot = pos_ohot.to(device).float()
                motions = motions.to(device).float()

                align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
                motions = motions[align_idx]
                m_lens = m_lens[align_idx]

                movements = movement_enc(motions).detach()
                m_lens_enc = m_lens // cfg.unit_length
                motion_embedding = motion_enc(movements, m_lens_enc)

                text_embedding = text_enc(word_emb, pos_ohot, cap_lens)
                text_embedding = text_embedding.clone()[align_idx]

                # All-pairs contrastive loss for validation too
                loss, loss_pos, loss_neg = allpairs_loss(text_embedding, motion_embedding)

                val_loss += loss.item()
                val_pos += loss_pos.item()
                val_neg += loss_neg.item()

        n = max(len(val_loader), 1)
        val_loss /= n
        val_pos /= n
        val_neg /= n
        print(f"  Validation Loss: {val_loss:.5f}, Pos: {val_pos:.5f}, Neg: {val_neg:.5f}")

        if use_wandb:
            wandb.log({
                "match/val/loss": val_loss,
                "match/val/pos": val_pos,
                "match/val/neg": val_neg,
                "match/epoch": epoch,
            })

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(state, pjoin(cfg.match_model_dir, "finest.tar"))
            print(f"  ** New best model saved (val_loss={val_loss:.5f})")
            if use_wandb:
                wandb.run.summary["match/best_val_loss"] = min_val_loss
                wandb.run.summary["match/best_epoch"] = epoch

        if (epoch + 1) % cfg.eval_every_e == 0:
            # Log full distance matrix statistics instead of single-shift
            dist_mat = torch.cdist(text_embedding, motion_embedding, p=2)  # [B, B]
            B = dist_mat.shape[0]
            pos_dist = torch.diag(dist_mat)  # [B] diagonal = positive pairs
            mask = ~torch.eye(B, dtype=torch.bool, device=dist_mat.device)
            neg_dist = dist_mat[mask]         # [B*(B-1)] off-diagonal = negatives
            pos_str = " ".join([f"{pos_dist[i]:.3f}" for i in range(B)])
            neg_mean = neg_dist.mean().item()
            neg_min = neg_dist.min().item()
            save_path = pjoin(cfg.match_eval_dir, f"E{epoch+1:03d}.txt")
            with cs.open(save_path, "w") as f:
                f.write("Positive Pairs Distance (diagonal)\n")
                f.write(pos_str + "\n")
                f.write(f"Negative Pairs Distance: mean={neg_mean:.3f}, min={neg_min:.3f}\n")
            if use_wandb:
                wandb.log({
                    "match/val/pos_dist_mean": pos_dist.mean().item(),
                    "match/val/neg_dist_mean": neg_mean,
                    "match/val/neg_dist_min": neg_min,
                    "match/val/dist_gap": neg_mean - pos_dist.mean().item(),
                    "match/epoch": epoch,
                })

    print(f"Text-Motion Matching training complete. Best val loss: {min_val_loss:.5f}")
    print(f"Final checkpoint: {pjoin(cfg.match_model_dir, 'finest.tar')}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train the semantic (text-motion matching) evaluator")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "mean_std", "decomp", "match"],
                        help="Which step to run")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Path to evaluator.yaml")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id (overrides config)")
    parser.add_argument("--decomp_epochs", type=int, default=None, help="Override decomp epochs")
    parser.add_argument("--match_epochs", type=int, default=None, help="Override match epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (overrides config)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto-generated)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root (default: from config, ${TSD_DATA}/CustomCombined)")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Override save root (default: from config, ${TSD_ASSETS}/verifiers/semantic/...)")
    args = parser.parse_args()

    yaml_cfg = OmegaConf.load(args.config)
    cfg = Config(yaml_cfg)

    if args.data_root:
        cfg.data_root = args.data_root
    if args.save_root:
        cfg.save_root = args.save_root
    cfg.refresh_paths()
    if args.gpu is not None:
        cfg.gpu_id = args.gpu
    if args.decomp_epochs:
        cfg.decomp_epochs = args.decomp_epochs
    if args.match_epochs:
        cfg.match_epochs = args.match_epochs
    if args.batch_size:
        cfg.decomp_batch_size = args.batch_size
        cfg.match_batch_size = args.batch_size

    validate_paths(cfg)
    write_run_metadata(cfg)

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("[WARNING] wandb not installed, skipping W&B logging. Run: pip install wandb")

    if use_wandb:
        run_name = args.wandb_run_name or f"step-{args.step}-{time.strftime('%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project or cfg.wandb_project,
            name=run_name,
            config={
                "step": args.step,
                "data_root": cfg.data_root,
                "save_root": cfg.save_root,
                "dim_pose": cfg.dim_pose,
                "seed": cfg.seed,
                "gpu_id": cfg.gpu_id,
            }
        )
        print(f"[W&B] Run: {wandb.run.url}")

    # Set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure dirs exist
    for d in [cfg.meta_dir, cfg.decomp_model_dir, cfg.decomp_eval_dir,
              cfg.match_model_dir, cfg.match_eval_dir]:
        os.makedirs(d, exist_ok=True)

    if args.step in ["all", "mean_std"]:
        mean, std = compute_mean_std(cfg)
    else:
        mean = np.load(pjoin(cfg.meta_dir, "mean.npy"))
        std = np.load(pjoin(cfg.meta_dir, "std.npy"))

    if args.step in ["all", "decomp"]:
        train_decomp(cfg, mean, std, device, use_wandb=use_wandb)

    if args.step in ["all", "match"]:
        train_text_motion_match(cfg, mean, std, device, use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()

    if args.step == "all":
        print("\n" + "=" * 60)
        print("ALL DONE! Evaluator trained successfully.")
        print("=" * 60)
        print(f"\nCheckpoint: {pjoin(cfg.match_model_dir, 'finest.tar')}")
        print(f"Mean/Std:   {cfg.meta_dir}")


if __name__ == "__main__":
    main()
