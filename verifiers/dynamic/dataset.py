"""
dataset.py — Data pipeline for Dynamic Verifier reward model.

Handles:
  - Loading train/eval splits from flat label CSV files
  - 36-dim → 94-dim dynamics-aware feature transform (unchanged)
  - Norm stats computation (with p1/p99 winsorization for acc/delta dims)
  - Label construction: y_success, q_dynamics, q_progress
  - ContrastiveBatchSampler for same-base contrastive batching
"""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

# Feature dimensions in the 94-dim output vector
_DELTA_XY_SLICE  = slice(0, 2)       # 2 dims
_ROOT_Z_SLICE    = slice(2, 3)       # 1 dim
_ROOT_QUAT_SLICE = slice(3, 7)       # 4 dims
_JOINT_POS_SLICE = slice(7, 36)      # 29 dims
_JOINT_VEL_SLICE = slice(36, 65)     # 29 dims
_JOINT_ACC_SLICE = slice(65, 94)     # 29 dims
INPUT_DIM = 94

# Dims to winsorize (p1/p99) before computing mean/std
_WINSORIZE_DIMS = list(range(0, 2)) + list(range(65, 94))  # delta_xy + joint_acc


# ---------------------------------------------------------------------------
# Feature transform (unchanged)
# ---------------------------------------------------------------------------

def transform_36_to_94(motion: np.ndarray) -> np.ndarray:
    """
    Transform (T+1, 36) raw motion to (T+1, 94) dynamics-aware features.

    Input layout:
      col 0:2   root_xy (global position)
      col 2:3   root_z  (height)
      col 3:7   root quaternion wxyz
      col 7:36  joint angles (29 dims)

    Output layout:
      0:2   delta_root_xy  (t=0 → zero)
      2:3   root_z
      3:7   root_quaternion
      7:36  joint_angles
      36:65 joint_velocity  (t=0 → zero)
      65:94 joint_accel     (t=0,1 → zero)
    """
    T1  = motion.shape[0]
    out = np.zeros((T1, INPUT_DIM), dtype=np.float32)

    root_xy = motion[:, 0:2]
    joints  = motion[:, 7:36]

    delta_xy = np.zeros_like(root_xy)
    delta_xy[1:] = root_xy[1:] - root_xy[:-1]

    joint_vel = np.zeros_like(joints)
    joint_vel[1:] = joints[1:] - joints[:-1]

    joint_acc = np.zeros_like(joints)
    joint_acc[2:] = joint_vel[2:] - joint_vel[1:-1]

    out[:, _DELTA_XY_SLICE]  = delta_xy
    out[:, _ROOT_Z_SLICE]    = motion[:, 2:3]
    out[:, _ROOT_QUAT_SLICE] = motion[:, 3:7]
    out[:, _JOINT_POS_SLICE] = joints
    out[:, _JOINT_VEL_SLICE] = joint_vel
    out[:, _JOINT_ACC_SLICE] = joint_acc

    return out


# ---------------------------------------------------------------------------
# CSV-based split loading
# ---------------------------------------------------------------------------

def load_from_csv(csv_path: str, motion_dir: str) -> List[Dict]:
    """
    Load samples from a flat label CSV and a directory of motion CSV files.

    Label CSV columns (at minimum):
      traj_id, success, progress, accel_dist, vel_dist
    The `path` column (NPZ pointer) is ignored; motion is read from motion_dir.

    Motion files are named {traj_id}.csv inside motion_dir.

    Returns a list of dicts with keys:
      csv_path, motion_key, base_id, is_success,
      success, progress, accel_dist, vel_dist
    """
    df = pd.read_csv(csv_path)
    motion_dir = Path(motion_dir)

    samples = []
    missing = 0
    for _, row in df.iterrows():
        traj_id  = str(row["traj_id"])

        # Find motion file: prefer .npy, then .csv, then .npz
        motion_path = None
        for ext in (".npy", ".csv", ".npz"):
            candidate = motion_dir / f"{traj_id}{ext}"
            if candidate.exists():
                motion_path = candidate
                break

        if motion_path is None:
            missing += 1
            continue

        # base_id: strip variant suffix (_out or _var)
        base_id = traj_id
        for marker in ("_out", "_var"):
            if marker in traj_id:
                base_id = traj_id.rsplit(marker, 1)[0]
                break

        samples.append({
            "csv_path":   str(motion_path),   # may be .csv, .npy, or .npz
            "motion_key": traj_id,
            "base_id":    base_id,
            "is_success": bool(int(row["success"])),
            "success":    float(row["success"]),
            "progress":   float(row["progress"]),
            "accel_dist": float(row["accel_dist"]),
            "vel_dist":   float(row["vel_dist"]),
            "mpjpe_l":    float(row["mpjpe_l"]) if "mpjpe_l" in row else float("nan"),
        })

    print(f"[load_from_csv] {csv_path}: {len(samples)} samples loaded"
          + (f"  ({missing} motion CSVs missing)" if missing else ""),
          flush=True)
    return samples


# ---------------------------------------------------------------------------
# Norm stats computation
# ---------------------------------------------------------------------------

def compute_norm_stats(
    samples: List[Dict],
    save_path: str,
    frames_per_file: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Compute normalization statistics from train samples and save to .npz.

    Feature stats: mean/std over 94 dims (with p1/p99 winsorization for
    delta_xy and joint_acc before computing mean/std).

    Label stats: p95 percentiles for dynamics labels (accel_dist, vel_dist).
    pos_weight: n_negative / n_positive for BCEWithLogitsLoss.
    """
    rng = np.random.default_rng(seed)

    # --- Label stats from sample dicts (already in memory) ---
    accel_vals, vel_vals = [], []
    n_pos, n_neg = 0, 0

    for s in samples:
        accel_vals.append(s["accel_dist"])
        vel_vals.append(s["vel_dist"])
        if s["is_success"]:
            n_pos += 1
        else:
            n_neg += 1

    # --- Feature stats from motion CSVs (sample frames_per_file frames each) ---
    print(f"[norm_stats] Sampling features from {len(samples)} CSV files...", flush=True)
    feat_chunks = []

    for s in samples:
        motion = MotionDataset._load_motion(s["csv_path"])
        if motion is None:
            continue
        feats  = transform_36_to_94(motion)   # (T+1, 94)
        T1 = feats.shape[0]

        if T1 <= frames_per_file:
            feat_chunks.append(feats)
        else:
            idx = rng.choice(T1, frames_per_file, replace=False)
            feat_chunks.append(feats[idx])

    all_feats = np.concatenate(feat_chunks, axis=0)  # (N, 94)

    # Winsorize delta_xy + joint_acc dims before stats
    for d in _WINSORIZE_DIMS:
        p1  = np.percentile(all_feats[:, d], 1)
        p99 = np.percentile(all_feats[:, d], 99)
        all_feats[:, d] = np.clip(all_feats[:, d], p1, p99)

    feat_mean = all_feats.mean(axis=0).astype(np.float32)
    feat_std  = (all_feats.std(axis=0) + 1e-6).astype(np.float32)

    stats = {
        "mean":       feat_mean,
        "std":        feat_std,
        "accel_p95":  np.float32(np.percentile(accel_vals, 95)),
        "vel_p95":    np.float32(np.percentile(vel_vals, 95)),
        "pos_weight": np.float32(n_neg / max(n_pos, 1)),
    }

    np.savez(save_path, **stats)
    print(f"[norm_stats] Saved to {save_path}")
    print(f"             pos_weight={stats['pos_weight']:.2f}  "
          f"accel_p95={stats['accel_p95']:.4f}  vel_p95={stats['vel_p95']:.4f}")
    return {k: v for k, v in stats.items()}


def load_norm_stats(path: str) -> Dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MotionDataset(Dataset):
    """
    Loads motion CSVs and label values, applies 36→94 transform,
    normalizes features, and returns padded tensors.
    """

    def __init__(
        self,
        samples: List[Dict],
        norm_stats: Dict,
        T_max: int = 512,
        quat_wxyz: bool = False,
    ):
        self.samples   = samples
        self.T_max     = T_max
        self.quat_wxyz = quat_wxyz  # if True, reorder cols 3:7 from wxyz → xyzw

        self.mean = norm_stats["mean"].astype(np.float32)  # (94,)
        self.std  = norm_stats["std"].astype(np.float32)   # (94,)

        self._accel_p95 = float(norm_stats["accel_p95"])
        self._vel_p95   = float(norm_stats["vel_p95"])

    def __len__(self) -> int:
        return len(self.samples)

    # Joint index mapping from NPZ order to CSV order (from visualize_npz.py)
    _NPZ_TO_CSV = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
                   2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
                   12, 16, 20, 22, 24, 26, 28]

    @staticmethod
    def _load_motion(path: str) -> Optional[np.ndarray]:
        """Load motion (T, 36) from .npy / .csv / .npz.
        For .csv, also checks for a sibling .npy (faster).
        For .npz: applies joint reordering (NPZ_TO_CSV) and quat wxyz→xyzw
                  to match the CSV format the model was trained on.
        Returns None if the file is corrupted or unreadable."""
        if path.endswith(".npz"):
            try:
                d = np.load(path)
                root_pos  = d["body_pos_w"][:, 0, :]              # (T, 3)
                root_quat = d["body_quat_w"][:, 0, [1, 2, 3, 0]]  # (T, 4) wxyz→xyzw
                joint_pos = d["joint_pos"][:, MotionDataset._NPZ_TO_CSV]  # (T, 29) reordered
                return np.concatenate(
                    [root_pos, root_quat, joint_pos], axis=1
                ).astype(np.float32)                               # (T, 36)
            except Exception:
                return None

        # For .csv: check sibling .npy first
        if path.endswith(".csv"):
            npy_path = path[:-4] + ".npy"
            if os.path.exists(npy_path):
                try:
                    return np.load(npy_path)
                except Exception:
                    return None
            try:
                return pd.read_csv(path, header=None).values.astype(np.float32)
            except Exception:
                return None

        # .npy directly
        try:
            return np.load(path)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Optional[Dict]:
        s = self.samples[idx]

        # --- Motion features (unchanged from original pipeline) ---
        motion = self._load_motion(s["csv_path"])
        if motion is None:
            return None  # corrupted file — collate_fn will skip this sample

        # Reorder quaternion wxyz → xyzw if needed (e.g. kimodo dataset)
        if self.quat_wxyz:
            motion = motion.copy()
            motion[:, 3:7] = motion[:, [4, 5, 6, 3]]

        feats  = transform_36_to_94(motion)                  # (T+1, 94)

        T1    = min(feats.shape[0], self.T_max)
        feats = feats[:T1]

        feats = (feats - self.mean) / self.std               # normalize
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = np.clip(feats, -10.0, 10.0)

        pad_len      = self.T_max - T1
        padding_mask = np.zeros(self.T_max, dtype=bool)      # False = valid

        if pad_len > 0:
            pad          = np.zeros((pad_len, INPUT_DIM), dtype=np.float32)
            feats        = np.concatenate([feats, pad], axis=0)
            padding_mask[T1:] = True                         # True = ignored

        # --- Labels ---
        y_success = s["success"]

        def _quality(val: float, p95: float) -> float:
            if not np.isfinite(p95) or p95 <= 0:
                return 0.0
            return float(np.clip(1.0 - val / p95, 0.0, 1.0))

        q_accel    = _quality(s["accel_dist"], self._accel_p95)
        q_vel      = _quality(s["vel_dist"],   self._vel_p95)
        q_dynamics = 0.5 * q_accel + 0.5 * q_vel

        # Progress rate is already in [0, 1]; use directly as supervision target.
        q_progress = float(np.clip(s["progress"], 0.0, 1.0))

        return {
            "feats":        torch.from_numpy(feats),                          # (T_max, 94)
            "padding_mask": torch.from_numpy(padding_mask),                   # (T_max,) bool
            "y_success":    torch.tensor(y_success,    dtype=torch.float32),  # scalar
            "q_dynamics":   torch.tensor(q_dynamics,   dtype=torch.float32),  # scalar
            "q_progress":   torch.tensor(q_progress,   dtype=torch.float32),  # scalar
            # Raw metric values — passed through for evaluation
            "accel_dist":   torch.tensor(s["accel_dist"],  dtype=torch.float32),
            "vel_dist":     torch.tensor(s["vel_dist"],    dtype=torch.float32),
            "mpjpe_l":      torch.tensor(s.get("mpjpe_l", float("nan")), dtype=torch.float32),
            "progress_raw": torch.tensor(s["progress"],   dtype=torch.float32),
            "base_id":      s["base_id"],
            "motion_key":   s["motion_key"],
        }


# ---------------------------------------------------------------------------
# Contrastive batch sampler
# ---------------------------------------------------------------------------

class ContrastiveBatchSampler(Sampler):
    """
    Yields batches that mix:
      - n_mixed_groups base_ids which have BOTH success and fail variants
        (all variants of the selected group are included)
      - remaining slots filled with random samples from the full dataset

    This ensures each batch contains same-base contrastive pairs for ranking loss.
    """

    def __init__(
        self,
        samples: List[Dict],
        batch_size: int = 48,
        n_mixed_groups: int = 12,
        seed: int = 42,
    ):
        self.batch_size     = batch_size
        self.n_mixed_groups = n_mixed_groups
        self._seed          = seed

        groups: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)
        for i, s in enumerate(samples):
            groups[s["base_id"]].append((i, s["is_success"]))

        self.mixed_groups: List[List[int]] = []
        self.all_indices: List[int] = list(range(len(samples)))

        for items in groups.values():
            has_success = any(ok for _, ok in items)
            has_fail    = any(not ok for _, ok in items)
            if has_success and has_fail:
                self.mixed_groups.append([i for i, _ in items])

        self._n_batches = len(self.all_indices) // batch_size

    def __iter__(self):
        rng = random.Random(self._seed)

        fill_pool = list(self.all_indices)
        rng.shuffle(fill_pool)
        fill_ptr = 0

        mixed_pool = [list(g) for g in self.mixed_groups]
        rng.shuffle(mixed_pool)
        mixed_queue = list(mixed_pool)

        for _ in range(self._n_batches):
            batch: List[int] = []
            in_batch: set = set()

            for _ in range(self.n_mixed_groups):
                if not mixed_queue:
                    rng.shuffle(mixed_pool)
                    mixed_queue = list(mixed_pool)
                group = mixed_queue.pop(0)
                for idx in group:
                    if idx not in in_batch and len(batch) < self.batch_size:
                        batch.append(idx)
                        in_batch.add(idx)

            fill_needed = self.batch_size - len(batch)
            filled = 0
            while filled < fill_needed:
                if fill_ptr >= len(fill_pool):
                    fill_ptr = 0
                    rng.shuffle(fill_pool)
                idx = fill_pool[fill_ptr]
                fill_ptr += 1
                if idx not in in_batch:
                    batch.append(idx)
                    in_batch.add(idx)
                    filled += 1

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._n_batches
