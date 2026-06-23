"""
Sliding Window Dataset for FSQ motion tokenizer training.

Direct copy of vqvae_dataloader_v3.py — data format is identical.

36-dim features:
- 0-2: anchor position (x, y, z) - raw world position
- 3-6: anchor quaternion (w, x, y, z) - raw world quaternion
- 7-35: joint positions (29 dims) - raw joint positions

NOTE: Delta encoding (root-pos differencing) is NOT done here;
      it is handled inside the model's forward pass so that
      the loss can be computed on both absolute and delta representations.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
import json
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from utilities.paths import data as data_path


def _to_float32_numpy(arr: np.ndarray) -> np.ndarray:
    """Convert numpy array to float32 (copy only if needed)."""
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


class SlidingWindowDataset(Dataset):
    """Dataset with sliding windows for FSQ training (lazy loading)."""

    def __init__(
        self,
        data_folder: str,
        window_size: int = 64,
        stride: int = 1,
        min_sequence_length: int = 64,
        device: str = "cpu",
        cache_size_files: int = 0,
        metadata_cache_file: Optional[str] = None,
        rebuild_metadata_cache: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride
        self.min_sequence_length = min_sequence_length
        self.device = device

        self.file_paths: List[str] = []
        self.file_frames: List[int] = []
        self.windows_info: List[Tuple[int, int]] = []  # (file_idx, start_frame)

        self.cache_size_files = int(cache_size_files)
        self._file_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

        self.metadata_cache_file = metadata_cache_file or os.path.join(
            data_folder, ".texedo_npz_index_v3.json"
        )
        self.rebuild_metadata_cache = rebuild_metadata_cache

        self._scan_folder(data_folder)
        self._create_sliding_windows()

        print(f"Dataset created with {len(self.windows_info)} windows")
        print(f"Window size: {self.window_size}, Stride: {self.stride}")
        print(f"Lazy loading enabled: supports num_workers > 0")

    # ------------------------------------------------------------------
    # Folder scanning with metadata cache
    # ------------------------------------------------------------------
    def _scan_folder(self, data_folder: str):
        npz_files = sorted(glob.glob(os.path.join(data_folder, "*.npz")))
        print(f"Found {len(npz_files)} npz files in {data_folder}")

        if not self.rebuild_metadata_cache:
            if self._try_load_metadata_cache(npz_files):
                return

        for motion_file in npz_files:
            try:
                with np.load(motion_file) as data:
                    total_frames = data["joint_pos"].shape[0]
                if total_frames >= self.min_sequence_length:
                    self.file_paths.append(motion_file)
                    self.file_frames.append(total_frames)
            except Exception as e:
                print(f"Skipping {os.path.basename(motion_file)}: {e}")

        self._write_metadata_cache()

    def _try_load_metadata_cache(self, npz_files: List[str]) -> bool:
        cache_path = self.metadata_cache_file
        if not cache_path or not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            entries: List[Dict] = payload.get("entries", [])
            cached_paths = [e.get("path") for e in entries]
            if cached_paths != npz_files:
                return False

            for e in entries:
                st = os.stat(e["path"])
                if int(e.get("size", -1)) != int(st.st_size):
                    return False
                if int(e.get("mtime_ns", -1)) != int(
                    getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                ):
                    return False

            for e in entries:
                total_frames = int(e["frames"])
                if total_frames >= self.min_sequence_length:
                    self.file_paths.append(e["path"])
                    self.file_frames.append(total_frames)

            print(f"Loaded metadata cache: {cache_path}")
            return True
        except Exception:
            return False

    def _write_metadata_cache(self) -> None:
        cache_path = self.metadata_cache_file
        if not cache_path:
            return
        try:
            entries = []
            for path, frames in zip(self.file_paths, self.file_frames):
                st = os.stat(path)
                entries.append({
                    "path": path,
                    "frames": int(frames),
                    "size": int(st.st_size),
                    "mtime_ns": int(
                        getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                    ),
                })
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "entries": entries}, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sliding windows
    # ------------------------------------------------------------------
    def _create_sliding_windows(self):
        self.windows_info = []
        for file_idx, total_frames in enumerate(self.file_frames):
            num_windows = (total_frames - self.window_size) // self.stride + 1
            for i in range(num_windows):
                self.windows_info.append((file_idx, i * self.stride))

    def __len__(self) -> int:
        return len(self.windows_info)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            (window_size, 36) tensor: [root_pos(3), root_quat(4), joint_pos(29)]
        """
        file_idx, start_frame = self.windows_info[idx]
        motion_file = self.file_paths[file_idx]
        return self._load_motion_window(motion_file, start_frame, self.window_size)

    # ------------------------------------------------------------------
    # Data loading with LRU file cache
    # ------------------------------------------------------------------
    def _get_cached_arrays(self, motion_file: str) -> Dict[str, np.ndarray]:
        if self.cache_size_files > 0:
            cached = self._file_cache.get(motion_file)
            if cached is not None:
                self._file_cache.move_to_end(motion_file)
                return cached

        with np.load(motion_file) as data:
            arrays = {
                "joint_pos": _to_float32_numpy(data["joint_pos"]),
                "body_pos_w": _to_float32_numpy(data["body_pos_w"]),
                "body_quat_w": _to_float32_numpy(data["body_quat_w"]),
            }

        if self.cache_size_files > 0:
            self._file_cache[motion_file] = arrays
            self._file_cache.move_to_end(motion_file)
            while len(self._file_cache) > self.cache_size_files:
                self._file_cache.popitem(last=False)

        return arrays

    def _load_motion_window(
        self, motion_file: str, start_frame: int, window_size: int
    ) -> torch.Tensor:
        end_frame = start_frame + window_size
        arrays = self._get_cached_arrays(motion_file)

        joint_pos = torch.from_numpy(arrays["joint_pos"][start_frame:end_frame])
        body_pos_w = torch.from_numpy(arrays["body_pos_w"][start_frame:end_frame])
        body_quat_w = torch.from_numpy(arrays["body_quat_w"][start_frame:end_frame])

        anchor_pos_w = body_pos_w[:, 0, :]   # (T, 3)
        anchor_quat_w = body_quat_w[:, 0, :]  # (T, 4)

        return torch.cat([anchor_pos_w, anchor_quat_w, joint_pos], dim=1)  # (T, 36)


def create_dataloader(
    data_folder: str,
    batch_size: int = 32,
    window_size: int = 100,
    num_workers: int = 0,
    shuffle: bool = True,
    device: str = "cpu",
    **kwargs,
) -> DataLoader:
    """Create DataLoader for motion data."""
    dataset = SlidingWindowDataset(
        data_folder=data_folder,
        window_size=window_size,
        device="cpu",
        **kwargs,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


if __name__ == "__main__":
    # Default points at the prepared dataset layout documented in
    # docs/tasks/CONTRACT.md (populated by tokenizer/encode_motion_tokens.py
    # and the data-prep scripts). Override via TSD_DATA or pass a folder
    # directly to SlidingWindowDataset for ad-hoc testing.
    data_folder = str(data_path("CustomCombined", "new_joint_vecs"))

    dataset = SlidingWindowDataset(
        data_folder=data_folder,
        window_size=100,
        stride=1,
        device="cpu",
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")  # (100, 36)

        loader = create_dataloader(
            data_folder=data_folder,
            batch_size=8,
            window_size=100,
            shuffle=True,
        )
        for batch in loader:
            print(f"Batch shape: {batch.shape}")  # (8, 100, 36)
            print(f"Root position range: [{batch[:,:,:3].min():.4f}, {batch[:,:,:3].max():.4f}]")
            print(f"Root quaternion range: [{batch[:,:,3:7].min():.4f}, {batch[:,:,3:7].max():.4f}]")
            print(f"Joint position range: [{batch[:,:,7:].min():.4f}, {batch[:,:,7:].max():.4f}]")
            break
