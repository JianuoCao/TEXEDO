#!/usr/bin/env python3
"""Precompute normalization statistics for the FSQ motion tokenizer.

Normalized dims:
- 0:3   root delta position
- 7:36  joint position

Skipped dims:
- 3:7   root quaternion (kept as identity transform)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fsq_arch import root_pos_to_delta
from fsq_dataloader import SlidingWindowDataset


NORMALIZE_DIMS = list(range(0, 3)) + list(range(7, 36))
SKIP_DIMS = list(range(3, 7))


@torch.no_grad()
def calculate_stats(dataset: SlidingWindowDataset, batch_size: int = 64, num_workers: int = 0):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    feature_sum = torch.zeros(36, dtype=torch.float64)
    feature_sq_sum = torch.zeros(36, dtype=torch.float64)
    feature_count = torch.zeros(36, dtype=torch.float64)

    for batch_idx, batch in enumerate(loader):
        batch = batch.float()  # (B, T, 36)
        batch_delta = root_pos_to_delta(batch)

        tracked = torch.zeros_like(batch_delta)
        tracked[:, :, :3] = batch_delta[:, :, :3]
        tracked[:, :, 7:36] = batch[:, :, 7:36]

        flat = tracked.reshape(-1, 36).double()
        feature_sum += flat.sum(dim=0)
        feature_sq_sum += (flat * flat).sum(dim=0)
        feature_count += flat.shape[0]

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1}/{len(loader)} batches")

    mean = torch.zeros(36, dtype=torch.float64)
    std = torch.ones(36, dtype=torch.float64)

    mean[NORMALIZE_DIMS] = feature_sum[NORMALIZE_DIMS] / feature_count[NORMALIZE_DIMS]
    variance = feature_sq_sum[NORMALIZE_DIMS] / feature_count[NORMALIZE_DIMS] - mean[NORMALIZE_DIMS] ** 2
    variance = torch.clamp(variance, min=1e-12)
    std[NORMALIZE_DIMS] = torch.sqrt(variance)

    mean[SKIP_DIMS] = 0.0
    std[SKIP_DIMS] = 1.0

    return mean.float(), std.float(), int(feature_count[0].item())


def main():
    parser = argparse.ArgumentParser(description="Precompute FSQ normalization stats")
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-size-files", type=int, default=0)
    args = parser.parse_args()

    dataset = SlidingWindowDataset(
        data_folder=args.data_folder,
        window_size=args.window_size,
        stride=args.stride,
        device="cpu",
        cache_size_files=args.cache_size_files,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot compute normalization stats")

    print("Computing FSQ normalization statistics")
    print(f"  windows: {len(dataset)}")
    print(f"  window_size: {args.window_size}")
    print(f"  stride: {args.stride}")
    print(f"  normalize dims: {NORMALIZE_DIMS}")

    mean, std, total_frames = calculate_stats(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean.numpy(),
        std=std.numpy(),
        normalize_dims=np.array(NORMALIZE_DIMS, dtype=np.int64),
        skip_dims=np.array(SKIP_DIMS, dtype=np.int64),
        total_frames=total_frames,
        window_size=args.window_size,
        stride=args.stride,
        version="fsq_root_delta_joint_pos_v1",
    )

    print(f"Saved stats to {output_path}")
    print(f"Root delta std range: [{std[:3].min().item():.6f}, {std[:3].max().item():.6f}]")
    print(f"Joint pos std range: [{std[7:36].min().item():.6f}, {std[7:36].max().item():.6f}]")


if __name__ == "__main__":
    main()
