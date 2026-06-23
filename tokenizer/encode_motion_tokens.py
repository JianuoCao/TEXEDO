#!/usr/bin/env python3
"""
Batch encode motion data to motion tokens using the FSQ motion tokenizer.

Usage:
    python encode_motion_tokens.py \
        --checkpoint ${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt \
        --data-folder ${TSD_DATA}/CustomCombined/new_joint_vecs \
        --output-dir ${TSD_DATA}/CustomCombined/TOKENS_FSQ \
        --device cuda

(``${TSD_ASSETS}`` / ``${TSD_DATA}`` are environment variables resolved by
``utilities.paths``; the CLI defaults below fall back to them automatically.)
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

_repo_root = os.path.dirname(current_dir)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from fsq_adapter import load_fsq_adapter
from utilities.paths import assets as assets_path, data as data_path


def encode_motion_file(adapter, motion_file: str, device: str = "cuda") -> np.ndarray:
    """
    Encode a single motion file to discrete FSQ tokens.

    Args:
        adapter: FSQAdapter
        motion_file: path to npz file
        device: target device

    Returns:
        motion_tokens: (T',) int32 array where T' = original_frames / 4
    """
    data = np.load(motion_file)

    joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
    body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)

    anchor_pos = body_pos_w[:, 0, :]
    anchor_quat = body_quat_w[:, 0, :]

    features = torch.cat([anchor_pos, anchor_quat, joint_pos], dim=1).unsqueeze(0)

    with torch.no_grad():
        tokens, _ = adapter.encode(features)

    return tokens[0].cpu().numpy().astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Batch encode motion to FSQ tokens")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(assets_path("tokenizer", "checkpoint_epoch_95.pt")),
        help="Path to FSQ checkpoint (default: ${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt)",
    )
    parser.add_argument(
        "--data-folder", type=str,
        default=str(data_path("CustomCombined", "new_joint_vecs")),
        help="Folder of .npz motions to encode (default: ${TSD_DATA}/CustomCombined/new_joint_vecs)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_folder, "motion_tokens_fsq")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    adapter = load_fsq_adapter(args.checkpoint, device=args.device)
    adapter.eval()

    npz_files = sorted(glob.glob(os.path.join(args.data_folder, "*.npz")))
    if args.num_samples:
        npz_files = npz_files[: args.num_samples]
    print(f"Encoding {len(npz_files)} files → {args.output_dir}")

    ok, fail = 0, 0
    for f in tqdm(npz_files, desc="Encoding"):
        try:
            tokens = encode_motion_file(adapter, f, args.device)
            name = os.path.splitext(os.path.basename(f))[0]
            np.save(os.path.join(args.output_dir, f"{name}.npy"), tokens)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"\n{os.path.basename(f)}: {e}")

    print(f"\nDone: {ok} success, {fail} failed")
    if ok > 0:
        examples = sorted(glob.glob(os.path.join(args.output_dir, "*.npy")))[:3]
        for ex in examples:
            t = np.load(ex)
            print(f"  {os.path.basename(ex)}: shape={t.shape}, range=[{t.min()}, {t.max()}]")


if __name__ == "__main__":
    main()
