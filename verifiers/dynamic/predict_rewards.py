"""
predict_rewards.py — Run a trained DynamicVerifier checkpoint on a folder of
motion files, save per-sample reward predictions merged with an existing
tracking-metrics CSV.

Usage:
  conda activate TEXEDO
  python -m verifiers.dynamic.predict_rewards \\
    --tracking_csv runs/tracking_predictions.csv \\
    --motion_dir /path/to/motion_npy_dir \\
    --checkpoint  ${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt \\
    --norm_stats  ${TSD_ASSETS}/verifiers/dynamic/norm_stats.npz \\
    --output_csv runs/predictions_with_reward.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from utilities.paths import assets

from .dataset import INPUT_DIM, load_norm_stats, transform_36_to_94
from .model import DynamicVerifier


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return default_collate(batch)


class PredictDataset(Dataset):
    def __init__(self, records, norm_stats, T_max=1024):
        self.records = records  # list of (motion_key, npy_path)
        self.mean = norm_stats["mean"].astype(np.float32)
        self.std = norm_stats["std"].astype(np.float32)
        self.T_max = T_max

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        motion_key, npy_path = self.records[idx]
        try:
            motion = np.load(npy_path).astype(np.float32)
        except Exception:
            return None

        feats = transform_36_to_94(motion)
        T1 = min(feats.shape[0], self.T_max)
        feats = feats[:T1]
        feats = (feats - self.mean) / self.std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = np.clip(feats, -10.0, 10.0)

        pad_len = self.T_max - T1
        padding_mask = np.zeros(self.T_max, dtype=bool)
        if pad_len > 0:
            pad = np.zeros((pad_len, INPUT_DIM), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
            padding_mask[T1:] = True

        return {
            "motion_key": motion_key,
            "feats": torch.from_numpy(feats),
            "padding_mask": torch.from_numpy(padding_mask),
        }


def main():
    p = argparse.ArgumentParser(description="Run DynamicVerifier inference on a folder of motions")
    p.add_argument("--tracking_csv", required=True,
                   help="Existing tracking_predictions.csv with motion_key and metrics")
    p.add_argument("--motion_dir", required=True,
                   help="Directory with {motion_key}.npy files")
    p.add_argument("--checkpoint", default=str(assets("verifiers/dynamic/checkpoint_last.pt")),
                   help="DynamicVerifier checkpoint .pt file "
                        "(default: ${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt)")
    p.add_argument("--norm_stats", default=None,
                   help="norm_stats.npz; defaults to <checkpoint_dir>/norm_stats.npz")
    p.add_argument("--output_csv", required=True,
                   help="Output CSV path with reward_hat column added")
    p.add_argument("--T_max", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    norm_stats_path = args.norm_stats or str(Path(args.checkpoint).parent / "norm_stats.npz")
    norm_stats = load_norm_stats(norm_stats_path)

    df = pd.read_csv(args.tracking_csv)
    motion_dir = Path(args.motion_dir)

    records = []
    missing_keys = []
    for _, row in df.iterrows():
        key = str(row["motion_key"])
        npy = motion_dir / f"{key}.npy"
        if npy.exists():
            records.append((key, str(npy)))
        else:
            missing_keys.append(key)

    if missing_keys:
        print(f"WARNING: {len(missing_keys)} motion files not found (first 5: {missing_keys[:5]})")
    print(f"Loaded {len(records)} records ({len(missing_keys)} missing)")

    dataset = PredictDataset(records, norm_stats, T_max=args.T_max)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        collate_fn=collate_skip_none,
    )

    model = DynamicVerifier(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    motion_keys, reward_hats, success_probs, dynamics_hats, progress_hats = [], [], [], [], []
    n_batches = len(loader)

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Inference")):
            if batch is None:
                continue
            out = model(batch["feats"].to(device), batch["padding_mask"].to(device))
            motion_keys.extend(batch["motion_key"])
            reward_hats.extend(out["reward_hat"].cpu().numpy().tolist())
            success_probs.extend(out["success_prob"].cpu().numpy().tolist())
            dynamics_hats.extend(out["dynamics_hat"].cpu().numpy().tolist())
            progress_hats.extend(out["progress_hat"].cpu().numpy().tolist())

            if (bi + 1) % 200 == 0 or (bi + 1) == n_batches:
                print(f"  [{bi+1}/{n_batches}] {len(reward_hats)} samples processed", flush=True)

    pred_df = pd.DataFrame({
        "motion_key": motion_keys,
        "reward_hat": reward_hats,
        "success_prob": success_probs,
        "dynamics_hat": dynamics_hats,
        "progress_hat": progress_hats,
    })

    merged = df.merge(pred_df, on="motion_key", how="left")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(merged)} rows to {args.output_csv}")
    print(f"reward_hat stats: mean={pred_df['reward_hat'].mean():.4f}  "
          f"min={pred_df['reward_hat'].min():.4f}  max={pred_df['reward_hat'].max():.4f}")


if __name__ == "__main__":
    main()
