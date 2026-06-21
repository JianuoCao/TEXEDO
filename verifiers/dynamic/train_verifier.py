"""
train_verifier.py — Training script for DynamicVerifier.

Joint-loss training (v4+):
  All three head losses are optimized simultaneously from epoch 0, with
  user-configurable weights (defaults: success=1.0, dynamics=0.6,
  progress=0.8). No staged schedule, no rank loss. The hierarchical
  guarantee comes from the Q* reward fusion in model.py, not from a
  pairwise ranking objective.

Usage:
  python -m verifiers.dynamic.train_verifier \
      --train_csv  /path/to/train_labels.csv \
      --eval_csv   /path/to/eval_labels.csv \
      --train_motion_dir /path/to/train_motion_csvs \
      --eval_motion_dir  /path/to/eval_motion_csvs \
      --save_dir   runs/v4 \
      [--w_success 1.0 --w_dynamics 0.6 --w_progress 0.8] \
      [--wandb_project dynamic-verifier --wandb_run v4]
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate

from .dataset import (
    MotionDataset,
    compute_norm_stats,
    load_from_csv,
    load_norm_stats,
)
from .model import DynamicVerifier


def collate_skip_none(batch):
    """Drop None items (corrupted files) and collate the rest."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return default_collate(batch)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_success(
    success_logit: torch.Tensor,
    y_success: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    pw = torch.tensor(pos_weight, dtype=success_logit.dtype, device=success_logit.device)
    return nn.functional.binary_cross_entropy_with_logits(
        success_logit, y_success, pos_weight=pw
    )


def loss_dynamics(
    dynamics_hat: torch.Tensor,
    q_dynamics: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.mse_loss(dynamics_hat, q_dynamics)


def loss_progress(
    progress_hat: torch.Tensor,
    q_progress:   torch.Tensor,
    y_success:    Optional[torch.Tensor] = None,
    fail_only:    bool = True,
) -> torch.Tensor:
    """
    MSE on progress. When `fail_only` is True, restrict the loss to failed
    samples (y_success < 0.5), since for successes progress ≡ 1.0 is a
    deterministic copy of y_success and provides no useful gradient.
    """
    if not fail_only or y_success is None:
        return nn.functional.mse_loss(progress_hat, q_progress)

    mask = (y_success < 0.5).to(progress_hat.dtype)
    n_fail = mask.sum()
    if n_fail < 1:
        # No failed samples in this batch — keep graph but contribute nothing.
        return (progress_hat.sum() * 0.0)
    sq_err = (progress_hat - q_progress) ** 2
    return (sq_err * mask).sum() / n_fail


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def train_epoch(
    model: DynamicVerifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    pos_weight: float,
    weights: Dict[str, float],
    device: torch.device,
    progress_fail_only: bool = True,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Dict[str, float]:
    model.train()
    totals   = defaultdict(float)
    n_steps  = 0
    n_total  = len(loader)
    t_start  = time.time()
    use_amp  = scaler is not None and device.type == "cuda"

    for batch in loader:
        if batch is None:
            continue
        feats        = batch["feats"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        y_success    = batch["y_success"].to(device)
        q_dynamics   = batch["q_dynamics"].to(device)
        q_progress   = batch["q_progress"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(feats, padding_mask)

            l_s = loss_success(out["success_logit"], y_success, pos_weight)
            l_d = loss_dynamics(out["dynamics_hat"], q_dynamics)
            l_p = loss_progress(
                out["progress_hat"], q_progress, y_success,
                fail_only=progress_fail_only,
            )

            loss = (
                weights["success"]  * l_s
                + weights["dynamics"] * l_d
                + weights["progress"] * l_p
            )

        if not torch.isfinite(loss):
            bad = [k for k, v in [("s", l_s), ("d", l_d), ("p", l_p),
                                   ("total", loss)]
                   if not torch.isfinite(v)]
            print(f"  WARNING: non-finite loss at step {n_steps+1} "
                  f"(nan in: {bad}), skipping batch", flush=True)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        totals["loss"]     += loss.item()
        totals["success"]  += l_s.item()
        totals["dynamics"] += l_d.item()
        totals["progress"] += l_p.item()
        n_steps += 1

        if n_steps % log_interval == 0:
            elapsed  = time.time() - t_start
            sec_step = elapsed / n_steps
            eta_sec  = sec_step * (n_total - n_steps)
            eta_str  = f"{eta_sec/60:.1f}min" if eta_sec >= 60 else f"{eta_sec:.0f}s"
            avg_loss = totals["loss"] / n_steps
            print(
                f"    step {n_steps:4d}/{n_total}"
                f"  loss={avg_loss:.4f}"
                f"  s={totals['success']/n_steps:.4f}"
                f"  d={totals['dynamics']/n_steps:.4f}"
                f"  p={totals['progress']/n_steps:.4f}"
                f"  {sec_step:.2f}s/step  ETA {eta_str}",
                flush=True,
            )

    if scheduler is not None and n_steps > 0:
        scheduler.step()

    if n_steps == 0:
        print("  ERROR: all batches skipped (all loss NaN). Check data or norm_stats.", flush=True)
        return {k: float("nan") for k in totals}

    return {k: v / n_steps for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Val step: losses + full metrics
# ---------------------------------------------------------------------------

def _extract_raw_metrics(samples: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Extract raw metrics keyed by motion_key (robust to skipped samples)."""
    return {
        s["motion_key"]: {
            "accel_dist": s["accel_dist"],
            "vel_dist":   s["vel_dist"],
            "progress":   s["progress"],
        }
        for s in samples
    }


@torch.no_grad()
def val_epoch(
    model: DynamicVerifier,
    loader: DataLoader,
    pos_weight: float,
    weights: Dict[str, float],
    device: torch.device,
    progress_fail_only: bool = True,
    raw_metrics: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute val losses (same weights as training) and evaluation metrics.
    raw_metrics: from _extract_raw_metrics(); enables correlation metrics.
    """
    model.eval()

    loss_totals = defaultdict(float)
    n_steps = 0

    all_reward       = []
    all_success_prob = []
    all_y_success    = []
    all_dyn_hat      = []
    all_prg_hat      = []
    all_base_ids     = []
    all_motion_keys  = []

    for batch in loader:
        if batch is None:
            continue
        feats        = batch["feats"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        y_success    = batch["y_success"].to(device)
        q_dynamics   = batch["q_dynamics"].to(device)
        q_progress   = batch["q_progress"].to(device)
        base_ids     = batch["base_id"]
        motion_keys  = batch["motion_key"]

        out = model(feats, padding_mask)

        l_s = loss_success(out["success_logit"], y_success, pos_weight)
        l_d = loss_dynamics(out["dynamics_hat"], q_dynamics)
        l_p = torch.nan_to_num(
            loss_progress(
                out["progress_hat"], q_progress, y_success,
                fail_only=progress_fail_only,
            ),
            nan=0.0,
        )

        loss = (
            weights["success"]  * l_s
            + weights["dynamics"] * l_d
            + weights["progress"] * l_p
        )

        loss_totals["loss"]     += loss.item()
        loss_totals["success"]  += l_s.item()
        loss_totals["dynamics"] += l_d.item()
        loss_totals["progress"] += l_p.item()
        n_steps += 1

        all_reward.extend(out["reward_hat"].cpu().numpy().tolist())
        all_success_prob.extend(out["success_prob"].cpu().numpy().tolist())
        all_y_success.extend(y_success.cpu().numpy().tolist())
        all_dyn_hat.extend(out["dynamics_hat"].cpu().numpy().tolist())
        all_prg_hat.extend(out["progress_hat"].cpu().numpy().tolist())
        all_base_ids.extend(base_ids)
        all_motion_keys.extend(motion_keys)

    logs = {k: v / n_steps for k, v in loss_totals.items()}

    # --- Evaluation metrics ---
    reward       = np.array(all_reward)
    success_prob = np.array(all_success_prob)
    y_success    = np.array(all_y_success)
    dyn_hat      = np.array(all_dyn_hat)
    prg_hat      = np.array(all_prg_hat)
    base_ids_arr = np.array(all_base_ids)

    logs.update(_compute_eval_metrics(
        reward, success_prob, y_success, dyn_hat, prg_hat,
        base_ids_arr, all_motion_keys, raw_metrics,
    ))
    return logs


def _compute_eval_metrics(
    reward: np.ndarray,
    success_prob: np.ndarray,
    y_success: np.ndarray,
    dyn_hat: np.ndarray,
    prg_hat: np.ndarray,
    base_ids: np.ndarray,
    motion_keys: List[str],
    raw_metrics: Optional[Dict],
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    import warnings
    from scipy.stats import spearmanr, ConstantInputWarning

    metrics: Dict[str, float] = {}

    # AUROC / AUPRC
    if len(np.unique(y_success)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_success, reward))
        metrics["auprc"] = float(average_precision_score(y_success, reward))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    # Fail recall at threshold 0.5
    fail_mask = y_success < 0.5
    if fail_mask.sum() > 0:
        metrics["fail_recall"] = float(((success_prob < 0.5)[fail_mask]).mean())
    else:
        metrics["fail_recall"] = float("nan")

    # Pairwise preference accuracy (same-base success > fail)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, bid in enumerate(base_ids):
        groups[bid].append(i)

    correct, total = 0, 0
    bon3_rand, bon3_rew, bon3_oracle = [], [], []
    wb_rhos = []
    rng = np.random.default_rng(0)

    for indices in groups.values():
        idx = np.array(indices)
        s   = y_success[idx] > 0.5
        r   = reward[idx]

        if s.any() and (~s).any():
            for rs in r[s]:
                for rf in r[~s]:
                    correct += int(rs > rf)
                    total   += 1

        # Within-base Spearman
        if len(idx) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                rho, _ = spearmanr(r, y_success[idx])
            if not np.isnan(rho):
                wb_rhos.append(rho)

        # Best-of-3
        if len(idx) >= 3:
            chosen = rng.choice(len(idx), 3, replace=False)
            c      = idx[chosen]
            bon3_rand.append(float(y_success[c[rng.integers(3)]] > 0.5))
            bon3_rew.append(float(y_success[c[np.argmax(r[chosen])]] > 0.5))
            bon3_oracle.append(float(y_success[c].max() > 0.5))

    metrics["pairwise_pref_acc"]    = correct / total if total > 0 else float("nan")
    metrics["within_base_spearman"] = float(np.mean(wb_rhos)) if wb_rhos else float("nan")
    metrics["bon3_random"]  = float(np.mean(bon3_rand))   if bon3_rand   else float("nan")
    metrics["bon3_reward"]  = float(np.mean(bon3_rew))    if bon3_rew    else float("nan")
    metrics["bon3_oracle"]  = float(np.mean(bon3_oracle)) if bon3_oracle else float("nan")

    # Correlation with raw simulator metrics (aligned by motion_key)
    if raw_metrics is not None:
        def _sp(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                return float("nan")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                r, _ = spearmanr(a[mask], b[mask])
            return float(r) if not np.isnan(r) else float("nan")

        # Look up raw values by motion_key — robust to any skipped samples
        accel = np.array([raw_metrics[k]["accel_dist"] for k in motion_keys])
        vel   = np.array([raw_metrics[k]["vel_dist"]   for k in motion_keys])
        prog  = np.array([raw_metrics[k]["progress"]   for k in motion_keys])

        metrics["dyn_vs_accel"]    = _sp(dyn_hat, accel)
        metrics["dyn_vs_vel"]      = _sp(dyn_hat, vel)
        metrics["prg_vs_progress"] = _sp(prg_hat, prog)

    return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    save_dir: Path,
    model: DynamicVerifier,
    optimizer,
    epoch: int,
    val_metrics: Dict,
    tag: str = "best",
) -> None:
    path = save_dir / f"checkpoint_{tag}.pt"
    torch.save(
        {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )
    print(f"  [ckpt] Saved {path}")


def load_checkpoint(
    path: str,
    model: DynamicVerifier,
    optimizer=None,
) -> Tuple[int, Dict]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt.get("val_metrics", {})


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def wandb_log_epoch(
    epoch: int,
    train_logs: Dict[str, float],
    val_logs: Dict[str, float],
    loss_weights: Dict[str, float],
    lr: float,
) -> None:
    import wandb

    log: Dict = {"epoch": epoch, "lr": lr}

    for k, v in train_logs.items():
        log[f"train/{k}"] = v

    loss_keys   = {"loss", "success", "dynamics", "progress"}
    metric_keys = set(val_logs.keys()) - loss_keys
    for k in loss_keys:
        if k in val_logs:
            log[f"val/loss_{k}"] = val_logs[k]
    for k in metric_keys:
        log[f"val/{k}"] = val_logs[k]

    for k, v in loss_weights.items():
        log[f"weights/{k}"] = v

    wandb.log(log, step=epoch)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DynamicVerifier reward model")

    # Data
    p.add_argument("--train_csv",        required=True,
                   help="Path to train label CSV")
    p.add_argument("--eval_csv",         required=True,
                   help="Path to eval label CSV")
    p.add_argument("--train_motion_dir", required=True,
                   help="Directory containing train motion CSV files ({traj_id}.csv)")
    p.add_argument("--eval_motion_dir",  required=True,
                   help="Directory containing eval motion CSV files ({traj_id}.csv)")
    p.add_argument("--save_dir",         required=True)
    p.add_argument("--norm_stats",       default=None,
                   help="Path to norm_stats.npz; auto-computed from train if missing")

    # Model
    p.add_argument("--d_model",  type=int,   default=256)
    p.add_argument("--n_heads",  type=int,   default=4)
    p.add_argument("--d_ff",     type=int,   default=1024)
    p.add_argument("--n_layers", type=int,   default=4)
    p.add_argument("--dropout",  type=float, default=0.2)

    # Training
    p.add_argument("--T_max",                  type=int,   default=1024)
    p.add_argument("--batch_size",             type=int,   default=256)
    p.add_argument("--lr",                     type=float, default=1e-4)
    p.add_argument("--weight_decay",           type=float, default=1e-4)
    p.add_argument("--max_epochs",             type=int,   default=20)
    p.add_argument("--patience",               type=int,   default=5)
    p.add_argument("--early_stop_start_epoch", type=int,   default=3,
                   help="Only start counting patience from this epoch")
    p.add_argument("--num_workers",            type=int,   default=16)
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--resume",                 default=None)
    p.add_argument("--log_interval",           type=int,   default=100,
                   help="Print progress every N steps within an epoch")
    p.add_argument("--amp",         action="store_true",
                   help="Enable automatic mixed precision (AMP) training")
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Subsample train set to at most N examples (stratified by success)")

    # Loss weights (constant from epoch 0)
    p.add_argument("--w_success",  type=float, default=1.0,
                   help="Weight for L_success (BCE w/ pos_weight).")
    p.add_argument("--w_dynamics", type=float, default=0.6,
                   help="Weight for L_dynamics (MSE).")
    p.add_argument("--w_progress", type=float, default=0.8,
                   help="Weight for L_progress (MSE).")
    p.add_argument("--progress_fail_only", action="store_true", default=True,
                   help="Restrict L_progress to failed samples only "
                        "(default: True). Successes have progress ≡ 1.0 "
                        "and provide no useful signal.")
    p.add_argument("--no_progress_fail_only",
                   dest="progress_fail_only", action="store_false",
                   help="Disable the fail-only mask on L_progress.")

    # WandB
    p.add_argument("--wandb_project", default="dynamic-verifier")
    p.add_argument("--wandb_run",     default=None,
                   help="WandB run name; defaults to save_dir basename")
    p.add_argument("--no_wandb",      action="store_true",
                   help="Disable WandB logging")

    return p.parse_args()


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    # --- WandB init ---
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or save_dir.name,
            config=vars(args),
        )

    # --- Load splits ---
    print("Loading train split...")
    t0 = time.time()
    train_samples = load_from_csv(args.train_csv, args.train_motion_dir)
    print(f"  {len(train_samples)} samples  ({time.time()-t0:.1f}s)")

    print("Loading eval split...")
    t0 = time.time()
    eval_samples = load_from_csv(args.eval_csv, args.eval_motion_dir)
    print(f"  {len(eval_samples)} samples  ({time.time()-t0:.1f}s)")

    # Optional stratified subsampling of train set
    if args.max_samples and len(train_samples) > args.max_samples:
        rng_sub = np.random.default_rng(args.seed)
        success_idx = [i for i, s in enumerate(train_samples) if s["is_success"]]
        fail_idx    = [i for i, s in enumerate(train_samples) if not s["is_success"]]
        n_suc = int(args.max_samples * len(success_idx) / len(train_samples))
        n_fal = args.max_samples - n_suc
        chosen = (
            rng_sub.choice(success_idx, min(n_suc, len(success_idx)), replace=False).tolist()
            + rng_sub.choice(fail_idx,  min(n_fal, len(fail_idx)),   replace=False).tolist()
        )
        train_samples = [train_samples[i] for i in chosen]
        print(f"Subsampled train to {len(train_samples)} samples "
              f"(success={n_suc}, fail={n_fal})")

    # Extract raw metrics for correlation analysis (no file I/O; data already in memory)
    eval_raw_metrics = _extract_raw_metrics(eval_samples)

    # --- Norm stats ---
    stats_path = args.norm_stats or str(save_dir / "norm_stats.npz")
    if not Path(stats_path).exists():
        norm_stats = compute_norm_stats(train_samples, stats_path)
    else:
        print(f"Loading norm stats from {stats_path}")
        norm_stats = load_norm_stats(stats_path)

    pos_weight = float(norm_stats["pos_weight"])
    print(f"pos_weight = {pos_weight:.2f}")

    # --- Datasets ---
    train_ds = MotionDataset(train_samples, norm_stats, T_max=args.T_max)
    eval_ds  = MotionDataset(eval_samples,  norm_stats, T_max=args.T_max)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
        collate_fn=collate_skip_none,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        collate_fn=collate_skip_none,
    )

    # --- Model ---
    model = DynamicVerifier(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.1
    )

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    # AMP scaler (None = disabled)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    if scaler is not None:
        print("AMP enabled")

    # --- Training loop ---
    best_val_loss  = float("inf")
    patience_count = 0

    loss_keys = ["loss", "success", "dynamics", "progress"]

    # Constant loss weights (no staged schedule).
    weights = {
        "success":  args.w_success,
        "dynamics": args.w_dynamics,
        "progress": args.w_progress,
    }
    w_str = "  ".join(f"{k}={v:.2f}" for k, v in weights.items())
    print(f"\nLoss weights: [{w_str}]")
    print(f"progress_fail_only = {args.progress_fail_only}")

    for epoch in range(start_epoch, args.max_epochs):
        print(f"\nEpoch {epoch:02d}")

        # Train
        t0         = time.time()
        train_logs = train_epoch(
            model, train_loader, optimizer, scheduler,
            epoch, pos_weight, weights, device,
            progress_fail_only=args.progress_fail_only,
            log_interval=args.log_interval,
            scaler=scaler,
        )
        t_train = time.time() - t0

        # Val (losses + metrics)
        t0       = time.time()
        val_logs = val_epoch(
            model, eval_loader, pos_weight, weights, device,
            progress_fail_only=args.progress_fail_only,
            raw_metrics=eval_raw_metrics,
        )
        t_val = time.time() - t0

        # Print
        print(
            f"  train  "
            + "  ".join(f"{k}={train_logs.get(k, 0.0):.4f}" for k in loss_keys)
            + f"  ({t_train:.1f}s)"
        )
        print(
            f"  val    "
            + "  ".join(f"{k}={val_logs.get(k, 0.0):.4f}" for k in loss_keys)
            + f"  ({t_val:.1f}s)"
        )
        metric_keys = [k for k in val_logs if k not in set(loss_keys)]
        if metric_keys:
            print(
                f"  metrics  "
                + "  ".join(f"{k}={val_logs[k]:.4f}" for k in sorted(metric_keys))
            )

        # WandB
        if use_wandb:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb_log_epoch(epoch, train_logs, val_logs, weights, current_lr)

        # Checkpoint last
        save_checkpoint(save_dir, model, optimizer, epoch, val_logs, tag="last")

        # Check improvement and save best
        improved = val_logs["loss"] < best_val_loss
        if improved:
            best_val_loss = val_logs["loss"]
            save_checkpoint(save_dir, model, optimizer, epoch, val_logs, tag="best")
            print(f"  [best] val_loss={best_val_loss:.4f}")

        # Early stopping — patience only counted from early_stop_start_epoch
        if epoch >= args.early_stop_start_epoch:
            if improved:
                patience_count = 0
            else:
                patience_count += 1
                print(f"  [patience {patience_count}/{args.patience}]")
                if patience_count >= args.patience:
                    print("Early stopping.")
                    break
        else:
            print(f"  [early stop inactive until epoch {args.early_stop_start_epoch}]")

    print(f"\nTraining complete.  Best val_loss = {best_val_loss:.4f}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
