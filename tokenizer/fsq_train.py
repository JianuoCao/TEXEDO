#!/usr/bin/env python3
"""Training script for FSQ motion tokenizer (single encoder/decoder with delta root-pos).

Adapted from vqvae_train_v3.py:
- Replaces VQVaeV3 with FSQVae
- Removes commitment loss (FSQ has no codebook collapse or commitment loss)
- All 5 reconstruction losses unchanged
- Same optimizer, scheduler, DDP, AMP, and WandB integration
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import wandb

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fsq_arch import FSQVae, convert_to_csv_format, normalize_quaternions
from fsq_dataloader import SlidingWindowDataset, create_dataloader
from textseedo.paths import data as data_path


# ============================================================================
# Loss
# ============================================================================

class FSQLoss(nn.Module):
    """
    Combined reconstruction loss for FSQ motion tokenizer.

    Identical to VQVaeV3Loss except commitment loss is removed — FSQ
    eliminates codebook collapse by design and needs no auxiliary losses.

    Feature layout (36 dims):
        0-2  : root position   – absolute world coords, scale ~O(1)
        3-6  : root quaternion – unit quaternion, scale O(1)
        7-35 : joint positions – joint offsets, scale ~O(0.1–1)

    Loss components:
        1. Root position reconstruction  (SmoothL1 on absolute)
        2. Root quaternion reconstruction (1 - <q, q̂>²)
        3. Joint position reconstruction (SmoothL1)
        4. Root position velocity (SmoothL1 on Δroot)
        5. Joint position velocity (SmoothL1 on Δjoint)
    """

    def __init__(
        self,
        lambda_root_pos: float = 1.0,
        lambda_root_quat: float = 5.0,
        lambda_joint_pos: float = 1.0,
        lambda_root_vel: float = 0.5,
        lambda_joint_vel: float = 1.0,
        lambda_root_acc: float = 0.0,
        lambda_joint_acc: float = 0.0,
        recons_loss: str = "l1_smooth",
    ):
        super().__init__()
        self.lambda_root_pos = lambda_root_pos
        self.lambda_root_quat = lambda_root_quat
        self.lambda_joint_pos = lambda_joint_pos
        self.lambda_root_vel = lambda_root_vel
        self.lambda_joint_vel = lambda_joint_vel
        self.lambda_root_acc = lambda_root_acc
        self.lambda_joint_acc = lambda_joint_acc

        loss_map = {
            "l1": nn.L1Loss,
            "l2": nn.MSELoss,
            "l1_smooth": nn.SmoothL1Loss,
        }
        self.recons_loss = loss_map[recons_loss]()

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def quaternion_loss(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """Angular quaternion loss: L = 1 - <q, q̂>²."""
        q_pred_n = F.normalize(q_pred, p=2, dim=-1)
        q_target_n = F.normalize(q_target, p=2, dim=-1)
        dot = torch.sum(q_pred_n * q_target_n, dim=-1)
        return torch.mean(1.0 - dot ** 2)

    @staticmethod
    def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """First-order temporal difference (velocity) loss."""
        pred_d = pred[:, 1:, :] - pred[:, :-1, :]
        tgt_d = target[:, 1:, :] - target[:, :-1, :]
        return F.smooth_l1_loss(pred_d, tgt_d)

    @staticmethod
    def acceleration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Second-order temporal difference (acceleration) loss."""
        pred_dd = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
        tgt_dd = target[:, 2:, :] - 2 * target[:, 1:-1, :] + target[:, :-2, :]
        return F.smooth_l1_loss(pred_dd, tgt_dd)

    # ---- forward ---------------------------------------------------------
    def forward(self, x_recon, x_orig, commit_loss, perplexity):
        """
        Args:
            x_recon:     (B, T, 36) reconstructed (absolute root pos, normalized quat)
            x_orig:      (B, T, 36) ground truth
            commit_loss: ignored (always 0.0 from FSQVae)
            perplexity:  scalar from FSQVae — logged only
        """
        if isinstance(perplexity, torch.Tensor) and perplexity.numel() > 1:
            perplexity = perplexity.mean()

        # Slice features
        rp_pred, rp_tgt = x_recon[:, :, :3], x_orig[:, :, :3]
        rq_pred, rq_tgt = x_recon[:, :, 3:7], x_orig[:, :, 3:7]
        jp_pred, jp_tgt = x_recon[:, :, 7:36], x_orig[:, :, 7:36]

        # Reconstruction losses
        loss_root_pos = self.recons_loss(rp_pred, rp_tgt)
        loss_root_quat = self.quaternion_loss(rq_pred, rq_tgt)
        loss_joint_pos = self.recons_loss(jp_pred, jp_tgt)

        # Velocity (temporal) losses
        loss_root_vel = self.velocity_loss(rp_pred, rp_tgt)
        loss_joint_vel = self.velocity_loss(jp_pred, jp_tgt)
        loss_root_acc = self.acceleration_loss(rp_pred, rp_tgt)
        loss_joint_acc = self.acceleration_loss(jp_pred, jp_tgt)

        # Weighted total — no commitment term for FSQ
        total = (
            self.lambda_root_pos * loss_root_pos
            + self.lambda_root_quat * loss_root_quat
            + self.lambda_joint_pos * loss_joint_pos
            + self.lambda_root_vel * loss_root_vel
            + self.lambda_joint_vel * loss_joint_vel
            + self.lambda_root_acc * loss_root_acc
            + self.lambda_joint_acc * loss_joint_acc
        )

        loss_recon = (
            self.lambda_root_pos * loss_root_pos
            + self.lambda_root_quat * loss_root_quat
            + self.lambda_joint_pos * loss_joint_pos
        )
        loss_vel = (
            self.lambda_root_vel * loss_root_vel
            + self.lambda_joint_vel * loss_joint_vel
        )
        loss_acc = (
            self.lambda_root_acc * loss_root_acc
            + self.lambda_joint_acc * loss_joint_acc
        )

        return total, {
            "total_loss": total.item(),
            "loss_recon": loss_recon.item(),
            "loss_velocity": loss_vel.item(),
            "loss_acceleration": loss_acc.item(),
            "loss_root_pos": loss_root_pos.item(),
            "loss_root_quat": loss_root_quat.item(),
            "loss_joint_pos": loss_joint_pos.item(),
            "loss_root_vel": loss_root_vel.item(),
            "loss_joint_vel": loss_joint_vel.item(),
            "loss_root_acc": loss_root_acc.item(),
            "loss_joint_acc": loss_joint_acc.item(),
            "perplexity": perplexity.item() if isinstance(perplexity, torch.Tensor) else float(perplexity),
        }


# ============================================================================
# Trainer
# ============================================================================

class FSQTrainer:
    """Trainer for FSQ motion tokenizer."""

    def __init__(self, config: dict):
        self.config = config

        # ---------- Distributed setup ----------
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
            )
        self.is_main = self.rank == 0

        # ---------- Model ----------
        self.model = FSQVae(**config["model"]).to(self.device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             output_device=self.local_rank, find_unused_parameters=False)
        if self.is_main:
            nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            fsq_levels = config["model"].get("fsq_levels", [3, 3, 3, 3, 3, 2, 2, 2, 2, 2])
            codebook_size = 1
            for lvl in fsq_levels:
                codebook_size *= lvl
            print(f"FSQVae: {nparams:,} trainable parameters")
            print(f"FSQ levels: {fsq_levels} → codebook_size={codebook_size}")

        # ---------- Loss ----------
        self.loss_fn = FSQLoss(**config["loss"])

        # ---------- Optimizer & scheduler ----------
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["optimizer"]["lr"],
            betas=tuple(config["optimizer"]["betas"]),
            weight_decay=config["optimizer"].get("weight_decay", 0.0),
        )
        num_epochs = config["training"]["num_epochs"]
        warmup_epochs = config.get("scheduler", {}).get("warmup_epochs", 0)
        eta_min = config.get("scheduler", {}).get("eta_min", 0.0)
        if warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
                eta_min=eta_min,
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=eta_min,
            )

        # ---------- AMP ----------
        self.use_amp = config["training"].get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # ---------- Data ----------
        data_cfg = config["data"]
        if self.is_distributed:
            train_ds = SlidingWindowDataset(
                data_folder=data_cfg["data_folder"],
                window_size=data_cfg.get("window_size", 100),
                stride=data_cfg.get("stride", 1),
                device="cpu",
                cache_size_files=data_cfg.get("cache_size_files", 0),
            )
            self.train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
            self.train_loader = DataLoader(
                train_ds, batch_size=config["training"]["batch_size"],
                sampler=self.train_sampler,
                num_workers=config["training"]["num_workers"],
                pin_memory=True, drop_last=True,
            )
        else:
            self.train_sampler = None
            self.train_loader = create_dataloader(
                data_folder=data_cfg["data_folder"],
                batch_size=config["training"]["batch_size"],
                window_size=data_cfg.get("window_size", 100),
                num_workers=config["training"]["num_workers"],
                shuffle=True, device="cpu",
                stride=data_cfg.get("stride", 1),
                cache_size_files=data_cfg.get("cache_size_files", 0),
            )

        val_folder = data_cfg.get("val_data_folder") or data_cfg["data_folder"]
        if self.is_distributed:
            val_ds = SlidingWindowDataset(
                data_folder=val_folder,
                window_size=data_cfg.get("window_size", 100),
                stride=data_cfg.get("val_stride", 10),
                device="cpu",
                cache_size_files=data_cfg.get("cache_size_files", 0),
            )
            self.val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
            self.val_loader = DataLoader(
                val_ds, batch_size=config["training"]["batch_size"],
                sampler=self.val_sampler,
                num_workers=config["training"]["num_workers"],
                pin_memory=True, drop_last=False,
            )
        else:
            self.val_sampler = None
            self.val_loader = create_dataloader(
                data_folder=val_folder,
                batch_size=config["training"]["batch_size"],
                window_size=data_cfg.get("window_size", 100),
                num_workers=config["training"]["num_workers"],
                shuffle=False, device="cpu",
                stride=data_cfg.get("val_stride", 10),
                cache_size_files=data_cfg.get("cache_size_files", 0),
            )

        if self.is_main:
            print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # ---------- Dirs ----------
        self.log_dir = Path(config["training"]["log_dir"])
        self.ckpt_dir = Path(config["training"]["checkpoint_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Wandb ----------
        if config.get("use_wandb", True) and self.is_main:
            wcfg = config.get("wandb", {})
            wandb.init(
                project=wcfg.get("project", "fsq-motion"),
                name=wcfg.get("name", f"fsq-{int(time.time())}"),
                id=wcfg.get("id"),
                resume=wcfg.get("resume"),
                config=config,
                tags=wcfg.get("tags", ["fsq", "motion"]),
                notes=wcfg.get("notes", "FSQ motion tokenizer training"),
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Wandb helper
    # ------------------------------------------------------------------
    def _wandb_log(self, data: dict, step: int = None):
        try:
            wandb.log(data, step=step)
        except Exception as e:
            print(f"wandb logging failed ({e}); disabling wandb for the rest of training.")
            self.use_wandb = False

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_epoch(self):
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        self.model.train()
        total_losses: dict = {}
        num_batches = len(self.train_loader)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            motion = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            # Sanitize input
            if torch.isnan(motion).any() or torch.isinf(motion).any():
                motion = torch.nan_to_num(motion, nan=0.0, posinf=10.0, neginf=-10.0)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    recon, commit_loss, perplexity = self.model(motion)
                    loss, loss_dict = self.loss_fn(recon, motion, commit_loss, perplexity)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.scaler.update()
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.scaler.update()
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon, commit_loss, perplexity = self.model(motion)
                loss, loss_dict = self.loss_fn(recon, motion, commit_loss, perplexity)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

            # Logging
            log_every = self.config["training"].get("log_every", 100)
            if batch_idx % log_every == 0 and self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {self.epoch} [{batch_idx}/{num_batches}] LR={lr:.2e} | "
                    f"Total={loss_dict['total_loss']:.4f} "
                    f"Recon={loss_dict['loss_recon']:.4f} "
                    f"Vel={loss_dict['loss_velocity']:.4f} "
                    f"PP={loss_dict['perplexity']:.1f}"
                )
                if self.use_wandb:
                    self._wandb_log(
                        {f"train_step/{k}": v for k, v in loss_dict.items()}
                        | {"train_step/lr": lr},
                        step=self.global_step,
                    )

            self.global_step += 1

        if self.is_main:
            elapsed = time.time() - t0
            print(f"Epoch {self.epoch} finished in {elapsed:.1f}s")

        for k in total_losses:
            total_losses[k] /= max(num_batches, 1)
        return total_losses

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self):
        self.model.eval()
        total_losses: dict = {}
        num_batches = len(self.val_loader)
        first_recon, first_orig = None, None

        amp_ctx = torch.amp.autocast(device_type="cuda") if self.use_amp else torch.amp.autocast(device_type="cuda", enabled=False)
        with torch.no_grad(), amp_ctx:
            for batch_idx, batch in enumerate(self.val_loader):
                motion = batch.to(self.device, non_blocking=True)
                recon, commit_loss, perplexity = self.model(motion)
                _, loss_dict = self.loss_fn(recon, motion, commit_loss, perplexity)
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
                if batch_idx == 0:
                    first_recon = recon.clone()
                    first_orig = motion.clone()

        # Generate validation video
        video_every = self.config["training"].get("video_every", 50)
        if first_recon is not None and self.epoch % video_every == 0 and self.is_main:
            self._generate_validation_video(first_recon, first_orig)

        for k in total_losses:
            total_losses[k] /= max(num_batches, 1)
        return total_losses

    # ------------------------------------------------------------------
    # Validation video generation
    # ------------------------------------------------------------------
    def _generate_validation_video(self, recon: torch.Tensor, orig: torch.Tensor, sample_idx: int = 0):
        """Dump a CSV pair (reconstruction vs. original) for the validation sample.

        NOTE: the original internal build shelled out to a standalone
        EGL-based renderer (``visualize_csv_egl.py``) to turn these CSVs into
        an mp4 and log it to WandB. That script lives outside this release
        and is not shipped here. If you want validation videos, render
        ``recon_motion.csv`` / ``orig_motion.csv`` with your own visualizer
        (see the repo-level visualization tooling) and log the result
        yourself — this hook intentionally stops at CSV export.
        """
        try:
            recon_np = recon[sample_idx].cpu().numpy()
            orig_np = orig[sample_idx].cpu().numpy()

            recon_csv = convert_to_csv_format(recon_np)
            orig_csv = convert_to_csv_format(orig_np)

            vis_dir = self.log_dir / "visualizations" / f"epoch_{self.epoch}"
            vis_dir.mkdir(parents=True, exist_ok=True)

            recon_csv_path = vis_dir / "recon_motion.csv"
            orig_csv_path = vis_dir / "orig_motion.csv"

            pd.DataFrame(recon_csv).to_csv(recon_csv_path, header=False, index=False)
            pd.DataFrame(orig_csv).to_csv(orig_csv_path, header=False, index=False)
            print(f"Saved validation CSVs to {vis_dir} (mp4 rendering not included in this release)")

            # TODO: render recon_csv_path / orig_csv_path to mp4 with your own
            # visualizer and, if desired, wandb.Video(...) it via self._wandb_log.
        except Exception as e:
            print(f"Validation video error: {e}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, is_best=False):
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        path = self.ckpt_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.ckpt_dir / "latest_checkpoint.pt")
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best_model.pt")
            print(f"Best model saved (val_loss={self.best_val_loss:.4f})")

    def load_checkpoint(self, ckpt_path, reset_lr=None):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        if reset_lr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = reset_lr
            if hasattr(self.scheduler, "base_lrs"):
                self.scheduler.base_lrs = [reset_lr] * len(self.scheduler.base_lrs)
        else:
            # Fast-forward the scheduler (built with the current config's T_max)
            # to the correct LR for self.epoch.  This avoids stale T_max values
            # that get baked into a saved scheduler_state_dict from an older run.
            for _ in range(self.epoch):
                self.scheduler.step()

        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self):
        num_epochs = self.config["training"]["num_epochs"]
        if self.is_main:
            print(f"Starting FSQ training for {num_epochs} epochs")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            t0 = time.time()

            train_losses = self.train_epoch()
            val_losses = self.validate()

            elapsed = time.time() - t0
            current_val = val_losses.get("total_loss", float("inf"))
            is_best = current_val < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val

            save_every = self.config["training"].get("save_every", 5)
            if epoch % save_every == 0 or is_best:
                if self.is_main:
                    self.save_checkpoint(is_best)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            if self.is_main:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch} ({elapsed:.1f}s, LR={lr:.2e})")
                print(f"TRAIN: Total={train_losses.get('total_loss',0):.4f} "
                      f"Recon={train_losses.get('loss_recon',0):.4f} "
                      f"Vel={train_losses.get('loss_velocity',0):.4f} "
                        f"Acc={train_losses.get('loss_acceleration',0):.4f} "
                      f"PP={train_losses.get('perplexity',0):.1f}")
                print(f"  RPos={train_losses.get('loss_root_pos',0):.4f} "
                      f"RQuat={train_losses.get('loss_root_quat',0):.4f} "
                      f"JPos={train_losses.get('loss_joint_pos',0):.4f} "
                      f"RVel={train_losses.get('loss_root_vel',0):.4f} "
                        f"JVel={train_losses.get('loss_joint_vel',0):.4f} "
                        f"RAcc={train_losses.get('loss_root_acc',0):.4f} "
                        f"JAcc={train_losses.get('loss_joint_acc',0):.4f}")
                print(f"VAL:   Total={val_losses.get('total_loss',0):.4f} "
                      f"Recon={val_losses.get('loss_recon',0):.4f} "
                      f"Vel={val_losses.get('loss_velocity',0):.4f} "
                        f"Acc={val_losses.get('loss_acceleration',0):.4f} "
                      f"PP={val_losses.get('perplexity',0):.1f}")
                print(f"  RPos={val_losses.get('loss_root_pos',0):.4f} "
                      f"RQuat={val_losses.get('loss_root_quat',0):.4f} "
                      f"JPos={val_losses.get('loss_joint_pos',0):.4f} "
                      f"RVel={val_losses.get('loss_root_vel',0):.4f} "
                        f"JVel={val_losses.get('loss_joint_vel',0):.4f} "
                        f"RAcc={val_losses.get('loss_root_acc',0):.4f} "
                        f"JAcc={val_losses.get('loss_joint_acc',0):.4f}")
                print(f"{'='*70}\n")

                if self.use_wandb:
                    wlog = {f"train_epoch/{k}": v for k, v in train_losses.items()}
                    wlog.update({f"val_epoch/{k}": v for k, v in val_losses.items()})
                    wlog["epoch"] = epoch
                    wlog["learning_rate"] = lr
                    wlog["best_val_loss"] = self.best_val_loss
                    self._wandb_log(wlog, step=self.global_step)

        if self.is_main:
            print("Training completed!")
        if self.use_wandb:
            wandb.finish()


# ============================================================================
# Config & CLI
# ============================================================================

def load_config(path):
    """Load a YAML config, resolving ``${oc.env:TSD_ASSETS}`` / ``${oc.env:TSD_DATA}``
    interpolations via OmegaConf, then return a plain dict (the rest of this
    script indexes the config like a dict)."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def find_latest_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    latest = ckpt_dir / "latest_checkpoint.pt"
    if latest.exists():
        return str(latest)
    files = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return str(files[-1]) if files else None


def main():
    parser = argparse.ArgumentParser(description="Train FSQ motion tokenizer")
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "fsq_combined.yaml"),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-lr", type=float, default=None)
    parser.add_argument("--data-folder", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    # DDP init
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}, using defaults")
        config = _default_config()
        with open(args.config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        config = load_config(args.config)

    if args.data_folder:
        config["data"]["data_folder"] = args.data_folder
    if args.no_wandb:
        config["use_wandb"] = False
    if args.wandb_project:
        config.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_name:
        config.setdefault("wandb", {})["name"] = args.wandb_name

    trainer = FSQTrainer(config)

    if args.resume:
        path = args.resume
        if path.lower() == "latest":
            path = find_latest_checkpoint(config["training"]["checkpoint_dir"])
        if path:
            trainer.load_checkpoint(path, reset_lr=args.reset_lr)

    trainer.train()


def _default_config():
    return {
        "model": {
            "nfeats": 36,
            "fsq_levels": [3, 3, 3, 3, 3, 2, 2, 2, 2, 2],  # 7776 codes
            "output_emb_width": 512,
            "down_t": 2,
            "stride_t": 2,
            "width": 512,
            "depth": 3,
            "dilation_growth_rate": 3,
            "norm": "BN",
            "activation": "relu",
            "upsample_mode": "nearest",
            "normalization_stats_file": None,
            "normalize_root_delta": False,
            "normalize_joint_pos": False,
            "normalization_epsilon": 1e-6,
        },
        "loss": {
            "lambda_root_pos": 1.0,
            "lambda_root_quat": 5.0,
            "lambda_joint_pos": 1.0,
            "lambda_root_vel": 0.5,
            "lambda_joint_vel": 1.0,
            "lambda_root_acc": 0.0,
            "lambda_joint_acc": 0.0,
            "recons_loss": "l1_smooth",
        },
        "optimizer": {
            "lr": 2e-4,
            "betas": [0.9, 0.99],
            "weight_decay": 0.0,
        },
        "training": {
            "num_epochs": 200,
            "batch_size": 256,
            "num_workers": 8,
            "log_every": 100,
            "save_every": 5,
            "video_every": 10,
            "use_amp": True,
            "log_dir": "logs-fsq",
            "checkpoint_dir": "checkpoints/checkpoints-fsq",
        },
        "data": {
            "data_folder": str(data_path("CustomCombined", "new_joint_vecs")),
            "val_data_folder": None,
            "window_size": 100,
            "stride": 5,
            "val_stride": 10,
            "cache_size_files": 0,
        },
        "wandb": {
            "project": "fsq-motion",
            "name": None,
            "tags": ["fsq", "motion", "delta-root", "BN"],
            "notes": "FSQ motion tokenizer: single enc/dec, delta root pos, BN",
        },
        "use_wandb": True,
        "use_cuda": True,
    }


if __name__ == "__main__":
    main()
