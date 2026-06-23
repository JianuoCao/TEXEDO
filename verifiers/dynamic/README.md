# Dynamic Verifier

A Transformer reward model that scores the physical plausibility and task
progress of a generated 36-dim motion. It is one of two verifiers used for
best-of-N candidate selection in the TEXEDO pipeline (the other is the
semantic verifier in `verifiers/semantic/`). This component is self-contained
(torch / numpy / pandas / sklearn / scipy only) and does **not** import from
`generator` or `TEXEDO generator`.

## Architecture (`model.py`)

`DynamicVerifier`: causal Transformer encoder over per-frame features, with
three heads predicting different aspects of motion quality:

- **Input**: `(B, T, 94)` normalized features + `(B, T)` boolean padding mask
  (`True` = padded position).
- `InputProjection` splits the 94 dims into 4 semantic groups (root dynamics
  7, joint pos 29, joint vel 29, joint accel 29), projects each to 128-d,
  concatenates, and fuses to `d_model=256`.
- `CausalTransformerEncoder`: 4-layer, 4-head, Pre-LN Transformer encoder
  with a causal attention mask (frame `t` only attends to frames `<= t`).
- `MeanAttentionPooling`: concatenation of masked mean-pooling and learned
  attention-pooling over the time axis, projected back to `d_model`.
- Three MLP heads on the pooled representation:
  - `success_head` -> `success_logit` / `success_prob = sigmoid(logit)`
  - `dynamics_head` -> `dynamics_hat` (motion smoothness/physical quality, sigmoid)
  - `progress_head` -> `progress_hat` (task-completion fraction, sigmoid)

**Output dict**: `success_logit, success_prob, dynamics_hat, progress_hat, reward_hat`.

### Q* hierarchical reward fusion

```
reward = success_prob * (1 + alpha * dynamics_hat) / (1 + alpha)
         + (1 - success_prob) * beta * progress_hat * dynamics_hat
```

with `alpha = 0.4`, `beta = 0.6` (module-level constants `ALPHA`, `BETA` in
`model.py`, also exposed as the `fuse_reward()` function). The constraint
`beta < 1 / (1 + alpha)` guarantees **hierarchical consistency**: any sample
with `success_prob = 1` scores strictly higher than any sample with
`success_prob = 0`, regardless of their dynamics/progress values. Among
samples with the same success outcome, the formula ranks by dynamics quality
(successes) or by progress x dynamics (failures).

## Feature transform: 36 -> 94 dims (`dataset.py`)

`transform_36_to_94(motion)` turns a raw `(T, 36)` motion (see
`utilities.motion_format`: root pos(3) + root quat wxyz(4) + 29 joints) into a
`(T, 94)` dynamics-aware feature vector:

| Slice | Dims | Content |
|---|---|---|
| `0:2`  | 2  | `delta_root_xy` (frame-to-frame root displacement; frame 0 = 0) |
| `2:3`  | 1  | `root_z` (height) |
| `3:7`  | 4  | root quaternion (wxyz) |
| `7:36` | 29 | joint angles |
| `36:65`| 29 | joint velocity (finite difference; frame 0 = 0) |
| `65:94`| 29 | joint acceleration (2nd finite difference; frames 0,1 = 0) |

Features are z-normalized with stored `mean`/`std`, then `nan_to_num` and
clipped to `[-10, 10]`.

## Checkpoint and norm-stats schema

`checkpoint_last.pt` / `checkpoint_best.pt` (produced by `train_verifier.py`,
loaded by `predict_rewards.py`) is a `torch.save` dict:

```python
{
    "epoch":       int,
    "model":       model.state_dict(),     # DynamicVerifier
    "optimizer":   optimizer.state_dict(), # AdamW
    "val_metrics": {...},                  # dict of val losses + metrics for this epoch
}
```

`norm_stats.npz` (produced by `dataset.compute_norm_stats`, consumed by
`MotionDataset` / `PredictDataset`) has keys:

| Key | Shape | Meaning |
|---|---|---|
| `mean` | `(94,)` | per-dim feature mean (winsorized p1/p99 for delta_xy + joint_accel dims before averaging) |
| `std` | `(94,)` | per-dim feature std (+1e-6) |
| `pos_weight` | scalar | `n_negative / n_positive`, for `BCEWithLogitsLoss` on the success head |
| `accel_p95` | scalar | 95th percentile of `accel_dist` label across train samples, used to map raw accel error -> a `[0,1]` quality score |
| `vel_p95` | scalar | 95th percentile of `vel_dist` label, same purpose for velocity error |

Expected asset paths (see `utilities.paths`):
- `${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt`
- `${TSD_ASSETS}/verifiers/dynamic/norm_stats.npz`

## Training

```bash
conda activate TEXEDO
cd TEXEDO

python -m verifiers.dynamic.train_verifier \
    --train_csv  /path/to/train_labels.csv \
    --eval_csv   /path/to/eval_labels.csv \
    --train_motion_dir /path/to/train_motion_csvs \
    --eval_motion_dir  /path/to/eval_motion_csvs \
    --save_dir   runs/dynamic_v1 \
    --w_success 1.0 --w_dynamics 0.6 --w_progress 0.8 \
    --no_wandb
```

Label CSVs need at least the columns `traj_id, success, progress, accel_dist,
vel_dist` (optional `mpjpe_l`). Motion files live in `--train_motion_dir` /
`--eval_motion_dir`, named `{traj_id}.npy` (preferred), `.csv`, or `.npz`.
`norm_stats.npz` is auto-computed from the train split into
`<save_dir>/norm_stats.npz` if `--norm_stats` is not given and that file
doesn't already exist. Checkpoints are written to `<save_dir>/checkpoint_{last,best}.pt`
every epoch (best = lowest val loss so far).

Joint-loss training optimizes all three heads simultaneously from epoch 0;
there is no staged schedule or pairwise ranking loss — the hierarchical
ordering guarantee comes entirely from the Q* fusion formula above, not from
the training objective.

## Inference

```bash
python -m verifiers.dynamic.predict_rewards \
    --tracking_csv runs/tracking_predictions.csv \
    --motion_dir   /path/to/motion_npy_dir \
    --checkpoint   ${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt \
    --norm_stats   ${TSD_ASSETS}/verifiers/dynamic/norm_stats.npz \
    --output_csv   runs/predictions_with_reward.csv
```

`--tracking_csv` must contain a `motion_key` column; matching `{motion_key}.npy`
files are read from `--motion_dir`. The output CSV is the input merged
(left join on `motion_key`) with `reward_hat, success_prob, dynamics_hat,
progress_hat`. If `--norm_stats` is omitted it defaults to
`<checkpoint_dir>/norm_stats.npz`; if `--checkpoint` is omitted it defaults to
`${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt`.

## What was intentionally dropped

This package keeps **training + inference only**:

- `evaluate_verifier.py` (AUROC/AUPRC/Kendall-tau/best-of-N evaluation suite) — not copied.
- `finetune_pairwise.py` (pairwise-ranking finetuning experiment) — not copied.
- `scripts/` (bias analysis, best-of-N plotting, etc.) — not copied.
- `runs/` (experiment checkpoints/predictions) — not copied; production
  checkpoints belong under `${TSD_ASSETS}/verifiers/dynamic/` instead.

## Smoke check

```bash
cd TEXEDO && python -c "import sys; sys.path.insert(0,'.'); from verifiers.dynamic import model, dataset; print('ok')"
```
