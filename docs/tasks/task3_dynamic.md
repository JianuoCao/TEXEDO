# TASK-3 — Dynamic Verifier

Read `docs/tasks/CONTRACT.md` first. **Train + inference only.**

## Objective
Package the dynamic verifier (Transformer reward model scoring physical plausibility / task progress
of a 36-dim motion) into `text-see-do/verifiers/dynamic/`.

## Source (READ-ONLY)
`/home/jianuo/projects/MotionGPT/Dynamic_Verifier/`

## Target
`/home/jianuo/projects/text-see-do/verifiers/dynamic/`

## Copy these
| Source | Target | Notes |
|---|---|---|
| `model.py` | `verifiers/dynamic/model.py` | `DynamicVerifier` (causal transformer, 3 heads: success/dynamics/progress; Q* reward fusion). |
| `dataset.py` | `verifiers/dynamic/dataset.py` | 36→94-dim feature transform, norm stats. Parametrize any default data dir via `textseedo.paths.data`. |
| `train_verifier.py` | `verifiers/dynamic/train_verifier.py` | Training entry (args: `--train_csv --eval_csv --train_motion_dir --eval_motion_dir --save_dir ...`). Already arg-driven; just remove any absolute defaults. |
| `predict_rewards_v3.py` | `verifiers/dynamic/predict_rewards.py` | Inference: motion CSVs → reward CSV. Default ckpt path → `${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt`, norm_stats → `.../norm_stats.npz`. |

## Drop (do NOT copy)
- `evaluate_verifier.py` (AUROC/AUPRC eval), `finetune_pairwise.py` (experiment), the entire
  `scripts/` dir (analysis/plotting incl. `analyze_dyn_bias.py`, `plot_best_of_n.py`), `runs/`
  (checkpoints/predictions → assets).

## Path cleanup
- `dataset.py` / `predict_rewards.py`: replace any `/home/jianuo` or `/data` default with an argparse
  default derived from `textseedo.paths` (e.g. `data("CustomCombined")`, `assets("verifiers/dynamic")`).
- The component must NOT import from `mGPT`/`generator` (it's self-contained — keep it that way).

## Document (in `verifiers/dynamic/README.md`)
- Checkpoint schema: `{"epoch", "model" (state_dict), "optimizer", "val_metrics"}` saved as
  `checkpoint_last.pt` / `checkpoint_best.pt`; `norm_stats.npz` keys: `mean, std, pos_weight,
  accel_p95, vel_p95`.
- The Q* reward formula: `r = p_s*(1+α·p_d)/(1+α) + (1-p_s)·β·p_g·p_d`, with `α=0.4, β=0.6`.
- Input `(B, T, 94)` features + padding mask; outputs `success_prob, dynamics_hat, progress_hat, reward_hat`.

## Smoke check
`cd text-see-do && python -c "import sys; sys.path.insert(0,'.'); from verifiers.dynamic import model, dataset; print('ok')"`
and `grep -rn '/home/jianuo\|/data/' verifiers/dynamic/` returns nothing.

## Acceptance
- Files import; train + predict are arg-driven with repo-relative defaults; README documents ckpt/
  norm_stats schema + reward formula; no absolute paths.
