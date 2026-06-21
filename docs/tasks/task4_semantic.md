# TASK-4 — Semantic Verifier (text–motion matching)

Read `docs/tasks/CONTRACT.md` first. **Train + inference only.**

## Objective
Package the semantic verifier (BiGRU text/motion encoders producing a text–motion matching distance)
into `text-see-do/verifiers/semantic/`.

## Source (READ-ONLY)
`/home/jianuo/projects/MotionGPT/train_evaluator_36dim/`

## Target
`/home/jianuo/projects/text-see-do/verifiers/semantic/`

## Copy / refactor
| Source | Target | Notes |
|---|---|---|
| `train_evaluator.py` | `verifiers/semantic/train_evaluator.py` | 3-step training: (1) mean/std, (2) decomp `MovementConvEncoder/Decoder`, (3) match `TextEncoderBiGRUCo`+`MotionEncoderBiGRUCo` (contrastive). |
| — | `verifiers/semantic/models.py` | **Extract** the 4 encoder classes (`MovementConvEncoder`, `MovementConvDecoder`, `TextEncoderBiGRUCo`, `MotionEncoderBiGRUCo`) into this module so they import from one place. `train_evaluator.py` should import them from `models.py`. |
| — | `verifiers/semantic/inference.py` | **New, minimal:** load `finest.tar`, encode a motion `(T,36)` and a caption → 512-d embeddings → return L2 matching distance. Reuse `root_pos_to_vel` preprocessing from the original. |

## Config refactor
The original uses a `Config` Python class (lines ~78–168) full of absolute paths. Replace with a YAML
`verifiers/semantic/configs/evaluator.yaml` using `${oc.env:...}`:
- `data_root` → `${oc.env:TSD_DATA}/CustomCombined`
- `glove_dir` → `${oc.env:TSD_ASSETS}/glove`
- `save_root` → `${oc.env:TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2`
- keep numeric hyperparams (`dim_pose=36`, `max_motion_length=2048`, `unit_length=4`,
  `dim_movement_latent=512`, `dim_text_hidden=512`, `dim_motion_hidden=1024`, match lr/batch/epochs).
`train_evaluator.py` loads this YAML (OmegaConf) instead of the hardcoded class.

## Drop (do NOT copy)
- `eval_best_of_n.py`, `eval_checkpoint.py`, `eval_generated.py`, `eval_kimodo.py` (all evaluation).

## Checkpoint paths
- Final matching ckpt: `${TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m/text_mot_match/model/finest.tar`
  (keys: `text_encoder, movement_encoder, motion_encoder`).
- Mean/std meta: `.../Comp_v6_KLD01/meta/{mean.npy,std.npy}`.

## Self-containment
Keep self-contained (torch/numpy only). `inference.py` is what the pipeline (TASK Phase-2) imports for
best-of-N matching — make its API clean: `load_evaluator(ckpt, meta_dir, glove_dir, device)` →
object with `.score(motion_36d, caption) -> float`.

## Smoke check
`cd text-see-do && python -c "import sys; sys.path.insert(0,'.'); from verifiers.semantic import models; print('ok')"`
and `grep -rn '/home/jianuo\|/data/' verifiers/semantic/` returns nothing.

## Acceptance
- `models.py` holds the encoders; `train_evaluator.py` trains from the YAML config; `inference.py`
  exposes a clean scoring API; README documents the 3-step training + finest.tar usage; no absolute paths.
