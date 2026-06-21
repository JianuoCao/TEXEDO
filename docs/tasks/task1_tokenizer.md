# TASK-1 — FSQ Tokenizer

Read `docs/tasks/CONTRACT.md` first. **Train + inference only.**

## Objective
Package the FSQ motion tokenizer (36-dim motion ↔ discrete tokens) into `text-see-do/tokenizer/`.

## Source (READ-ONLY)
`/home/jianuo/projects/GenMimic/stage1-tokenize/fsq/`

## Target
`/home/jianuo/projects/text-see-do/tokenizer/`

## Copy these files (then clean paths)
| Source | Target | Notes |
|---|---|---|
| `fsq_arch.py` | `tokenizer/fsq_arch.py` | `FSQVae` model (encoder→FSQ→decoder). |
| `fsq_dataloader.py` | `tokenizer/fsq_dataloader.py` | `SlidingWindowDataset` over `.npz`. Default data path → `data()` from `textseedo.paths`. |
| `fsq_train.py` | `tokenizer/fsq_train.py` | DDP/AMP training loop. **Remove** the hardcoded `visualize_csv_egl.py` subprocess hook (lines ~489); replace with a no-op or a TODO comment — do not call an external absolute script. |
| `fsq_adapter.py` | `tokenizer/fsq_adapter.py` | `load_fsq_adapter()` inference wrapper. |
| `encode_motion_tokens.py` | `tokenizer/encode_motion_tokens.py` | Batch encode `.npz` → token arrays. Used by data prep (TASK-5). |
| `precompute_fsq_stats.py` | `tokenizer/precompute_fsq_stats.py` | Normalization stats. |
| `fsq_config_combined.yaml` | `tokenizer/configs/fsq_combined.yaml` | Active config. |
| `normalization/fsq_motion_stats_combined.npz` | `tokenizer/normalization/fsq_motion_stats_combined.npz` | Small stats file — copy into repo. |

## Drop (do NOT copy)
- `fsq_inference.py` (metrics/eval), `compare_then_finish.py` (experiment orchestrator),
  all other `fsq_config_*.yaml` experiment variants, `scripts/convert_bones_csv_to_amass_csv.py`,
  any `checkpoints/`, `logs/`, `wandb/`.

## Path cleanup (known absolutes — verify by grepping the copies)
- `fsq_config_combined.yaml`:
  - `normalization_stats_file:` → `${oc.env:TSD_ASSETS}/tokenizer/fsq_motion_stats_combined.npz`
    **OR** keep it pointing at the in-repo `tokenizer/normalization/...` — pick the in-repo copy and
    document it. (Stats ship in-repo; ckpt ships via assets.)
  - `data.data_folder: /data/combined_motion_npz` → `${oc.env:TSD_DATA}/CustomCombined/new_joint_vecs`
    (this is where prepared motions live; document that prep is TASK-5).
- `fsq_dataloader.py` line ~244 `__main__` example path → use `data(...)`.
- `fsq_train.py` line ~489 viz hook + line ~739 fallback data path → remove/parametrize.
- Confirm config loader reads `normalization_stats_file` correctly after the change.

## Notes
- `vector-quantize-pytorch` is the FSQ dep (already in repo `requirements.txt`).
- Checkpoint format keys: `model_state_dict`, `config`, `epoch`; `load_fsq_adapter` strips `module.`.

## Smoke check (state command in report)
`cd text-see-do && python -c "import sys; sys.path.insert(0,'.'); from tokenizer import fsq_arch, fsq_adapter; print('ok')"`
and confirm `grep -rn '/home/jianuo\|/data/' tokenizer/` returns nothing.

## Acceptance
- All listed files copied + import cleanly, no absolute user paths remain, `tokenizer/README.md`
  documents train (`fsq_train.py --config configs/fsq_combined.yaml`) and encode
  (`encode_motion_tokens.py`) usage.
