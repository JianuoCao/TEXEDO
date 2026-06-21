# TASK-2 — Motion Generator (MotionGPT FSQ-multitask)

Read `docs/tasks/CONTRACT.md` first. **Train + inference only.**

## Objective
Package the FSQ-multitask MotionGPT path (text → motion tokens → motion) into `text-see-do/generator/`.
Keep the Python import name `mgpt` to minimize edits inside the package.

## Source (READ-ONLY)
`/home/jianuo/projects/MotionGPT/`

## Target
`/home/jianuo/projects/text-see-do/generator/`

## Copy these
| Source | Target | Notes |
|---|---|---|
| `mGPT/` | `generator/mgpt/` | Whole package. After copy, fix only what's needed for the FSQ-multitask path + path cleanup. Keep `import mGPT...`? **No** — rename package dir to `mgpt` and update internal imports from `mGPT` → `mgpt` (repo-wide within the copy only). |
| `train.py` | `generator/train.py` | Stage-2 LM training entry. |
| `demo_ours.py` | `generator/demo.py` | Text→motion inference. |
| `configs/config_combined_stage2_fsq_multitask.yaml` | `generator/configs/config_fsq_multitask.yaml` | Primary config. |
| `configs/assets_custom36.yaml` | `generator/configs/assets.yaml` | Asset/dataset paths. |
| `configs/vq/fsq.yaml` | `generator/configs/vq/fsq.yaml` | FSQ tokenizer config block. |
| `configs/default_ours.yaml` | `generator/configs/default.yaml` | Base defaults (auto-loaded). Update any internal references to the renamed config files. |
| `mGPT/utils/load_checkpoint.py` | (inside `mgpt/`) | FSQ-ckpt loader — keep, it's the load path. |
| `mGPT/archs/motion_tokenizer.py` | (inside `mgpt/`) | `FSQTokenizer` wrapper — keep. |
| `mGPT/data/CustomCombined.py` | (inside `mgpt/`) | The active datamodule. |

## Drop (do NOT copy)
- `test.py` (evaluation), `web_app.py`, `fit.py`, `render.py`, `demo_ours.py`'s eval branches if any.
- Legacy datamodules/configs: `CustomData.py`, `CustomData_LONG.py`, `CustomDataAll.py`, HumanML3D/Kit
  configs, all `config_custom_*` except the FSQ-multitask one, vqvae_v1/v2 references, whisper.
- `deps/`, `datasets/`, `experiments/`, `checkpoints/` (these are assets/data, handled elsewhere).
- Any `import`s of whisper / SMPL render paths — stub or remove cleanly so the FSQ path still imports.

## Path cleanup (known absolutes)
In `config_fsq_multitask.yaml`:
- `TRAIN.RESUME` → `''` (empty; user sets their own run dir) or `${oc.env:TSD_ASSETS}/generator`.
- `TRAIN.PRETRAINED_VAE` → `${oc.env:TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`.
- `DATASET.DATAPATH` / `CUSTOM_COMBINED.ROOT` → `${oc.env:TSD_DATA}/CustomCombined`.
- `TEST.CHECKPOINTS` → `${oc.env:TSD_ASSETS}/generator/epoch=489.ckpt`.
In `assets.yaml`:
- `METRIC.TM2T.t2m_path: deps/t2m_custom36_combinedv2/` → `${oc.env:TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2/`.
- `WORD_VERTILIZER_PATH: deps/glove/` → `${oc.env:TSD_ASSETS}/glove/`.
- `model.whisper_path` → remove.
- dataset ROOTs under `datasets/...` → `${oc.env:TSD_DATA}/...`.
- flan-t5: ensure the LM loads `google/flan-t5-base` from HF hub (not a local `deps/flan-t5-base`).
  Update the relevant config key / `mgpt/archs/mgpt_lm.py` model path to the HF id.

## Confirm the config loader supports `${oc.env:...}`
`mGPT/config.py parse_args()` uses OmegaConf — env interpolation should resolve since `textseedo.paths`
exports `TSD_ASSETS`/`TSD_DATA`. If the loader registers a custom `eval` resolver, keep it.

## Smoke check
`cd text-see-do && python -c "import sys; sys.path.insert(0,'.'); import generator.mgpt; print('ok')"`
and `grep -rn '/home/jianuo\|/data/' generator/` returns nothing. (Importing the full model needs
torch/transformers; a package import + config-parse smoke is sufficient.)

## Acceptance
- `generator/mgpt/` imports; configs resolve env paths; `generator/README.md` documents train
  (`python train.py --cfg configs/config_fsq_multitask.yaml --cfg_assets configs/assets.yaml --nodebug`)
  and inference (`python demo.py --task t2m ...`). No absolute user paths remain.
