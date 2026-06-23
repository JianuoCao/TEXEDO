# Generator — TEXEDO generator (FSQ multitask)

A flan-t5-base language model over FSQ motion tokens, trained multitask (text→motion, motion→text,
motion prediction). Stage 1 (the FSQ tokenizer) is frozen; this stage trains the LM.

## Attribution
Parts of the generator implementation are adapted from
[OpenMotionLab/MotionGPT](https://github.com/OpenMotionLab/MotionGPT).

## Package
`texedo_generator/` — the core package (import name `texedo_generator`), holding `archs`, `models`, `data`, `losses`,
`metrics`, `utils`. Key pieces:
- `texedo_generator/models/generator_model.py` — the `TEXEDO generator` Lightning module.
- `texedo_generator/archs/language_model.py` — flan-t5 LM wrapper (loads `google/flan-t5-base` from the HF hub).
- `texedo_generator/archs/motion_tokenizer.py` — `FSQTokenizer` wrapper; `texedo_generator/utils/load_checkpoint.py` loads the
  frozen FSQ checkpoint named by `TRAIN.PRETRAINED_VAE`.
- `texedo_generator/data/CustomCombined.py` (the `texedo_generator.data` subpackage) — the multitask datamodule.

## Configs (`configs/`)
Loaded by `texedo_generator/config.py:parse_args`, which globs `configs/*/*.yaml` into namespaces. **Run from this
`generator/` directory** so `./configs` resolves.
- `config_fsq_multitask.yaml` — primary experiment config (paths via `${oc.env:TSD_ASSETS|TSD_DATA}`).
- `assets.yaml` — asset/dataset roots (passed as `--cfg_assets`).
- `default.yaml` — base defaults (auto-loaded).
- `language_model/default.yaml`, `evaluator/tm2t.yaml`, `fsq/default.yaml` — sub-config namespaces.

Paths resolve from environment: `TSD_ASSETS` (checkpoints) and `TSD_DATA` (datasets). Importing
`utilities.paths` (or sourcing `.env`) sets sane defaults (`<repo>/assets`, `<repo>/data`).

## Data prep
```bash
# Download TEXEDO, flatten to the training layout, regenerate FSQ tokens + copy instruction templates
python scripts/prepare_dataset.py            # add --limit 50 for a smoke subset
```
Produces `${TSD_DATA}/CustomCombined/{new_joint_vecs,texts,TOKENS_FSQ,train|val|test.txt,
template_pretrain.json,template_instructions.json}`.

## Train
```bash
cd generator
python train.py --cfg configs/config_fsq_multitask.yaml --cfg_assets configs/assets.yaml --nodebug
```

## Inference (text → motion)
```bash
cd generator
python demo.py --task t2m --cfg configs/config_fsq_multitask.yaml --cfg_assets configs/assets.yaml
```
Requires the generator checkpoint at `${TSD_ASSETS}/generator/epoch=489.ckpt` and the frozen FSQ
checkpoint at `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`.

## Notes
- flan-t5-base is pulled from the public HF hub (`google/flan-t5-base`) at runtime — not vendored.
- SMPL rendering and the whisper/web-app path were intentionally dropped (the pipeline renders the
  Unitree G1 robot from the 36-dim representation).
- This release is **FSQ-only**: legacy VQVAE tokenizer configs have been removed. The only tokenizer config is
  `fsq/default.yaml`.
