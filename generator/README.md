# Generator — MotionGPT (FSQ multitask)

A flan-t5-base language model over FSQ motion tokens, trained multitask (text→motion, motion→text,
motion prediction). Stage 1 (the FSQ tokenizer) is frozen; this stage trains the LM.

## Package
`mgpt/` — the model/data/arch package (import name `mgpt`). Key pieces:
- `mgpt/models/mgpt.py` — the `MotionGPT` Lightning module.
- `mgpt/archs/mgpt_lm.py` — flan-t5 LM wrapper (loads `google/flan-t5-base` from the HF hub).
- `mgpt/archs/motion_tokenizer.py` — `FSQTokenizer` wrapper; `mgpt/utils/load_checkpoint.py` loads the
  frozen FSQ checkpoint named by `TRAIN.PRETRAINED_VAE`.
- `mgpt/data/CustomCombined.py` — the multitask datamodule.

## Configs (`configs/`)
Loaded by `mgpt/config.py:parse_args`, which globs `configs/*/*.yaml` into namespaces. **Run from this
`generator/` directory** so `./configs` resolves.
- `config_fsq_multitask.yaml` — primary experiment config (paths via `${oc.env:TSD_ASSETS|TSD_DATA}`).
- `assets.yaml` — asset/dataset roots (passed as `--cfg_assets`).
- `default_ours.yaml` — base defaults (auto-loaded).
- `lm/default.yaml`, `evaluator/tm2t.yaml`, `vq/fsq.yaml` — sub-config namespaces.

Paths resolve from environment: `TSD_ASSETS` (checkpoints) and `TSD_DATA` (datasets). Importing
`textseedo.paths` (or sourcing `.env`) sets sane defaults (`<repo>/assets`, `<repo>/data`).

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
- This release is **FSQ-only**: the legacy VQVAE tokenizer (`DualVQVaeWrapper`, `vq/vqvae_ours.yaml`,
  `vq/default.yaml`) has been removed. The only tokenizer is `FSQTokenizer` (`vq/fsq.yaml`).
