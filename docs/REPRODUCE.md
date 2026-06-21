# Reproduce: end-to-end recipe

All paths resolve from `TSD_ASSETS` (checkpoints, default `./assets`) and `TSD_DATA`
(datasets, default `./data`). Importing the package or sourcing `.env` sets the defaults.

## 0. Install + fetch
```bash
pip install -e .
python scripts/download_assets.py        # checkpoints -> assets/, TEXEDO -> data/texedo
```

## 1. Prepare the dataset
```bash
python generator/scripts/prepare_dataset.py        # add --limit 50 for a smoke run
```
Downloads TEXEDO, flattens to `${TSD_DATA}/CustomCombined/`, and regenerates `TOKENS_FSQ/`
using the FSQ checkpoint. Also copies the multitask instruction templates into the dataset.

## 2. (Optional) Train the FSQ tokenizer
The released checkpoint is `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`. To retrain:
```bash
# (optional) recompute normalization stats
python tokenizer/precompute_fsq_stats.py
# train
python tokenizer/fsq_train.py --config tokenizer/configs/fsq_combined.yaml
# encode tokens for downstream training
python tokenizer/encode_motion_tokens.py \
    --checkpoint ${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt \
    --data-folder ${TSD_DATA}/CustomCombined/new_joint_vecs \
    --output-dir  ${TSD_DATA}/CustomCombined/TOKENS_FSQ
```

## 3. Train the generator (flan-t5 multitask LM)
```bash
cd generator
python train.py --cfg configs/config_fsq_multitask.yaml --cfg_assets configs/assets.yaml --nodebug
cd ..
```
Loads the frozen FSQ tokenizer from `TRAIN.PRETRAINED_VAE` and `google/flan-t5-base` from the HF hub.

## 4. Train the verifiers
Dynamic (physical plausibility reward):
```bash
python verifiers/dynamic/train_verifier.py \
    --train_csv <train_labels.csv> --eval_csv <eval_labels.csv> \
    --train_motion_dir <dir> --eval_motion_dir <dir> \
    --save_dir ${TSD_ASSETS}/verifiers/dynamic
```
Semantic (text–motion matching):
```bash
python verifiers/semantic/train_evaluator.py --config verifiers/semantic/configs/evaluator.yaml --step all
```

## 5. Inference + best-of-N
```bash
python -m pipeline.generate --task t2m --num-samples 8 --prompt "a person waves"
python -m pipeline.score    --motion-dir <dir> --caption "a person waves" --output scores.csv
python -m pipeline.select_best_of_n --scores scores.csv --motion-dir <dir> --copy-best-to best/
python scripts/visualize_npz.py --input-dir best/ --output-dir viz/
```

## Smoke (no GPU / before assets are uploaded)
```bash
python generator/scripts/prepare_dataset.py --limit 10 --skip-tokens   # flatten only
python scripts/download_assets.py --dry-run                            # resolve manifest
```
Per-component import smoke checks are listed in each component's README.
