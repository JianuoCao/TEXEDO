# Reproduce: end-to-end recipe

All paths resolve from `TSD_ASSETS` (checkpoints, default `./assets`) and `TSD_DATA`
(datasets, default `./data`). Importing the package or sourcing `.env` sets the defaults.

> **Environments.** Per the project convention, use the `mgpt` conda env for the generator
> and verifiers (needs `transformers`), and the `mimic` env for the FSQ tokenizer / data prep /
> visualization. Both training and inference run on **CPU** when no GPU is present (slow);
> they auto-select CUDA when available.
>
> **What is reproducible from the public TEXEDO dataset:** the **generator** and **semantic
> verifier** train directly from the prepared `CustomCombined`. The **FSQ tokenizer** and
> **dynamic verifier** ship as frozen checkpoints — retraining them needs extra inputs not in
> TEXEDO (see the notes in steps 2 and 4).

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
The released checkpoint is `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`.

> **Note:** retraining needs the *raw* multi-body motion NPZs (`body_pos_w`, `body_quat_w`,
> `joint_pos`) that `tokenizer/fsq_dataloader.py` consumes — the public TEXEDO dataset only
> ships the reduced flat `(T, 36)` `.npy`. Point `data.data_folder` in
> `tokenizer/configs/fsq_combined.yaml` at your raw NPZ folder to retrain. To retrain:
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
Semantic (text–motion matching) — trains directly from `CustomCombined`:
```bash
python verifiers/semantic/train_evaluator.py --step all      # add --gpu -1 to force CPU
```
Dynamic (physical-plausibility reward) — needs labeled motion CSVs (`success`, `progress`,
`accel_dist`, `vel_dist`) produced by physics rollouts; these are **not** part of TEXEDO, so the
dynamic verifier ships as a frozen checkpoint. To retrain with your own labels:
```bash
python verifiers/dynamic/train_verifier.py \
    --train_csv <train_labels.csv> --eval_csv <eval_labels.csv> \
    --train_motion_dir <dir_of_{id}.npy> --eval_motion_dir <dir> \
    --save_dir ${TSD_ASSETS}/verifiers/dynamic
```

## 5. Inference + best-of-N
```bash
python -m pipeline.generate --prompt "a person waves" --num-samples 8 --out-dir candidates/
python -m pipeline.score    --motion-dir candidates/ --caption "a person waves" --output scores.csv
python -m pipeline.select_best_of_n --scores scores.csv --motion-dir candidates/ --copy-best-to best/
python scripts/visualize_npz.py --input-dir best/ --output-dir viz/
```

## Smoke (no GPU / before assets are uploaded)
```bash
python generator/scripts/prepare_dataset.py --limit 10 --skip-tokens   # flatten only
python scripts/download_assets.py --dry-run                            # resolve manifest
```
Per-component import smoke checks are listed in each component's README.
