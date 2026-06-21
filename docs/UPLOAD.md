# Uploading checkpoints to your own Hugging Face model repo

`scripts/download_assets.py` fetches every checkpoint from **one Hugging Face model repo** that you
control (the public `JianuoCao/TEXEDO` dataset repo is separate and already populated). Because the
checkpoints already live in this repo's `assets/` folder with the exact layout the manifest expects,
uploading is just packaging `assets/` and pushing it.

## 0. Prerequisites
```bash
pip install -U huggingface_hub
huggingface-cli login                                   # token with `write` access
huggingface-cli repo create TEXEDO-checkpoints --type model
```

## 1. Build the two tarballs the manifest expects
Two entries are marked `unpack: true` in `configs/paths.yaml` and ship as `.tar.gz`. Build them from
the bundled `assets/` (run from the repo root):
```bash
mkdir -p upload/{tokenizer,generator,verifiers/dynamic,verifiers/semantic,glove}

# semantic evaluator tree -> one archive
tar -czf upload/verifiers/semantic/t2m_custom36_combinedv2.tar.gz \
    -C assets/verifiers/semantic t2m_custom36_combinedv2

# glove vocab -> one archive (our_vab_* files at the archive root)
tar -czf upload/glove/glove.tar.gz -C assets/glove .
```

## 2. Stage the plain checkpoint files (copy from assets/)
```bash
cp assets/tokenizer/checkpoint_epoch_95.pt          upload/tokenizer/
cp assets/tokenizer/fsq_motion_stats_combined.npz   upload/tokenizer/
cp "assets/generator/epoch=489.ckpt"                upload/generator/
cp assets/verifiers/dynamic/checkpoint_last.pt      upload/verifiers/dynamic/
cp assets/verifiers/dynamic/norm_stats.npz          upload/verifiers/dynamic/
```

The resulting `upload/` mirrors each manifest entry's `remote` path exactly.

## 3. Upload
```bash
# resumable + parallel (best for the 3.4 GB generator ckpt)
hf upload-large-folder <you>/TEXEDO-checkpoints upload --repo-type=model
```
Or file-by-file: `huggingface-cli upload <you>/TEXEDO-checkpoints <local> <remote>`.

## 4. Point the manifest at your repo
Edit `configs/paths.yaml`:
```yaml
checkpoints:
  hf_repo: <you>/TEXEDO-checkpoints   # was: TODO_USER_MODEL_REPO
```

## 5. Verify
```bash
python scripts/download_assets.py --dry-run    # resolution only, no network
python scripts/download_assets.py              # fetch into assets/ on a fresh clone
```
`download_assets.py` downloads each `remote` into `${TSD_ASSETS}/<local>`, extracts any
`unpack: true` archive, and deletes the archive. The TEXEDO dataset downloads independently via
`snapshot_download` and needs no setup.

> Note: `assets/` is git-ignored, so the checkpoints are **not** committed to the code repo — they
> are distributed only through the Hugging Face model repo (or, on this machine, already staged
> locally for validation).
