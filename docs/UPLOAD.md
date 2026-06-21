# Uploading checkpoints to your own Hugging Face model repo

This repo's `scripts/download_assets.py` expects every checkpoint to live in **one Hugging Face
model repo** that you control (the public `JianuoCao/TEXEDO` dataset repo is separate and already
populated). Follow these steps once, after training/collecting the checkpoints listed in
`docs/MODELS.md`.

## 0. Prerequisites

```bash
pip install -U huggingface_hub
huggingface-cli login          # paste a token with `write` access
```

Create the model repo (pick any name; this doc uses `<you>/text-see-do-checkpoints`):

```bash
huggingface-cli repo create text-see-do-checkpoints --type model
```

## 1. Stage the files locally with the exact remote layout

`download_assets.py` downloads each manifest entry's `remote` path verbatim from the repo root, so
stage a local folder that mirrors it exactly before uploading. Using the **original-tree source
paths** as upload sources (read-only — do not move/delete anything there):

| Manifest entry | Upload source (original tree, read-only) | Remote path in your HF repo |
|---|---|---|
| `fsq_tokenizer` | `GenMimic/stage1-tokenize/fsq/checkpoints/checkpoints-fsq-combined/checkpoint_epoch_95.pt` | `tokenizer/checkpoint_epoch_95.pt` |
| `fsq_norm_stats` | `GenMimic/stage1-tokenize/fsq/normalization/fsq_motion_stats_combined.npz` | `tokenizer/fsq_motion_stats_combined.npz` |
| `generator` | `MotionGPT/experiments/mgpt/CustomCombined_Stage2_FSQ_MultiTask-4-30/checkpoints/epoch=489.ckpt` | `generator/epoch=489.ckpt` |
| `dynamic_verifier` | `MotionGPT/Dynamic_Verifier/runs/v5/checkpoint_last.pt` | `verifiers/dynamic/checkpoint_last.pt` |
| `dynamic_norm_stats` | `MotionGPT/Dynamic_Verifier/runs/v3/norm_stats.npz` (paired with the `v5` checkpoint — see `docs/MODELS.md`) | `verifiers/dynamic/norm_stats.npz` |
| `semantic_evaluator` | `MotionGPT/deps/t2m_custom36_combinedv2/custom36/t2m/{text_mot_match,Decomp_SP001_SM001_H512,Comp_v6_KLD01}` packaged as one tarball | `verifiers/semantic/t2m_custom36_combinedv2.tar.gz` |
| `glove` | `MotionGPT/deps/t2m/glove/` packaged as one tarball | `glove/glove.tar.gz` |

Build the two tarballs (run from anywhere; these only *read* the original trees):

```bash
# semantic evaluator
tar -czf /tmp/t2m_custom36_combinedv2.tar.gz -C /home/jianuo/projects/MotionGPT/deps/t2m_custom36_combinedv2/custom36/t2m \
    text_mot_match Decomp_SP001_SM001_H512 Comp_v6_KLD01

# glove
tar -czf /tmp/glove.tar.gz -C /home/jianuo/projects/MotionGPT/deps/t2m glove
```

Stage everything else by copying (not moving) the plain files into a scratch upload folder, e.g.:

```bash
mkdir -p /tmp/upload/{tokenizer,generator,verifiers/dynamic,verifiers/semantic,glove}
cp /home/jianuo/projects/GenMimic/stage1-tokenize/fsq/checkpoints/checkpoints-fsq-combined/checkpoint_epoch_95.pt \
    /tmp/upload/tokenizer/
cp /home/jianuo/projects/GenMimic/stage1-tokenize/fsq/normalization/fsq_motion_stats_combined.npz \
    /tmp/upload/tokenizer/
cp "/home/jianuo/projects/MotionGPT/experiments/mgpt/CustomCombined_Stage2_FSQ_MultiTask-4-30/checkpoints/epoch=489.ckpt" \
    /tmp/upload/generator/
cp /home/jianuo/projects/MotionGPT/Dynamic_Verifier/runs/v5/checkpoint_last.pt \
    /tmp/upload/verifiers/dynamic/
cp /home/jianuo/projects/MotionGPT/Dynamic_Verifier/runs/v3/norm_stats.npz \
    /tmp/upload/verifiers/dynamic/
cp /tmp/t2m_custom36_combinedv2.tar.gz /tmp/upload/verifiers/semantic/
cp /tmp/glove.tar.gz /tmp/upload/glove/
```

## 2. Upload

For a handful of large files, `hf upload-large-folder` (resumable, parallel) is the most robust
option:

```bash
hf upload-large-folder <you>/text-see-do-checkpoints /tmp/upload --repo-type=model
```

Or upload file-by-file with `huggingface-cli upload` if you prefer more control / smaller batches:

```bash
huggingface-cli upload <you>/text-see-do-checkpoints \
    /tmp/upload/tokenizer/checkpoint_epoch_95.pt tokenizer/checkpoint_epoch_95.pt

huggingface-cli upload <you>/text-see-do-checkpoints \
    /tmp/upload/tokenizer/fsq_motion_stats_combined.npz tokenizer/fsq_motion_stats_combined.npz

huggingface-cli upload <you>/text-see-do-checkpoints \
    "/tmp/upload/generator/epoch=489.ckpt" "generator/epoch=489.ckpt"

huggingface-cli upload <you>/text-see-do-checkpoints \
    /tmp/upload/verifiers/dynamic/checkpoint_last.pt verifiers/dynamic/checkpoint_last.pt

huggingface-cli upload <you>/text-see-do-checkpoints \
    /tmp/upload/verifiers/dynamic/norm_stats.npz verifiers/dynamic/norm_stats.npz

huggingface-cli upload <you>/text-see-do-checkpoints \
    /tmp/upload/verifiers/semantic/t2m_custom36_combinedv2.tar.gz verifiers/semantic/t2m_custom36_combinedv2.tar.gz

huggingface-cli upload <you>/text-see-do-checkpoints \
    /tmp/upload/glove/glove.tar.gz glove/glove.tar.gz
```

## 3. Point the manifest at your repo

Edit `configs/paths.yaml`:

```yaml
checkpoints:
  hf_repo: <you>/text-see-do-checkpoints   # was: TODO_USER_MODEL_REPO
```

## 4. Verify

```bash
cd text-see-do
python scripts/download_assets.py --dry-run   # sanity-check resolution, no network calls
python scripts/download_assets.py              # actually fetch everything
```

`download_assets.py` will `hf_hub_download` each `remote` path into `${TSD_ASSETS}/<local>` and
automatically extract any entry marked `unpack: true` (the two `.tar.gz` archives above), then
delete the archive. The dataset (`JianuoCao/TEXEDO`) downloads independently via
`snapshot_download` and needs no setup on your part.
