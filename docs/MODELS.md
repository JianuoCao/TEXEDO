# Models & assets

Every large file (checkpoints + dataset) is hosted on the Hugging Face Hub rather than committed to
this repo. `scripts/download_assets.py` fetches them into `${TSD_ASSETS}` (checkpoints) and
`${TSD_DATA}` (datasets) using the manifest in `configs/paths.yaml`.

```bash
python scripts/download_assets.py --dry-run   # see what would be fetched
python scripts/download_assets.py             # fetch everything
python scripts/download_assets.py --only fsq_tokenizer,generator
```

The dataset (`JianuoCao/TEXEDO`) is already public. The checkpoints below ship in a *separate*
Hugging Face **model** repo that you create yourself — see `docs/UPLOAD.md` for the exact upload
steps. Until `checkpoints.hf_repo` in `configs/paths.yaml` is set to your repo, `--dry-run` still
resolves and prints the intended layout; a real (non-dry-run) download will raise a clear error
pointing back to `docs/UPLOAD.md`.

## Checkpoint manifest

| Logical name | What it is | Original-tree source (READ-ONLY) | Destination (`${TSD_ASSETS}/...`) | Approx. size |
|---|---|---|---|---|
| `fsq_tokenizer` | FSQ motion tokenizer weights (encoder/decoder + FSQ codebook), 36-dim G1 motion | `GenMimic/stage1-tokenize/fsq/checkpoints/checkpoints-fsq-combined/checkpoint_epoch_95.pt` | `tokenizer/checkpoint_epoch_95.pt` | ~216 MB |
| `fsq_norm_stats` | Per-channel normalization stats (mean/std) used by the FSQ tokenizer | `GenMimic/stage1-tokenize/fsq/normalization/fsq_motion_stats_combined.npz` | `tokenizer/fsq_motion_stats_combined.npz` | ~2 KB |
| `generator` | Stage-2 text→motion generator: flan-t5-base LM head fine-tuned on FSQ motion tokens (multi-task) | `MotionGPT/experiments/mgpt/CustomCombined_Stage2_FSQ_MultiTask-4-30/checkpoints/epoch=489.ckpt` | `generator/epoch=489.ckpt` | ~3.2 GB |
| `dynamic_verifier` | Dynamic-feasibility verifier (kinematic/dynamics scorer) checkpoint, run `v5` | `MotionGPT/Dynamic_Verifier/runs/v5/checkpoint_last.pt` | `verifiers/dynamic/checkpoint_last.pt` | ~40 MB |
| `dynamic_norm_stats` | Normalization stats paired with the `v5` checkpoint | `MotionGPT/Dynamic_Verifier/runs/v3/norm_stats.npz` | `verifiers/dynamic/norm_stats.npz` | ~2 KB |
| `semantic_evaluator` | Semantic text/motion match evaluator (text-motion matching + motion/text decomposition nets), packaged as a tarball | `MotionGPT/deps/t2m_custom36_combinedv2/custom36/t2m/{text_mot_match,Decomp_SP001_SM001_H512,Comp_v6_KLD01}` | `verifiers/semantic/t2m_custom36_combinedv2/` (unpacked from `t2m_custom36_combinedv2.tar.gz`) | ~3.7 GB |
| `glove` | GloVe word vectors + vocab used by the semantic evaluator's text encoder, packaged as a tarball | `MotionGPT/deps/t2m/glove/` | `glove/` (unpacked from `glove.tar.gz`) | ~20 MB |
| `flan_t5_base` (runtime, not hosted) | Base language model the generator fine-tunes from | loaded from the public HF hub at runtime: `google/flan-t5-base` | n/a — not downloaded by this script | n/a |

Notes:

- **Dynamic verifier norm_stats choice.** The `v5` checkpoint (`runs/v5/checkpoint_last.pt`) was
  trained continuing from `v3` and was never given its own `norm_stats.npz`. Every evaluation run in
  the original tree's `Dynamic_Verifier/HOW_TO_RUN.md` (and `scripts/run_fixed_eval_sensitivity.py`)
  loads `v5/checkpoint_last.pt` together with `runs/v3/norm_stats.npz`
  (`--checkpoint runs/v5/checkpoint_last.pt --norm_stats runs/v3/norm_stats.npz`). We therefore ship
  `v3`'s `norm_stats.npz` as the `dynamic_norm_stats` entry — it is the stats file the `v5` checkpoint
  was actually evaluated and is meant to be used with.
- **Semantic evaluator tree.** Only the three subfolders actually used for evaluation
  (`text_mot_match/`, `Decomp_SP001_SM001_H512/`, `Comp_v6_KLD01/`) under
  `custom36/t2m/` are packaged; nothing else from `deps/t2m_custom36_combinedv2/` is needed.
- **GloVe location.** The original tree keeps GloVe under `MotionGPT/deps/t2m/glove/` (not a
  top-level `deps/glove/`); that is the directory packaged into `glove.tar.gz`.
- **flan-t5-base** is not re-hosted: the generator loads `google/flan-t5-base` directly from the
  public Hugging Face Hub at first run (and HF's local cache keeps it afterward).
- **Excluded on purpose:** SMPL body models (`MotionGPT/deps/smpl_models/`) and Whisper weights are
  *not* part of this manifest — the 36-dim Unitree G1 pipeline does not use SMPL meshes or speech
  transcription, so neither is downloaded or referenced anywhere in this repo.
