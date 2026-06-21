# Models & assets

Every large file (checkpoints + dataset) lives outside the code repo and is fetched into
`${TSD_ASSETS}` (checkpoints) and `${TSD_DATA}` (datasets) by `scripts/download_assets.py`, using
the manifest in `configs/paths.yaml`.

```bash
python scripts/download_assets.py --dry-run   # see what would be fetched
python scripts/download_assets.py             # fetch everything
python scripts/download_assets.py --only fsq_tokenizer,generator
```

The dataset (`JianuoCao/TEXEDO`) is public. The checkpoints ship in a *separate* Hugging Face
**model** repo that you create — see `docs/UPLOAD.md`. Until `checkpoints.hf_repo` in
`configs/paths.yaml` is set, `--dry-run` still resolves and prints the intended layout; a real
download raises a clear error pointing back to `docs/UPLOAD.md`.

## Checkpoint manifest

| Logical name | What it is | Destination (`${TSD_ASSETS}/...`) | Approx. size |
|---|---|---|---|
| `fsq_tokenizer` | FSQ motion tokenizer (encoder/decoder + FSQ codebook), 36-dim G1 motion | `tokenizer/checkpoint_epoch_95.pt` | ~216 MB |
| `fsq_norm_stats` | Per-channel normalization stats (mean/std) for the FSQ tokenizer | `tokenizer/fsq_motion_stats_combined.npz` | ~2 KB |
| `generator` | Stage-2 text→motion generator: flan-t5-base fine-tuned on FSQ motion tokens (multi-task) | `generator/epoch=489.ckpt` | ~3.2 GB |
| `dynamic_verifier` | Dynamic-feasibility verifier (physical-plausibility scorer), run `v5` | `verifiers/dynamic/checkpoint_last.pt` | ~40 MB |
| `dynamic_norm_stats` | Normalization stats paired with the `v5` checkpoint | `verifiers/dynamic/norm_stats.npz` | ~2 KB |
| `semantic_evaluator` | Text–motion matching evaluator (match net + motion/text decomposition + meta), tarball | `verifiers/semantic/t2m_custom36_combinedv2/` (unpacked) | ~variable |
| `glove` | GloVe vocab (`our_vab_*`) for the semantic text encoder, tarball | `glove/` (unpacked) | ~20 MB |
| `flan_t5_base` (runtime) | Base LM the generator fine-tunes from | loaded from `google/flan-t5-base` at runtime — **not** downloaded here | n/a |

Notes:

- **Dynamic verifier norm_stats.** The `v5` checkpoint continues training from `v3` and has no
  `norm_stats.npz` of its own; it is meant to be evaluated with `v3`'s `norm_stats.npz`. That is the
  file shipped as `dynamic_norm_stats`, and `predict_rewards.py` loads it alongside the `v5` weights.
- **Semantic evaluator tree.** Only the three subfolders used at inference/training
  (`text_mot_match/`, `Decomp_SP001_SM001_H512/`, `Comp_v6_KLD01/`) under `custom36/t2m/` are
  packaged. For inference, only `text_mot_match/model/finest.tar` + `Comp_v6_KLD01/meta/{mean,std}.npy`
  are read.
- **GloVe layout.** The `our_vab_data.npy` / `our_vab_idx.pkl` / `our_vab_words.pkl` files must sit
  directly in `assets/glove/` (the semantic text encoder's `WordVectorizer` reads them there).
- **flan-t5-base** is not re-hosted; it loads from the public HF hub at first run and is cached.
- **Excluded on purpose:** SMPL body models and Whisper weights — the 36-dim Unitree G1 pipeline uses
  neither, so they are not downloaded or referenced anywhere in this repo.
