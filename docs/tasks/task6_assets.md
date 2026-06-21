# TASK-6 — Asset hosting & download

Read `docs/tasks/CONTRACT.md` first.

## Objective
Provide a single script to fetch all checkpoints (and the dataset) from the Hugging Face Hub into the
repo's `assets/`/`data/` roots, plus the manifest and upload instructions.

## Target
- `text-see-do/scripts/download_assets.py`
- `text-see-do/docs/MODELS.md`
- `text-see-do/docs/UPLOAD.md`
- (edit) `text-see-do/configs/paths.yaml` — finalize the manifest (skeleton already exists).

## `download_assets.py` requirements
A self-contained CLI (`def main()`):
- Read `configs/paths.yaml` (OmegaConf/yaml).
- For the **dataset** entry: `huggingface_hub.snapshot_download("JianuoCao/TEXEDO", repo_type="dataset",
  local_dir=DATA_ROOT/"texedo")`.
- For **checkpoints**: from the configured model `hf_repo`, `hf_hub_download` each file into
  `ASSETS_ROOT/<local>`. If an entry has `unpack: true` (`.tar.gz`), extract it after download.
- Flags: `--dry-run` (print what would be fetched, resolve every manifest entry, fetch nothing),
  `--only fsq_tokenizer,generator,...` (subset), `--assets-root`, `--data-root` (default from
  `textseedo.paths`).
- Use `textseedo.paths.ASSETS_ROOT` / `DATA_ROOT` for destinations. No absolute user paths.
- Gracefully handle `hf_repo: TODO_USER_MODEL_REPO` — in `--dry-run` show the intended layout; in a
  real run, error with a clear message pointing to `docs/UPLOAD.md`.

## `configs/paths.yaml` (finalize)
Manifest skeleton already lists: `fsq_tokenizer`, `fsq_norm_stats`, `generator`, `dynamic_verifier`,
`dynamic_norm_stats`, `semantic_evaluator` (unpack), `glove` (unpack), and the runtime
`google/flan-t5-base`. Adjust remote/local names to match what `download_assets.py` expects.

## `docs/MODELS.md`
Table of every checkpoint: logical name, what it is, source file in the original tree, destination
under `${TSD_ASSETS}`, approx size. List the originals (READ-ONLY, do not move):
- FSQ: `GenMimic/stage1-tokenize/fsq/checkpoints/checkpoints-fsq-combined/checkpoint_epoch_95.pt`
  (+ `normalization/fsq_motion_stats_combined.npz`).
- Generator: `MotionGPT/experiments/mgpt/CustomCombined_Stage2_FSQ_MultiTask-4-30/checkpoints/epoch=489.ckpt`.
- Dynamic: `MotionGPT/Dynamic_Verifier/runs/v5/checkpoint_last.pt` (+ `runs/v3/norm_stats.npz` — confirm
  which norm_stats matches v5; document the choice).
- Semantic: `MotionGPT/deps/t2m_custom36_combinedv2/` (the `custom36/t2m/{text_mot_match,Decomp_*,Comp_v6_KLD01}` tree).
- GloVe: `MotionGPT/deps/glove/` (or `deps/t2m/glove/`).
- Note flan-t5-base loads from HF `google/flan-t5-base` at runtime (not hosted).
- Note: SMPL/whisper are intentionally excluded (not used by the 36-dim G1 pipeline).

## `docs/UPLOAD.md`
Exact steps for the user to push checkpoints to their HF model repo: which local files map to which
remote paths in the manifest, the `huggingface-cli upload` / `hf upload-large-folder` commands, and a
reminder to set `checkpoints.hf_repo` in `configs/paths.yaml` afterward.

## Smoke check
`cd text-see-do && python scripts/download_assets.py --dry-run` resolves the dataset + every checkpoint
entry and prints destinations, fetching nothing. `grep -rn '/home/jianuo\|/data/' scripts/download_assets.py`
returns nothing (the original-tree source paths belong only in `docs/MODELS.md`/`UPLOAD.md` as references).

## Acceptance
- `--dry-run` works and lists all assets; manifest is consistent with the script; MODELS.md + UPLOAD.md
  give the user a precise upload recipe; download code uses repo-relative roots.
