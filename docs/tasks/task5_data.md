# TASK-5 — Dataset prep + format docs

Read `docs/tasks/CONTRACT.md` first.

## Objective
Make the published HF dataset `JianuoCao/TEXEDO` usable for training: download it, flatten it into the
training layout the generator/verifiers expect, and regenerate FSQ tokens. Plus write the data docs.

## Source of truth (READ-ONLY)
- HF dataset: `JianuoCao/TEXEDO` (public).
- Local mirror to inspect layout: `/home/jianuo/projects/MotionGPT/datasets/TEXEDO_dataset/`
  (has `README.md`, `prepare_texedo_dataset.py`, `motions/{source}/{bucket}/{id}.npy`,
  `texts/{source}/{bucket}/{id}.txt`, `data/*.jsonl`, `train|val|test.txt`).
- Reference for the *target* training layout (READ-ONLY): `/home/jianuo/projects/MotionGPT/datasets/CustomCombined/`
  (`new_joint_vecs/{id}.npy`, `texts/{id}.txt`, `TOKENS_FSQ/{id}.npy`, `train|val|test.txt`).

## Target
- Script: `text-see-do/generator/scripts/prepare_dataset.py`
- Docs: `text-see-do/docs/FORMAT.md`, `text-see-do/docs/DATA.md`

## `prepare_dataset.py` requirements
A self-contained CLI (`def main()`), using `textseedo.paths` for defaults:
1. **Download** `JianuoCao/TEXEDO` via `huggingface_hub.snapshot_download(repo_id, repo_type="dataset")`
   into `${TSD_DATA}/texedo` (or accept `--texedo-dir`).
2. **Flatten** into `${TSD_DATA}/CustomCombined/`:
   - `motions/{source}/{bucket}/{id}.npy` → `new_joint_vecs/{id}.npy`
   - `texts/{source}/{bucket}/{id}.txt`   → `texts/{id}.txt`
   - copy `train.txt`, `val.txt`, `test.txt` as-is.
   Use the `data/*.jsonl` index (fields `id, source, motion_path, text_path`) to resolve files
   robustly rather than reconstructing bucket paths by hand.
3. **Regenerate FSQ tokens** → `TOKENS_FSQ/{id}.npy`, by importing the tokenizer:
   `from tokenizer.encode_motion_tokens import ...` (or call its function), loading the FSQ checkpoint
   from `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`. Provide `--fsq-checkpoint` (default from
   `assets("tokenizer/checkpoint_epoch_95.pt")`). Note: encode reads 36-dim arrays — confirm whether
   `encode_motion_tokens.py` expects `.npz` (body_pos/quat/joint) or flat `(T,36)` `.npy`; TEXEDO stores
   flat `(T,36)` `.npy`, so add a thin adapter if the encoder expects `.npz` keys.
4. Flags: `--limit N` (process a subset for smoke tests), `--skip-tokens`, `--device`.

> Coordinate point: TASK-1 owns `tokenizer/encode_motion_tokens.py`. Assume its public entry takes a
> motion array `(T,36)` and an adapter; if it only accepts `.npz`, wrap by constructing the array and
> calling the FSQ adapter (`tokenizer.fsq_adapter.load_fsq_adapter`) directly. Prefer the adapter path
> (`adapter.encode`) for robustness.

## `docs/FORMAT.md`
Reuse the content of the TEXEDO `README.md` "Motion Format" + joint-order table (36-dim layout, G1
joint names, wxyz quaternion). Cite `textseedo.motion_format` as the code source of truth.

## `docs/DATA.md`
Document: TEXEDO source/splits (train 18,590 / val 2,324 / test 2,325; sources amass + claw), the HF
download command, the flatten + token-regeneration step, and the final `CustomCombined` layout the
trainers consume.

## Smoke check
`cd text-see-do && python -c "import sys; sys.path.insert(0,'.'); import generator.scripts.prepare_dataset as p; print(hasattr(p,'main'))"`
and `grep -rn '/home/jianuo\|/data/' generator/scripts/ docs/FORMAT.md docs/DATA.md` returns nothing.

## Acceptance
- `prepare_dataset.py` runs end-to-end logic (download→flatten→tokenize) with repo-relative defaults
  and a `--limit` smoke path; docs accurately describe TEXEDO + the prepared layout; no absolute paths.
