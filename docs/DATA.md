# Data

This page documents the `JianuoCao/TEXEDO` dataset, how to fetch it, and how
`generator/scripts/prepare_dataset.py` turns it into the flat `CustomCombined` layout
that the generator and verifiers train on. See `docs/FORMAT.md` for the 36-dim motion
representation itself.

## TEXEDO dataset

TEXEDO is a public text-motion dataset prepared from two sources, AMASS and CLAW
(the CLAW source is normalized to `claw` in the release).

| Split | Samples |
| --- | ---: |
| train | 18,590 |
| validation | 2,324 |
| test | 2,325 |
| **total** | **23,239** |

| Source | Samples |
| --- | ---: |
| AMASS | 9,245 |
| CLAW | 13,994 |

### Layout (as downloaded)

```text
texedo/
  README.md
  train.txt
  val.txt
  test.txt
  data/
    train.jsonl
    validation.jsonl
    test.jsonl
    all.jsonl
  motions/
    amass/{bucket}/{id}.npy
    claw/{bucket}/{id}.npy
  texts/
    amass/{bucket}/{id}.txt
    claw/{bucket}/{id}.txt
  metadata/
    dataset_summary.json
```

Each sample has one `(T, 36)` motion `.npy` file and one text annotation `.txt` file
(see `docs/FORMAT.md`). `{bucket}` is the first three digits of the six-digit `{id}`.

Each row of `data/*.jsonl` carries `id`, `split`, `source`, `motion_path`, `text_path`
(plus `num_frames`, `motion_dim`, `num_texts`, `captions`) — `motion_path`/`text_path`
are the authoritative relative paths to each sample's files, so dataset prep resolves
files through this index instead of reconstructing `{source}/{bucket}` paths by hand.

### Download

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('JianuoCao/TEXEDO', repo_type='dataset', local_dir='data/texedo')
"
```

or simply run `generator/scripts/prepare_dataset.py`, which downloads it for you (see
below). The default destination is `${TSD_DATA}/texedo`, i.e. a `texedo` directory under
the repo's data root if `TSD_DATA` is unset.

## Preparing the training layout

`generator/scripts/prepare_dataset.py` turns a TEXEDO snapshot into the flat
`CustomCombined` layout the generator and verifiers expect:

```bash
cd text-see-do
python generator/scripts/prepare_dataset.py
```

This runs three steps:

1. **Download** — `huggingface_hub.snapshot_download("JianuoCao/TEXEDO", repo_type="dataset")`
   into `${TSD_DATA}/texedo` (or `--texedo-dir`; pass `--skip-download` to reuse an
   existing snapshot).
2. **Flatten** — using the `data/*.jsonl` index, copies
   - `motions/{source}/{bucket}/{id}.npy` -> `CustomCombined/new_joint_vecs/{id}.npy`
   - `texts/{source}/{bucket}/{id}.txt` -> `CustomCombined/texts/{id}.txt`

   and copies `train.txt`/`val.txt`/`test.txt` as-is (TEXEDO's `validation.txt`/HF
   `validation` split maps to the local `val.txt` name).
3. **Regenerate FSQ tokens** — every flattened `new_joint_vecs/{id}.npy` is encoded with
   the frozen FSQ tokenizer into `CustomCombined/TOKENS_FSQ/{id}.npy`, an `int32` array
   of shape `(T // 4,)`. The checkpoint defaults to
   `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt` (override with `--fsq-checkpoint`).

   TEXEDO/`CustomCombined` motions are flat `(T, 36)` arrays, which is exactly the input
   the FSQ adapter's `encode()` expects (as a batch of size 1), so this step calls
   `tokenizer.fsq_adapter.load_fsq_adapter(...).encode(...)` directly rather than going
   through `tokenizer/encode_motion_tokens.py` (which is built around `.npz` inputs with
   separate `body_pos_w`/`body_quat_w`/`joint_pos` arrays — a different on-disk format
   used elsewhere in the tokenizer's training pipeline, not by TEXEDO/CustomCombined).

   This import is deferred until the token-regeneration step actually runs, so the
   script stays importable even before/without the `tokenizer` package installed (e.g.
   `--skip-tokens` runs need nothing from `tokenizer/`).

Flags:

| Flag | Effect |
| --- | --- |
| `--repo-id` | HF dataset repo id (default `JianuoCao/TEXEDO`). |
| `--texedo-dir` | Raw snapshot location (default `${TSD_DATA}/texedo`). |
| `--output-dir` | Flattened layout location (default `${TSD_DATA}/CustomCombined`). |
| `--fsq-checkpoint` | FSQ checkpoint (default `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`). |
| `--skip-download` | Reuse an existing `--texedo-dir` snapshot. |
| `--skip-tokens` | Flatten only; skip FSQ token regeneration. |
| `--limit N` | Only process the first `N` indexed samples — useful for a fast smoke test. |
| `--device` | Device for FSQ encoding (default `cuda`, falls back to `cpu` if unavailable). |

Smoke test (flatten only, no GPU/checkpoint needed):

```bash
python generator/scripts/prepare_dataset.py --limit 8 --skip-tokens
```

## Final `CustomCombined` layout

```text
${TSD_DATA}/CustomCombined/
  new_joint_vecs/{id}.npy   # (T, 36) float32 motion, see docs/FORMAT.md
  texts/{id}.txt            # caption(s) for {id}
  TOKENS_FSQ/{id}.npy       # (T // 4,) int32 FSQ token sequence
  train.txt                 # one id per line
  val.txt
  test.txt
```

This is the layout consumed directly by `generator/` training/inference and by the
verifiers.
