# Shared contract for all release-packaging tasks

You are packaging one component of the **text-see-do** release repo at
`/home/jianuo/projects/text-see-do/`. Read this before your task file.

## Hard rules (non-negotiable)
1. **READ-ONLY sources.** `/home/jianuo/projects/GenMimic/` and `/home/jianuo/projects/MotionGPT/`
   (and anything outside `text-see-do/`) are read-only. **Copy** files out — never edit, move, or
   delete anything in the source trees.
2. **Write only inside your assigned target subdirectory** of `text-see-do/`. Do not touch other
   components' directories or the shared `textseedo/` package.
3. **Training + inference only.** Do NOT copy evaluation/metric scripts, experiment/figure scripts,
   plotting, or analysis utilities. (Specifics in your task file.)

## Path-cleanup rules
Every hardcoded absolute path (`/home/jianuo/...`, `/data/...`) must be removed from copied files.
Replace with one of:
- **In YAML configs:** OmegaConf env interpolation — `${oc.env:TSD_ASSETS}/...` for checkpoints,
  `${oc.env:TSD_DATA}/...` for datasets.
- **In Python:** `from textseedo.paths import ASSETS_ROOT, DATA_ROOT, assets, data` and build paths
  from those. For CLI scripts, expose the path as an `argparse` argument whose **default** derives
  from `ASSETS_ROOT`/`DATA_ROOT` (never a literal user path).
- The repo root resolver is `text-see-do/textseedo/paths.py`. It exports `REPO_ROOT`, `ASSETS_ROOT`
  (`$TSD_ASSETS` | `<repo>/assets`), `DATA_ROOT` (`$TSD_DATA` | `<repo>/data`) plus helpers
  `assets(*parts)`, `data(*parts)`, `repo(*parts)`.

## Asset layout (where the resolver points)
Checkpoints land under `${TSD_ASSETS}`; datasets under `${TSD_DATA}`. Canonical sub-paths:
- FSQ ckpt:           `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`
- FSQ norm stats:     `${TSD_ASSETS}/tokenizer/fsq_motion_stats_combined.npz`
- Generator ckpt:     `${TSD_ASSETS}/generator/epoch=489.ckpt`
- Dynamic verifier:   `${TSD_ASSETS}/verifiers/dynamic/checkpoint_last.pt` (+ `norm_stats.npz`)
- Semantic evaluator: `${TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2/...`
- GloVe:              `${TSD_ASSETS}/glove/`
- Prepared dataset:   `${TSD_DATA}/CustomCombined/{new_joint_vecs,texts,TOKENS_FSQ,train|val|test.txt}`
- TEXEDO download:    `${TSD_DATA}/texedo/` (raw HF snapshot)

## Deliverables for every task
- Copied + cleaned source files in your target subdir.
- A short `README.md` in your target subdir: what the component is, how to train, how to run
  inference, expected checkpoint paths.
- A self-contained smoke check: a `if __name__ == "__main__"`-runnable snippet or a one-line import
  that proves the module imports without the absolute paths. State the exact command in your final
  report.
- In your final report: list every file you created, every absolute path you removed (old → new),
  and anything you intentionally dropped.

## 36-dim motion format
`motion[:, 0:3]`=root pos, `[:,3:7]`=root quat (wxyz), `[:,7:36]`=29 G1 joint values.
Constants available in `textseedo.motion_format`.
