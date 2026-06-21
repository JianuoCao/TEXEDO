# FSQ Motion Tokenizer

Finite Scalar Quantization (FSQ) tokenizer that discretizes 36-dim Unitree G1
motion (`root_pos(3) + root_quat_wxyz(4) + joint_pos(29)`, see
`textseedo.motion_format`) into a single stream of discrete tokens. Used as
Stage-1 of the TEXEDO pipeline; the generator (`generator/`) is trained
on top of the resulting token sequences.

FSQ reference: Mentzer et al., *"Finite Scalar Quantization: VQ-VAE Made
Simple"* (2023), https://arxiv.org/abs/2309.15505.

## Files

| File | Purpose |
|---|---|
| `fsq_arch.py` | `FSQVae` model: Conv1D encoder → FSQ quantizer → Conv1D decoder. |
| `fsq_dataloader.py` | `SlidingWindowDataset` — lazy sliding windows over a folder of `.npz` motions. |
| `fsq_train.py` | Training loop (DDP, AMP, WandB). |
| `fsq_adapter.py` | `load_fsq_adapter(checkpoint_path)` — inference-ready wrapper with `encode`/`decode`. |
| `encode_motion_tokens.py` | Batch-encode a folder of `.npz` motions to per-file token arrays (`.npy`). Used by data prep. |
| `precompute_fsq_stats.py` | Computes the mean/std normalization stats file consumed by `FSQVae`. |
| `configs/fsq_combined.yaml` | Active training config. |
| `normalization/fsq_motion_stats_combined.npz` | Precomputed normalization stats (ships in-repo; small file). |

Codebook size = product of `fsq_levels` (default `[3,3,3,3,3,2,2,2,2,2]` →
3⁵·2⁵ = 7,776 codes). Checkpoint dict keys: `model_state_dict`, `config`,
`epoch`; `load_fsq_adapter` strips a `module.` prefix automatically (DDP
checkpoints).

## Paths

Everything resolves through `textseedo.paths` (Python) or
`${oc.env:TSD_ASSETS}` / `${oc.env:TSD_DATA}` (YAML configs) — see
`docs/tasks/CONTRACT.md`. No absolute user paths are baked into this
directory. Expected locations:

- Checkpoint: `${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt`
- Norm stats (downloaded copy, optional): `${TSD_ASSETS}/tokenizer/fsq_motion_stats_combined.npz`
  — the config instead points at the copy checked into this repo at
  `tokenizer/normalization/fsq_motion_stats_combined.npz` (same file; ships
  in-repo since it's tiny, so no download is required for inference).
- Training data: `${TSD_DATA}/CustomCombined/new_joint_vecs` (prepared by
  the data-prep task; see `docs/tasks/task5_data.md`).

`fsq_combined.yaml`'s `data.data_folder` and `normalization_stats_file` use
the conventions above; run scripts from the repo root so the in-repo
relative stats path resolves.

## Train

```bash
cd TEXEDO
python tokenizer/fsq_train.py --config tokenizer/configs/fsq_combined.yaml
```

Useful flags: `--resume latest|<path>`, `--reset-lr <lr>`, `--data-folder
<path>` (override `data.data_folder`), `--no-wandb`.

Note: the original internal training loop optionally rendered validation
reconstructions to mp4 via an external EGL-based visualizer script. That
script is not part of this release; `fsq_train.py` still dumps
`recon_motion.csv` / `orig_motion.csv` per validation epoch under
`<log_dir>/visualizations/epoch_<N>/` but stops short of rendering video
(see the `TODO` in `FSQTrainer._generate_validation_video`).

## Precompute normalization stats (only needed to retrain from scratch / on new data)

```bash
python tokenizer/precompute_fsq_stats.py \
    --data-folder ${TSD_DATA}/CustomCombined/new_joint_vecs \
    --output-file tokenizer/normalization/fsq_motion_stats_combined.npz
```

## Inference / encode tokens

```python
from tokenizer.fsq_adapter import load_fsq_adapter

adapter = load_fsq_adapter("path/to/checkpoint_epoch_95.pt", device="cuda")
tokens, lengths = adapter.encode(features)   # features: (B, T, 36)
recon = adapter.decode(tokens)               # (B, T, 36)
```

Batch-encode a folder of `.npz` motions (defaults derive from
`${TSD_ASSETS}` / `${TSD_DATA}`, override with flags):

```bash
python tokenizer/encode_motion_tokens.py \
    --checkpoint ${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt \
    --data-folder ${TSD_DATA}/CustomCombined/new_joint_vecs \
    --output-dir ${TSD_DATA}/CustomCombined/TOKENS_FSQ \
    --device cuda
```

## Dependency

`vector-quantize-pytorch` (provides `FSQ`) — already pinned in the repo
`requirements.txt`.

## Dropped from this release

Train + inference only, per `docs/tasks/CONTRACT.md`. Not copied from the
original source tree:
- `fsq_inference.py` (evaluation/metrics script)
- `compare_then_finish.py` (experiment orchestrator)
- all `fsq_config_*.yaml` variants other than `fsq_config_combined.yaml`
- `scripts/convert_bones_csv_to_amass_csv.py`
- `checkpoints/`, `logs*/`, `wandb/` (run artifacts)
