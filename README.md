# Text-See-Do

A reproducible **text-to-motion** pipeline for the Unitree G1 humanoid. Text is generated
into discrete motion tokens by a language model, decoded to motion by an FSQ tokenizer, and
candidate motions are ranked by two verifiers (physical plausibility + text–motion match) for
best-of-N selection.

```
            ┌─────────────┐     tokens      ┌──────────────┐   36-dim motion
  text ───▶ │  generator  │ ───────────────▶│  FSQ decode  │ ─────────────────┐
            │  (flan-t5)  │                 │ (tokenizer)  │                  │
            └─────────────┘                 └──────────────┘                  ▼
                                                                   ┌────────────────────┐
                                                                   │  dynamic verifier  │  reward
                                                                   │  semantic verifier │  match
                                                                   └────────────────────┘
                                                                             │
                                                                       best-of-N select
```

## Components
| Dir | What | Train | Inference |
|---|---|---|---|
| [`tokenizer/`](tokenizer/) | FSQ motion tokenizer (36-dim ↔ tokens) | `fsq_train.py` | `encode_motion_tokens.py`, `fsq_adapter.py` |
| [`generator/`](generator/) | flan-t5 LM over motion tokens (multitask) | `train.py` | `demo.py` |
| [`verifiers/dynamic/`](verifiers/dynamic/) | physical-plausibility reward model | `train_verifier.py` | `predict_rewards.py` |
| [`verifiers/semantic/`](verifiers/semantic/) | text–motion matching evaluator | `train_evaluator.py` | `inference.py` |
| [`pipeline/`](pipeline/) | generate → score → best-of-N | — | `generate / score / select_best_of_n` |

## Install
```bash
conda env create -f environment.yml && conda activate textseedo   # or: pip install -r requirements.txt
pip install -e .
cp .env.example .env        # optional: point TSD_ASSETS / TSD_DATA elsewhere
```
Python 3.10, PyTorch 2.5.1, CUDA 12.4.

## Get assets & data
```bash
python scripts/download_assets.py --dry-run     # show what will be fetched
python scripts/download_assets.py               # checkpoints -> assets/, TEXEDO -> data/
```
The dataset is public ([`JianuoCao/TEXEDO`](https://huggingface.co/datasets/JianuoCao/TEXEDO)).
Checkpoints live in a Hugging Face model repo — set `checkpoints.hf_repo` in
[`configs/paths.yaml`](configs/paths.yaml) (see [docs/MODELS.md](docs/MODELS.md), [docs/UPLOAD.md](docs/UPLOAD.md)).

## Prepare training data
```bash
python generator/scripts/prepare_dataset.py     # download + flatten TEXEDO + regenerate FSQ tokens
```
Produces `${TSD_DATA}/CustomCombined/{new_joint_vecs,texts,TOKENS_FSQ,*.txt,template_*.json}`.

## Inference (best-of-N)
```bash
python -m pipeline.generate --task t2m --num-samples 8 --prompt "a person waves"
python -m pipeline.score    --motion-dir <dir> --caption "a person waves" --output scores.csv
python -m pipeline.select_best_of_n --scores scores.csv --motion-dir <dir> --copy-best-to best/
python scripts/visualize_npz.py --input-dir best/ --output-dir viz/
```

## Training
See per-stage commands in [docs/REPRODUCE.md](docs/REPRODUCE.md): FSQ tokenizer → encode tokens →
generator → both verifiers.

## Documentation
- [docs/FORMAT.md](docs/FORMAT.md) — the 36-dim motion representation + G1 joint order.
- [docs/DATA.md](docs/DATA.md) — TEXEDO dataset + the prepared training layout.
- [docs/MODELS.md](docs/MODELS.md) / [docs/UPLOAD.md](docs/UPLOAD.md) — checkpoints and hosting.
- [docs/REPRODUCE.md](docs/REPRODUCE.md) — full end-to-end train + inference recipe.

## Notes
- This release is **FSQ-only** (the legacy VQVAE tokenizer is removed).
- flan-t5-base is pulled from the public HF hub at runtime; SMPL/whisper are not used.

## License
MIT (code) — see [LICENSE](LICENSE). Third-party data (AMASS, CLAW via TEXEDO) and the
flan-t5 weights keep their own licenses.
