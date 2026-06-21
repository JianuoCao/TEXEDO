# Text-See-Do

A reproducible **text → motion** pipeline for the Unitree G1 humanoid. A language model turns
text into discrete motion tokens, an FSQ tokenizer decodes them to 36-dim motion, and two
verifiers (physical plausibility + text–motion match) rank candidates for best-of-N selection.

```
            ┌─────────────┐    tokens     ┌──────────────┐  36-dim motion
  text ───▶ │  generator  │ ─────────────▶│  FSQ decode  │ ───────────────┐
            │  (flan-t5)  │               │ (tokenizer)  │                │
            └─────────────┘               └──────────────┘                ▼
                                                                ┌────────────────────┐
                                                                │  dynamic verifier  │ reward
                                                                │  semantic verifier │ match
                                                                └────────────────────┘
                                                                          │ best-of-N
                                                                          ▼ selected motion
```

## Repository layout
| Dir | What | Train entry | Inference entry |
|---|---|---|---|
| [`textseedo/`](textseedo/) | shared path resolver + 36-dim format (no model) | — | — |
| [`tokenizer/`](tokenizer/) | FSQ motion tokenizer (36-dim ↔ tokens) | `fsq_train.py` | `encode_motion_tokens.py`, `fsq_adapter.py` |
| [`generator/`](generator/) | flan-t5 LM over motion tokens (multitask) | `train.py` | `demo.py` |
| [`verifiers/dynamic/`](verifiers/dynamic/) | physical-plausibility reward model | `train_verifier.py` | `predict_rewards.py` |
| [`verifiers/semantic/`](verifiers/semantic/) | text–motion matching evaluator | `train_evaluator.py` | `inference.py` |
| [`pipeline/`](pipeline/) | generate → score → best-of-N | — | `generate / score / select_best_of_n` |
| `scripts/` | asset staging/download + visualization | — | — |

---

## 1. Install
```bash
git clone <this-repo> text-see-do && cd text-see-do
conda env create -f environment.yml && conda activate textseedo   # or: pip install -r requirements.txt
pip install -e .
```
Python 3.10, PyTorch 2.5.1, CUDA 12.4.

Paths resolve from two environment variables (defaults shown):
```bash
export TSD_ASSETS=$PWD/assets    # checkpoints live here
export TSD_DATA=$PWD/data        # datasets live here
```
You can skip the exports to use the defaults (`./assets`, `./data`).

---

## 2. Get the checkpoints

**Which checkpoints, and where they go** (everything lives under `$TSD_ASSETS`, i.e. `./assets`):

| Checkpoint | Destination under `assets/` | Needed for | Size |
|---|---|---|---|
| FSQ tokenizer | `tokenizer/checkpoint_epoch_95.pt` | tokenize, generator (frozen), data prep | 225 MB |
| FSQ norm stats | `tokenizer/fsq_motion_stats_combined.npz` | tokenizer | 2 KB |
| Generator LM | `generator/epoch=489.ckpt` | text→motion inference | 3.4 GB |
| Dynamic verifier | `verifiers/dynamic/checkpoint_last.pt` | dynamic scoring | 42 MB |
| Dynamic norm stats | `verifiers/dynamic/norm_stats.npz` | dynamic scoring | 2 KB |
| Semantic evaluator | `verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m/text_mot_match/model/finest.tar` | semantic scoring | ~20 MB |
| Semantic mean/std | `verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m/Comp_v6_KLD01/meta/{mean,std}.npy` | semantic scoring | small |
| GloVe vocab | `glove/our_vab_data.npy`, `glove/our_vab_idx.pkl`, `glove/our_vab_words.pkl` | semantic text encoding | small |
| flan-t5-base | *(not stored — pulled from HF `google/flan-t5-base` at runtime)* | generator LM | auto |

Fetch them from the Hugging Face Hub:
```bash
# set checkpoints.hf_repo in configs/paths.yaml first (see docs/UPLOAD.md), then:
python scripts/download_assets.py --dry-run     # show what will be fetched (no network)
python scripts/download_assets.py               # download into assets/
```

Verify placement any time:
```bash
find assets -maxdepth 4 -type f | sort
```
> The dataset is public, but the checkpoints must be hosted in a model repo you control — see
> [docs/UPLOAD.md](docs/UPLOAD.md) to publish them and set `checkpoints.hf_repo`.

---

## 3. Prepare the dataset
Downloads the public [`JianuoCao/TEXEDO`](https://huggingface.co/datasets/JianuoCao/TEXEDO)
dataset, flattens it to the training layout, and regenerates FSQ tokens (needs the FSQ checkpoint
from step 2).
```bash
python generator/scripts/prepare_dataset.py                 # full dataset
python generator/scripts/prepare_dataset.py --limit 50      # quick smoke subset
```
Result: `data/CustomCombined/{new_joint_vecs/, texts/, TOKENS_FSQ/, train|val|test.txt, template_*.json}`.

> Inference (step 4) does **not** need this — only training (step 5) does.

---

## 4. Run inference (text → motion, best-of-N)
Needs: generator + FSQ + both verifiers + GloVe staged in step 2.
```bash
# (a) generate N candidates for a prompt
python -m pipeline.generate --prompt "a person waves with the right hand" \
    --num-samples 8 --out-dir candidates/

# (b) score every candidate with both verifiers
python -m pipeline.score --motion-dir candidates/ \
    --caption "a person waves with the right hand" --output scores.csv

# (c) pick the best and copy it out
python -m pipeline.select_best_of_n --scores scores.csv \
    --motion-dir candidates/ --copy-best-to best/

# (d) visualize the winner (matplotlib summary PNG)
python scripts/visualize_npz.py --input-dir best/ --output-dir viz/
```

Just want generation, no verifiers? Run the generator directly:
```bash
cd generator
python demo.py --task t2m --num_samples 5 \
    --cfg configs/config_fsq_multitask.yaml --cfg_assets configs/assets.yaml
cd ..
```

Score a motion programmatically:
```python
from pipeline.score import DynamicScorer, SemanticScorer
import numpy as np
motion = np.load("best/<id>.npy")                       # (T, 36)
print(DynamicScorer().score(motion))                    # {'reward_hat': ..., ...}  higher = better
print(SemanticScorer().score(motion, "a person waves")) # L2 distance, lower = better
```

---

## 5. Run training
Run stages in order. Each writes its own checkpoint; later stages consume earlier ones.

**5.1 FSQ tokenizer** (optional — a trained checkpoint ships in step 2):
```bash
python tokenizer/precompute_fsq_stats.py        # optional: recompute normalization stats
python tokenizer/fsq_train.py --config tokenizer/configs/fsq_combined.yaml
python tokenizer/encode_motion_tokens.py \
    --checkpoint assets/tokenizer/checkpoint_epoch_95.pt \
    --data-folder data/CustomCombined/new_joint_vecs \
    --output-dir  data/CustomCombined/TOKENS_FSQ
```

**5.2 Generator** (flan-t5 multitask LM; loads the frozen FSQ tokenizer):
```bash
cd generator
python train.py --cfg configs/config_fsq_multitask.yaml \
    --cfg_assets configs/assets.yaml --nodebug
cd ..
```

**5.3 Dynamic verifier** (physical-plausibility reward):
```bash
python verifiers/dynamic/train_verifier.py \
    --train_csv <train_labels.csv> --eval_csv <eval_labels.csv> \
    --train_motion_dir <dir_of_{id}.npy> --eval_motion_dir <dir> \
    --save_dir assets/verifiers/dynamic
```

**5.4 Semantic verifier** (text–motion matching):
```bash
python verifiers/semantic/train_evaluator.py \
    --config verifiers/semantic/configs/evaluator.yaml --step all
```

---

## Documentation
- [docs/FORMAT.md](docs/FORMAT.md) — the 36-dim motion representation + G1 joint order.
- [docs/DATA.md](docs/DATA.md) — TEXEDO dataset + prepared training layout.
- [docs/MODELS.md](docs/MODELS.md) / [docs/UPLOAD.md](docs/UPLOAD.md) — checkpoint inventory and HF hosting.
- [docs/REPRODUCE.md](docs/REPRODUCE.md) — condensed end-to-end recipe.

## Notes
- **FSQ-only** release (the legacy VQVAE tokenizer was removed).
- flan-t5-base is pulled from the public HF hub at runtime; SMPL meshes / whisper are not used
  (visualization renders the G1 from the 36-dim representation).

## License
MIT (code) — see [LICENSE](LICENSE). Third-party data (TEXEDO ← AMASS/CLAW) and flan-t5 weights
keep their own licenses.
