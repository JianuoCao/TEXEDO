# Semantic Verifier

A BiGRU text/motion matching evaluator that scores how well a generated 36-dim
motion matches a text caption. It is one of two verifiers used for best-of-N
candidate selection in the text-see-do pipeline (the other is the dynamic
verifier in `verifiers/dynamic/`). Same architecture family as the
text-to-motion / MotionGPT evaluator (FID / R-Precision / Matching-Score),
retrained here for the 36-dim Unitree G1 motion format (root pos(3) + root
quat wxyz(4) + 29 joints, see `textseedo.motion_format`). Self-contained
(torch / numpy only).

## Files

| File | Purpose |
|---|---|
| `models.py` | The 4 encoder classes: `MovementConvEncoder`, `MovementConvDecoder`, `TextEncoderBiGRUCo`, `MotionEncoderBiGRUCo`. |
| `train_evaluator.py` | 3-step training (mean/std -> decomp -> contrastive match). Loads `configs/evaluator.yaml`. |
| `inference.py` | `load_evaluator(...)` -> object with `.score(motion_36d, caption) -> float`. |
| `configs/evaluator.yaml` | All paths + hyperparams (replaces the original hardcoded `Config` class). |

## Architecture

Three-stage pipeline, same as the text-to-motion paper:

1. **Mean/std** (`compute_mean_std`): per-dim normalization stats over the
   training motions (after converting root position to root velocity).
2. **Decomposition** (`MovementConvEncoder` + `MovementConvDecoder`,
   `train_decomp`): a 1D-conv autoencoder that downsamples a 64-frame window
   of raw 36-dim motion by 4x into 512-d "movement" latents and reconstructs
   it back (L1 reconstruction + sparsity + smoothness losses).
3. **Contrastive matching** (`TextEncoderBiGRUCo` + `MotionEncoderBiGRUCo`,
   `train_text_motion_match`): with the movement encoder **frozen**, two
   bidirectional-GRU encoders map a caption and a movement-latent sequence
   into a shared 512-d co-embedding space, trained with an all-pairs
   contrastive loss (every (text_i, motion_j) pair in the batch; diagonal =
   positive, off-diagonal = negative, margin-based push/pull).

At inference, `score(motion, caption)` is the L2 distance between the two
512-d co-embeddings — lower means a better text-motion match.

## Config refactor

The original `train_evaluator.py` had a `Config` python class (paths +
hyperparams hardcoded). Here all of that lives in `configs/evaluator.yaml`,
loaded with OmegaConf:

- `data.data_root` -> `${oc.env:TSD_DATA}/CustomCombined`
- `data.glove_dir` -> `${oc.env:TSD_ASSETS}/glove`
- `save_root` -> `${oc.env:TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2`
- Numeric hyperparams (`dim_pose=36`, `max_motion_length=2048`, `unit_length=4`,
  `dim_movement_latent=512`, `dim_text_hidden=512`, `dim_motion_hidden=1024`,
  decomp/match lr & batch size & epochs, etc.) are unchanged from the
  original, just moved under `decomp:` / `match:` / `arch:` / `length:` keys.

`train_evaluator.py`'s `Config` class is now a thin object built from the
loaded YAML (`Config(OmegaConf.load(...))`) instead of a class with literal
defaults — the rest of the training code (`compute_mean_std`, `train_decomp`,
`train_text_motion_match`, the dataset classes) is unchanged from the
original.

## Expected directory layout

`{data_root}` (i.e. `${TSD_DATA}/CustomCombined`) must contain:
```
new_joint_vecs/*.npy   # (T, 36) continuous motion arrays
texts/*.txt            # caption#tokens#from#to lines (HumanML3D format)
train.txt, val.txt     # one motion id per line
```

`{glove_dir}` (i.e. `${TSD_ASSETS}/glove`) must contain
`our_vab_data.npy`, `our_vab_words.pkl`, `our_vab_idx.pkl`.

Training writes under `{save_root}/custom36/t2m/`:
```
Comp_v6_KLD01/meta/{mean.npy,std.npy,run_metadata.json}
Decomp_SP001_SM001_H512/model/{latest,finest,E####}.tar
text_mot_match/model/{latest,finest,E####}.tar   # keys: text_encoder, motion_encoder, movement_encoder
text_mot_match/eval/E###.txt                     # periodic pos/neg distance dumps
```

## Train

```bash
conda activate mgpt
cd text-see-do

python verifiers/semantic/train_evaluator.py --step all --gpu 0
# or one step at a time:
python verifiers/semantic/train_evaluator.py --step mean_std
python verifiers/semantic/train_evaluator.py --step decomp
python verifiers/semantic/train_evaluator.py --step match
```

Useful flags: `--config <path>` (default `configs/evaluator.yaml`),
`--data-root <path>` / `--save-root <path>` (override the YAML), `--gpu`,
`--decomp_epochs`, `--match_epochs`, `--batch_size`, `--wandb`.

Final checkpoint:
`${TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m/text_mot_match/model/finest.tar`
Mean/std meta: `.../Comp_v6_KLD01/meta/{mean.npy,std.npy}`.

## Inference

```python
from verifiers.semantic.inference import load_evaluator

evaluator = load_evaluator(
    checkpoint="${TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m/text_mot_match/model/finest.tar",
    meta_dir="${TSD_ASSETS}/verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m/Comp_v6_KLD01/meta",
    glove_dir="${TSD_ASSETS}/glove",
    device="cuda",
)

distance = evaluator.score(motion_36d, "the person walks forward and waves")
# motion_36d: np.ndarray (T, 36); lower distance = better text-motion match
```

`load_evaluator` builds the three encoders (text/motion/movement), loads
their weights from `finest.tar`, and loads `mean.npy` / `std.npy` plus the
GloVe vocabulary. `score()` internally: converts root position to root
velocity (`root_pos_to_vel_np`, reused from `train_evaluator.py`), crops to a
multiple of `unit_length`, z-normalizes with mean/std, runs the frozen
movement encoder then the motion encoder to get a 512-d motion embedding,
tokenizes + embeds the caption through the text encoder to get a 512-d text
embedding, and returns their L2 distance.

Caption tokenization at inference time is a lightweight self-contained
fallback (lower-case + word split, no spaCy/POS tagging) — out-of-vocabulary
or untagged words fall back to the `unk` embedding / `OTHER` POS tag exactly
like `WordVectorizer` does during training (the original training data's
`word/POS` tokens already had this fallback built in).

## What was intentionally dropped

This package keeps **training + inference only**, per `docs/tasks/CONTRACT.md`:

- `eval_best_of_n.py`, `eval_checkpoint.py`, `eval_generated.py`,
  `eval_kimodo.py` (all evaluation/metrics scripts) — not copied.
- `run_train.sh`, `runs/` (shell wrapper + run artifacts) — not copied;
  use `train_evaluator.py` directly.

## Smoke check

```bash
cd text-see-do && python -c "import sys; sys.path.insert(0,'.'); from verifiers.semantic import models; print('ok')"
```

A repo-wide grep for leaked absolute user paths under this directory (per the
path-cleanup rule in `docs/tasks/CONTRACT.md`) should also return no matches.

`inference.py` also runs standalone as a self-test (random-init encoders + a
tiny fake GloVe vocab, no checkpoint/downloads needed):

```bash
cd text-see-do/verifiers/semantic && python inference.py
```
