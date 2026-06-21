# Pipeline — best-of-N inference

Glue that turns a text prompt into a selected motion using the four components.

```
text prompt ──> generate.py ──> N candidate motions (.npy, 36-dim)
                                      │
                                      ▼
                                  score.py ──> per-candidate CSV
                                      │          R_dyn_* (dynamic verifier reward)
                                      │          R_sem_matching_dist (semantic verifier)
                                      ▼
                            select_best_of_n.py ──> best candidate
```

## Quick run
```bash
# 1) Generate N candidates (thin wrapper over generator/demo.py)
python -m pipeline.generate --task t2m --num-samples 8 --prompt "a person waves"

# 2) Score them with both verifiers
python -m pipeline.score --motion-dir <candidates_dir> \
    --caption "a person waves" --output scores.csv

# 3) Pick the best (alpha weights dynamic reward, beta weights semantic match)
python -m pipeline.select_best_of_n --scores scores.csv \
    --alpha 1.0 --beta 1.0 --motion-dir <candidates_dir> --copy-best-to best/
```

## Programmatic API
```python
from pipeline.score import DynamicScorer, SemanticScorer
dyn = DynamicScorer()                  # -> {"reward_hat", "success_prob", "dynamics_hat", "progress_hat"}
sem = SemanticScorer()                 # -> matching distance (lower = better)
r = dyn.score(motion_36d)
d = sem.score(motion_36d, "a person waves")
```

All checkpoints resolve from `${TSD_ASSETS}` (see `scripts/download_assets.py`).
The dynamic reward is in [0, 1] (higher = more physically plausible); the semantic
score is an L2 text–motion distance (lower = better match), z-normalized inside
`select_best_of_n.py` so the two are combinable.
