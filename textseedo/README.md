# `textseedo/` — shared utilities

This is **not** a model. It is the small shared package that every component imports so
that paths and the motion format are defined in exactly one place.

| File | What it provides |
|---|---|
| `paths.py` | Path resolver. Exposes `REPO_ROOT`, `ASSETS_ROOT` (`$TSD_ASSETS` or `<repo>/assets`), `DATA_ROOT` (`$TSD_DATA` or `<repo>/data`) and helpers `assets(*p)`, `data(*p)`, `repo(*p)`. Importing it also exports `TSD_ASSETS`/`TSD_DATA` into the environment so YAML `${oc.env:TSD_ASSETS}` interpolation resolves identically. |
| `motion_format.py` | The 36-dim motion layout constants: `NFEATS=36`, `ROOT_POS_SLICE`, `ROOT_QUAT_SLICE`, `JOINT_SLICE`, the 29 `JOINT_NAMES` (Unitree G1 order), and `FPS=50`. |

Why it exists: the original code had `/home/jianuo/...` and `/data/...` hardcoded everywhere.
Here, no absolute path is baked in — checkpoints resolve under `ASSETS_ROOT` and datasets under
`DATA_ROOT`, both overridable with the `TSD_ASSETS` / `TSD_DATA` environment variables.

```python
from textseedo.paths import assets, data
assets("tokenizer/checkpoint_epoch_95.pt")   # -> <assets>/tokenizer/checkpoint_epoch_95.pt
data("CustomCombined/new_joint_vecs")        # -> <data>/CustomCombined/new_joint_vecs
```
