# Motion Format

This page documents the 36-dim motion representation used throughout TEXEDO
(tokenizer, generator, verifiers, pipeline). It mirrors the format documented in the
public `JianuoCao/TEXEDO` dataset card.

The code source of truth is `textseedo/motion_format.py` — import constants from there
instead of hardcoding slice indices or joint names:

```python
from textseedo.motion_format import (
    NFEATS, ROOT_POS_SLICE, ROOT_QUAT_SLICE, JOINT_SLICE, NUM_JOINTS, JOINT_NAMES, FPS,
)
```

## Layout

Every motion is a float array of shape `(T, 36)` (Unitree G1, 50 fps), where `T` is the
number of frames. Both the TEXEDO dataset and the prepared `CustomCombined` training
layout (see `docs/DATA.md`) store motions as flat `.npy` files in this format.

| Feature slice | Size | Description |
| --- | ---: | --- |
| `motion[:, 0:3]` | 3 | root position, `(x, y, z)`, meters |
| `motion[:, 3:7]` | 4 | root quaternion, `(w, x, y, z)`, unit-norm, wxyz convention |
| `motion[:, 7:36]` | 29 | joint positions / joint angles (radians), in the order below |

## Joint order

Joint order for `motion[:, 7:36]` (also `textseedo.motion_format.JOINT_NAMES`):

| Joint index | Joint name |
| ---: | --- |
| 0 | `left_hip_pitch_joint` |
| 1 | `right_hip_pitch_joint` |
| 2 | `waist_yaw_joint` |
| 3 | `left_hip_roll_joint` |
| 4 | `right_hip_roll_joint` |
| 5 | `waist_roll_joint` |
| 6 | `left_hip_yaw_joint` |
| 7 | `right_hip_yaw_joint` |
| 8 | `waist_pitch_joint` |
| 9 | `left_knee_joint` |
| 10 | `right_knee_joint` |
| 11 | `left_shoulder_pitch_joint` |
| 12 | `right_shoulder_pitch_joint` |
| 13 | `left_ankle_pitch_joint` |
| 14 | `right_ankle_pitch_joint` |
| 15 | `left_shoulder_roll_joint` |
| 16 | `right_shoulder_roll_joint` |
| 17 | `left_ankle_roll_joint` |
| 18 | `right_ankle_roll_joint` |
| 19 | `left_shoulder_yaw_joint` |
| 20 | `right_shoulder_yaw_joint` |
| 21 | `left_elbow_joint` |
| 22 | `right_elbow_joint` |
| 23 | `left_wrist_roll_joint` |
| 24 | `right_wrist_roll_joint` |
| 25 | `left_wrist_pitch_joint` |
| 26 | `right_wrist_pitch_joint` |
| 27 | `left_wrist_yaw_joint` |
| 28 | `right_wrist_yaw_joint` |

## Discrete tokens

The FSQ tokenizer (`tokenizer/`) maps a `(T, 36)` motion to a sequence of `(T // 4,)`
discrete integer codes (4x temporal downsampling), and back. See `tokenizer/README.md`
and `docs/DATA.md` ("Token regeneration") for how `TOKENS_FSQ/{id}.npy` is produced.
