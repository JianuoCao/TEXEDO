"""Lightweight visualization for 36-dim G1 motions.

Renders a quick **matplotlib summary** (root trajectory + joint-angle curves) for a
``(T, 36)`` motion stored as ``.npy`` (or a 36/37-column ``.csv``). This is a
sanity-check view, not a photoreal render: rendering the Unitree G1 mesh requires the
robot's URDF/MJCF + a physics/render engine (mujoco), which are out of scope for this
code-only release.

    python scripts/visualize_npz.py --input motion.npy --output-dir viz/
    python scripts/visualize_npz.py --input-dir samples/ --output-dir viz/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from textseedo.motion_format import (
    ROOT_POS_SLICE, JOINT_SLICE, JOINT_NAMES, FPS,
)


def _load_motion(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",", skiprows=0)
        if arr.shape[1] >= 37:  # some CSVs carry a leading frame index
            arr = arr[:, -36:]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return np.asarray(arr, dtype=np.float32)


def visualize(motion: np.ndarray, out_png: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = motion.shape[0]
    t = np.arange(T) / FPS
    root = motion[:, ROOT_POS_SLICE]
    joints = motion[:, JOINT_SLICE]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title or out_png.stem)

    ax1 = fig.add_subplot(2, 2, 1)
    for i, lbl in enumerate("xyz"):
        ax1.plot(t, root[:, i], label=f"root {lbl}")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position (m)"); ax1.legend(); ax1.set_title("Root position")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(root[:, 0], root[:, 1])
    ax2.scatter([root[0, 0]], [root[0, 1]], c="g", label="start")
    ax2.scatter([root[-1, 0]], [root[-1, 1]], c="r", label="end")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.axis("equal"); ax2.legend(); ax2.set_title("Root path (top-down)")

    ax3 = fig.add_subplot(2, 1, 2)
    im = ax3.imshow(joints.T, aspect="auto", origin="lower",
                    extent=[0, t[-1] if T > 1 else 1, 0, len(JOINT_NAMES)], cmap="twilight")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("joint index"); ax3.set_title("Joint angles (29 DoF)")
    fig.colorbar(im, ax=ax3, label="rad")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize 36-dim G1 motions (matplotlib summary)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="Single .npy/.csv motion")
    g.add_argument("--input-dir", type=Path, help="Directory of motions")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    if args.input:
        files = [args.input]
    else:
        files = sorted(list(args.input_dir.glob("*.npy")) + list(args.input_dir.glob("*.csv")))
    if not files:
        raise SystemExit("No motion files found")

    for f in files:
        out_png = args.output_dir / f"{f.stem}.png"
        visualize(_load_motion(f), out_png, title=f.stem)
        print(f"  {f.name} -> {out_png}")
    print(f"Done: {len(files)} file(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
