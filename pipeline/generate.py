"""Generate N candidate motions for a text prompt.

This is a thin orchestrator over the generator's ``generator/demo.py`` (which owns
model build + FSQ decode). It runs the demo as a subprocess from the ``generator/``
directory so that the demo's ``./configs`` relative paths resolve, then leaves the
generated candidates for ``pipeline/score.py`` to consume.

    python -m pipeline.generate --task t2m --num-samples 8 \
        --prompt "a person waves with the right hand"

For full control over decoding / checkpoints, call ``generator/demo.py`` directly
(see generator/README.md). This wrapper exists for the quick best-of-N path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = REPO_ROOT / "generator"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate candidate motions via generator/demo.py")
    p.add_argument("--task", default="t2m", choices=["t2m", "m2t", "pred"])
    p.add_argument("--num-samples", type=int, default=8, help="N candidates")
    p.add_argument("--prompt", default=None, help="Text prompt (t2m)")
    p.add_argument("--cfg", default="configs/config_fsq_multitask.yaml")
    p.add_argument("--cfg-assets", default="configs/assets.yaml")
    p.add_argument("--checkpoint", default=None, help="Override generator checkpoint")
    p.add_argument("--out-dir", default=None, help="Where demo.py writes candidates")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded verbatim to generator/demo.py")
    args = p.parse_args()

    cmd = [sys.executable, "demo.py",
           "--task", args.task,
           "--num_samples", str(args.num_samples),
           "--cfg", args.cfg,
           "--cfg_assets", args.cfg_assets]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.out_dir:
        cmd += ["--out_dir", args.out_dir]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    cmd += list(args.extra)

    print(f"[generate] running in {GENERATOR_DIR}:\n  {' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd, cwd=str(GENERATOR_DIR)))


if __name__ == "__main__":
    main()
