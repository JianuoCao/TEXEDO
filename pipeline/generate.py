"""Generate N candidate motions for a text prompt.

Thin orchestrator over the generator's ``generator/demo.py`` (which owns model build,
checkpoint load, and FSQ decode). It writes the prompt to a temporary prompts file and
runs demo.py from the ``generator/`` directory (so demo's ``./configs`` relative paths
resolve). Each candidate is written as a raw ``(T, 36)`` ``.npy`` that ``pipeline/score.py``
consumes, plus a matching ``.csv``/``.txt``.

    python -m pipeline.generate --num-samples 8 \
        --prompt "a person waves with the right hand" --out-dir candidates/

For full control (decoding params, multiple prompts from a file), call
``generator/demo.py`` directly — see generator/README.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = REPO_ROOT / "generator"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate candidate motions via generator/demo.py")
    p.add_argument("--prompt", required=True, help="Text prompt to generate motions for")
    p.add_argument("--num-samples", type=int, default=8, help="N candidates for the prompt")
    p.add_argument("--out-dir", required=True, help="Directory to write candidate .npy/.csv into")
    p.add_argument("--cfg", default="configs/config_fsq_multitask.yaml")
    p.add_argument("--cfg-assets", default="configs/assets.yaml")
    p.add_argument("--checkpoint", default=None, help="Override generator checkpoint")
    p.add_argument("--temperature", type=float, default=0.9)
    args = p.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # demo.py --example reads one prompt per line and emits one sample per line; we put the
    # single prompt on one line and ask for --num_samples candidates of it.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(args.prompt + "\n")
        example_file = fh.name

    cmd = [sys.executable, "demo.py",
           "--task", "t2m",
           "--example", example_file,
           "--num_samples", str(args.num_samples),
           "--out_dir", str(out_dir),
           "--exact_out_dir",
           "--cfg", args.cfg,
           "--cfg_assets", args.cfg_assets,
           "--temperature", str(args.temperature)]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]

    print(f"[generate] running in {GENERATOR_DIR}:\n  {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(GENERATOR_DIR))
    if rc == 0:
        n = len(list(out_dir.glob("*.npy")))
        print(f"[generate] {n} candidate .npy files in {out_dir}")
        print(f"[generate] next:  python -m pipeline.score --motion-dir {out_dir} "
              f"--caption \"{args.prompt}\" --output {out_dir}/scores.csv")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
