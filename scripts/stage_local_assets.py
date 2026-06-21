"""Stage checkpoints from their original local training locations into ${TSD_ASSETS}.

Use this when you have the trained checkpoints on the same machine and have NOT yet
uploaded them to the Hugging Face Hub. It only READS the source files and copies them
into the assets root with the exact layout the code expects. (For public users who
fetch from the Hub instead, use ``scripts/download_assets.py``.)

    python scripts/stage_local_assets.py            # copy everything with default sources
    python scripts/stage_local_assets.py --dry-run  # show what would be copied
    python scripts/stage_local_assets.py --only fsq_tokenizer,generator

Override any source with its flag if your originals live elsewhere.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from textseedo.paths import ASSETS_ROOT

# Default source locations (the training trees). Read-only; never modified.
_GENMIMIC = Path("/home/jianuo/projects/GenMimic/stage1-tokenize/fsq")
_MGPT = Path("/home/jianuo/projects/MotionGPT")
_SEM = _MGPT / "deps/t2m_custom36_combinedv2/custom36/t2m"

DEFAULT_SOURCES = {
    "fsq_tokenizer":   _GENMIMIC / "checkpoints/checkpoints-fsq-combined/checkpoint_epoch_95.pt",
    "fsq_norm_stats":  _GENMIMIC / "normalization/fsq_motion_stats_combined.npz",
    "generator":       _MGPT / "experiments/mgpt/CustomCombined_Stage2_FSQ_MultiTask-4-30/checkpoints/epoch=489.ckpt",
    "dynamic_ckpt":    _MGPT / "Dynamic_Verifier/runs/v5/checkpoint_last.pt",
    "dynamic_norm":    _MGPT / "Dynamic_Verifier/runs/v3/norm_stats.npz",
    "semantic_finest": _SEM / "text_mot_match/model/finest.tar",
    "semantic_mean":   _SEM / "Comp_v6_KLD01/meta/mean.npy",
    "semantic_std":    _SEM / "Comp_v6_KLD01/meta/std.npy",
    "glove_dir":       _MGPT / "deps/t2m/glove",  # contains our_vab_data.npy / our_vab_idx.pkl / our_vab_words.pkl
}

# Destination (relative to ASSETS_ROOT) for each source.
_SEM_DST = "verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m"
DEST = {
    "fsq_tokenizer":   "tokenizer/checkpoint_epoch_95.pt",
    "fsq_norm_stats":  "tokenizer/fsq_motion_stats_combined.npz",
    "generator":       "generator/epoch=489.ckpt",
    "dynamic_ckpt":    "verifiers/dynamic/checkpoint_last.pt",
    "dynamic_norm":    "verifiers/dynamic/norm_stats.npz",
    "semantic_finest": f"{_SEM_DST}/text_mot_match/model/finest.tar",
    "semantic_mean":   f"{_SEM_DST}/Comp_v6_KLD01/meta/mean.npy",
    "semantic_std":    f"{_SEM_DST}/Comp_v6_KLD01/meta/std.npy",
    # glove_dir handled specially: copy the 3 vocab files directly into assets/glove/
}
_GLOVE_FILES = ["our_vab_data.npy", "our_vab_idx.pkl", "our_vab_words.pkl"]

# Group keys -> the "--only" component names.
COMPONENTS = {
    "fsq_tokenizer": ["fsq_tokenizer", "fsq_norm_stats"],
    "generator": ["generator"],
    "dynamic": ["dynamic_ckpt", "dynamic_norm"],
    "semantic": ["semantic_finest", "semantic_mean", "semantic_std"],
    "glove": ["glove_dir"],
}


def _copy(src: Path, dst: Path, dry: bool) -> bool:
    if not src.exists():
        print(f"  MISSING source: {src}")
        return False
    print(f"  {src}\n    -> {dst}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Stage local checkpoints into ${TSD_ASSETS}")
    p.add_argument("--assets-root", type=Path, default=ASSETS_ROOT)
    p.add_argument("--only", default=None,
                   help="Comma list of components: " + ",".join(COMPONENTS))
    p.add_argument("--dry-run", action="store_true")
    for key, src in DEFAULT_SOURCES.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=Path, default=src,
                       help=f"Source for {key} (default: {src})")
    args = p.parse_args()

    sources = {k: getattr(args, k) for k in DEFAULT_SOURCES}
    want = set(COMPONENTS) if not args.only else {c.strip() for c in args.only.split(",")}
    print(f"assets-root: {args.assets_root}\ndry-run: {args.dry_run}\n")

    ok = miss = 0
    for comp in COMPONENTS:
        if comp not in want:
            continue
        print(f"[{comp}]")
        if comp == "glove":
            gdir = sources["glove_dir"]
            jobs = [(gdir / name, args.assets_root / "glove" / name) for name in _GLOVE_FILES]
        else:
            jobs = [(sources[key], args.assets_root / DEST[key]) for key in COMPONENTS[comp]]
        for src, dst in jobs:
            if _copy(src, dst, args.dry_run):
                ok += 1
            else:
                miss += 1
        print()

    print(f"Done: {ok} copied, {miss} missing.")
    if miss:
        print("Some sources were missing — pass the matching --<source> flag to point at them.")


if __name__ == "__main__":
    main()
