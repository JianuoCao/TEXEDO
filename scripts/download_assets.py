#!/usr/bin/env python
"""Download all text-see-do assets (dataset + checkpoints) from the Hugging Face Hub.

Reads the manifest in ``configs/paths.yaml`` and fetches:
  - the TEXEDO dataset (``snapshot_download``) into ``${TSD_DATA}/texedo``
  - every checkpoint file (``hf_hub_download``) from ``checkpoints.hf_repo`` into
    ``${TSD_ASSETS}/<local>``, unpacking ``.tar.gz`` archives when the manifest entry
    sets ``unpack: true``.

Usage
-----
    python scripts/download_assets.py                       # fetch everything
    python scripts/download_assets.py --dry-run              # resolve + print only
    python scripts/download_assets.py --only fsq_tokenizer,generator
    python scripts/download_assets.py --assets-root /mnt/assets --data-root /mnt/data

See ``docs/MODELS.md`` for what each entry is and ``docs/UPLOAD.md`` for how to
populate ``checkpoints.hf_repo`` once you have uploaded your own checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import yaml

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from textseedo.paths import ASSETS_ROOT, DATA_ROOT, REPO_ROOT

DEFAULT_MANIFEST = REPO_ROOT / "configs" / "paths.yaml"
TODO_HF_REPO = "TODO_USER_MODEL_REPO"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path, "r") as f:
        return yaml.safe_load(f)


def _unpack_if_needed(archive_path: Path, dest_dir: Path, dry_run: bool) -> None:
    """Extract a .tar.gz archive next to itself and remove the archive."""
    if dry_run:
        print(f"    [dry-run] would extract {archive_path.name} -> {dest_dir}/")
        return
    print(f"    extracting {archive_path.name} -> {dest_dir}/")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    archive_path.unlink()


def download_dataset(entry: dict[str, Any], data_root: Path, dry_run: bool) -> None:
    from huggingface_hub import snapshot_download

    hf_repo = entry["hf_repo"]
    hf_type = entry.get("hf_type", "dataset")
    local_dir = data_root / entry["local"]

    print(f"[dataset] {hf_repo} ({hf_type}) -> {local_dir}")
    if dry_run:
        print("    [dry-run] would snapshot_download into the path above")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_repo, repo_type=hf_type, local_dir=str(local_dir))
    print(f"    done -> {local_dir}")


def download_checkpoint(
    name: str,
    file_entry: dict[str, Any],
    hf_repo: str,
    hf_type: str,
    assets_root: Path,
    dry_run: bool,
) -> None:
    remote = file_entry["remote"]
    local_rel = file_entry["local"]
    unpack = bool(file_entry.get("unpack", False))
    dest_path = assets_root / local_rel

    print(f"[checkpoint] {name}: {hf_repo}::{remote} -> {dest_path}")

    if hf_repo == TODO_HF_REPO:
        if dry_run:
            print("    [dry-run] hf_repo not set yet (placeholder) -- showing intended layout only")
            if unpack:
                print(f"    [dry-run] would unpack into {dest_path.parent}/")
            return
        raise RuntimeError(
            f"checkpoints.hf_repo is still '{TODO_HF_REPO}' in configs/paths.yaml.\n"
            f"Upload your checkpoints to a Hugging Face model repo first, then set "
            f"checkpoints.hf_repo accordingly.\nSee docs/UPLOAD.md for the exact steps."
        )

    if dry_run:
        print("    [dry-run] would hf_hub_download the file above")
        if unpack:
            print(f"    [dry-run] would unpack into {dest_path.parent}/")
        return

    from huggingface_hub import hf_hub_download

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=hf_repo,
        repo_type=hf_type,
        filename=remote,
        local_dir=str(assets_root),
    )
    downloaded_path = Path(downloaded)

    # hf_hub_download places the file at <local_dir>/<remote>; move it to the
    # manifest's `local` path if that differs from `remote`.
    if downloaded_path != dest_path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.replace(dest_path)

    print(f"    downloaded -> {dest_path}")

    if unpack:
        _unpack_if_needed(dest_path, dest_path.parent, dry_run=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to the asset manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve every manifest entry and print what would be fetched; fetch nothing.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of entries to fetch (e.g. 'fsq_tokenizer,generator'). "
        "Use 'texedo' to select the dataset entry. Default: fetch everything.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=ASSETS_ROOT,
        help=f"Destination root for checkpoints (default: {ASSETS_ROOT}, i.e. $TSD_ASSETS)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Destination root for datasets (default: {DATA_ROOT}, i.e. $TSD_DATA)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    manifest = load_manifest(args.manifest)
    only = {name.strip() for name in args.only.split(",")} if args.only else None

    assets_root: Path = args.assets_root
    data_root: Path = args.data_root

    print(f"manifest:    {args.manifest}")
    print(f"assets-root: {assets_root}")
    print(f"data-root:   {data_root}")
    print(f"dry-run:     {args.dry_run}")
    print()

    # --- dataset ---
    dataset_entries = manifest.get("dataset", {}) or {}
    for ds_name, ds_entry in dataset_entries.items():
        if only is not None and ds_name not in only:
            continue
        download_dataset(ds_entry, data_root, dry_run=args.dry_run)

    # --- checkpoints ---
    ckpt_block = manifest.get("checkpoints", {}) or {}
    hf_repo = ckpt_block.get("hf_repo", TODO_HF_REPO)
    hf_type = ckpt_block.get("hf_type", "model")
    files = ckpt_block.get("files", {}) or {}

    for name, file_entry in files.items():
        if only is not None and name not in only:
            continue
        download_checkpoint(name, file_entry, hf_repo, hf_type, assets_root, dry_run=args.dry_run)

    # --- runtime-only entries (informational; nothing to fetch) ---
    runtime_entries = manifest.get("runtime", {}) or {}
    if runtime_entries and only is None:
        print()
        print("[runtime] loaded directly from the HF hub at inference time (not downloaded here):")
        for name, hf_id in runtime_entries.items():
            print(f"    {name}: {hf_id}")

    print()
    print("done." if not args.dry_run else "dry-run complete; nothing was fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
