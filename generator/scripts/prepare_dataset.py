#!/usr/bin/env python3
"""Prepare the ``JianuoCao/TEXEDO`` dataset for generator/verifier training.

Three steps, run end to end by :func:`main`:

1. **Download** the public ``JianuoCao/TEXEDO`` dataset from the Hugging Face Hub into
   ``${TSD_DATA}/texedo`` via ``huggingface_hub.snapshot_download``.
2. **Flatten** the bucketed TEXEDO layout (``motions/{source}/{bucket}/{id}.npy``,
   ``texts/{source}/{bucket}/{id}.txt``) into the flat ``CustomCombined`` layout the
   trainers expect (``new_joint_vecs/{id}.npy``, ``texts/{id}.txt``), using the
   ``data/*.jsonl`` index to resolve files rather than reconstructing bucket paths by hand.
3. **Regenerate FSQ tokens** (``TOKENS_FSQ/{id}.npy``) by encoding each flattened motion
   with the frozen FSQ tokenizer.

The token-generation step imports the tokenizer lazily (inside the function that needs
it) because ``tokenizer/`` may be packaged by a different task and not be importable at
the time this module is first loaded. See :func:`_load_encoder`.

Usage
-----
    python generator/scripts/prepare_dataset.py
    python generator/scripts/prepare_dataset.py --limit 8 --device cpu   # smoke test
    python generator/scripts/prepare_dataset.py --skip-tokens            # flatten only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Allow running as a plain script (python generator/scripts/prepare_dataset.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from textseedo.paths import ASSETS_ROOT, DATA_ROOT, assets, data

# TEXEDO's "validation" split corresponds to CustomCombined's "val" split file.
_HF_SPLIT_TO_LOCAL = {"train": "train", "validation": "val", "test": "test"}
_JSONL_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")


def download_texedo(texedo_dir: Path, repo_id: str = "JianuoCao/TEXEDO") -> Path:
    """Download the TEXEDO dataset snapshot into ``texedo_dir``."""
    from huggingface_hub import snapshot_download

    texedo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {texedo_dir}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(texedo_dir))
    return texedo_dir


def _iter_index_rows(texedo_dir: Path) -> Iterable[dict]:
    """Yield every row across the TEXEDO ``data/*.jsonl`` index files.

    Each row has at least: ``id``, ``split``, ``source``, ``motion_path``, ``text_path``
    (paths relative to ``texedo_dir``).
    """
    index_dir = texedo_dir / "data"
    jsonl_files = [index_dir / name for name in _JSONL_FILES if (index_dir / name).exists()]
    if not jsonl_files:
        # Fall back to whatever *.jsonl files exist (e.g. a single combined index).
        jsonl_files = sorted(index_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No *.jsonl index files found under {index_dir}")

    seen_ids: set[str] = set()
    for jsonl_path in jsonl_files:
        if jsonl_path.name == "all.jsonl":
            continue  # redundant with the per-split files; avoid double-processing ids
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                yield row


def flatten_texedo(
    texedo_dir: Path,
    combined_dir: Path,
    limit: int | None = None,
) -> list[str]:
    """Flatten the bucketed TEXEDO layout into the flat CustomCombined layout.

    Copies ``motions/{source}/{bucket}/{id}.npy`` -> ``new_joint_vecs/{id}.npy`` and
    ``texts/{source}/{bucket}/{id}.txt`` -> ``texts/{id}.txt`` for every row in the
    ``data/*.jsonl`` index, then copies ``train.txt``/``val.txt``/``test.txt`` as-is.

    Returns the list of sample ids that were flattened (respecting ``limit``).
    """
    motions_out = combined_dir / "new_joint_vecs"
    texts_out = combined_dir / "texts"
    motions_out.mkdir(parents=True, exist_ok=True)
    texts_out.mkdir(parents=True, exist_ok=True)

    # When --limit is set, take up to `limit` samples from EACH split so the resulting
    # mini-dataset has non-empty train/val/test (the datamodule needs all three).
    per_split_count: dict[str, int] = {}

    ids: list[str] = []
    for row in _iter_index_rows(texedo_dir):
        if limit is not None:
            split = row.get("split", "train")
            if per_split_count.get(split, 0) >= limit:
                continue
            per_split_count[split] = per_split_count.get(split, 0) + 1

        sample_id = row["id"]
        motion_src = texedo_dir / row["motion_path"]
        text_src = texedo_dir / row["text_path"]
        if not motion_src.exists():
            raise FileNotFoundError(f"Missing motion file for id={sample_id}: {motion_src}")
        if not text_src.exists():
            raise FileNotFoundError(f"Missing text file for id={sample_id}: {text_src}")

        shutil.copy2(motion_src, motions_out / f"{sample_id}.npy")
        shutil.copy2(text_src, texts_out / f"{sample_id}.txt")
        ids.append(sample_id)

    _copy_split_files(texedo_dir, combined_dir, keep_ids=set(ids) if limit is not None else None)
    _copy_instruction_templates(combined_dir)

    print(f"Flattened {len(ids)} samples -> {combined_dir}")
    return ids


def _copy_instruction_templates(combined_dir: Path) -> None:
    """Copy the multitask instruction templates into the dataset root.

    The generator's multitask datamodule reads ``template_pretrain.json`` /
    ``template_instructions.json`` from the dataset root. TEXEDO does not ship them,
    so they live in the repo at ``generator/templates/`` and are copied here.
    """
    templates_dir = Path(__file__).resolve().parents[1] / "templates"
    for name in ("template_pretrain.json", "template_instructions.json"):
        src = templates_dir / name
        if src.exists():
            shutil.copy2(src, combined_dir / name)
        else:
            print(f"WARNING: instruction template not found: {src}")


def _copy_split_files(texedo_dir: Path, combined_dir: Path, keep_ids: set[str] | None) -> None:
    """Copy train.txt/val.txt/test.txt, optionally filtered down to ``keep_ids``."""
    for hf_name, local_name in _HF_SPLIT_TO_LOCAL.items():
        src = texedo_dir / f"{local_name}.txt"
        if not src.exists():
            # Some mirrors only ship the HF split names; fall back to those.
            alt = texedo_dir / f"{hf_name}.txt"
            src = alt if alt.exists() else src
        if not src.exists():
            print(f"Warning: split file not found, skipping: {src}")
            continue

        sample_ids = src.read_text(encoding="utf-8").split()
        if keep_ids is not None:
            sample_ids = [sid for sid in sample_ids if sid in keep_ids]

        dst = combined_dir / f"{local_name}.txt"
        dst.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")


def _load_encoder(fsq_checkpoint: Path, device: str):
    """Lazily load the FSQ tokenizer adapter.

    This import is deliberately deferred to call time (not module load time): the
    ``tokenizer`` package is packaged by a separate task and may not exist yet in a
    partially-built checkout. Importing it here keeps this script importable regardless.
    """
    from tokenizer.fsq_adapter import load_fsq_adapter

    adapter = load_fsq_adapter(str(fsq_checkpoint), device=device)
    adapter.eval()
    return adapter


def encode_tokens(
    combined_dir: Path,
    ids: list[str],
    fsq_checkpoint: Path,
    device: str = "cuda",
) -> None:
    """Encode flattened ``new_joint_vecs/{id}.npy`` motions to ``TOKENS_FSQ/{id}.npy``.

    TEXEDO/CustomCombined motions are flat ``(T, 36)`` float32 arrays (root pos + wxyz
    quat + 29 joint values), which is exactly what :func:`fsq_adapter.FSQAdapter.encode`
    expects (as a batch of size 1). We therefore call the adapter directly rather than
    going through ``tokenizer/encode_motion_tokens.py``, which expects ``.npz`` inputs
    with separate ``body_pos_w``/``body_quat_w``/``joint_pos`` arrays.
    """
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    adapter = _load_encoder(fsq_checkpoint, device)

    tokens_out = combined_dir / "TOKENS_FSQ"
    tokens_out.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    for sample_id in ids:
        motion_path = combined_dir / "new_joint_vecs" / f"{sample_id}.npy"
        try:
            motion = np.load(motion_path).astype(np.float32)
            features = torch.from_numpy(motion).unsqueeze(0).to(device)  # (1, T, 36)
            with torch.no_grad():
                code_idx, _ = adapter.encode(features)
            tokens = code_idx[0].detach().cpu().numpy().astype(np.int32)
            np.save(tokens_out / f"{sample_id}.npy", tokens)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed += 1
            print(f"  [skip] {sample_id}: {exc}")

    print(f"Encoded FSQ tokens: {ok} ok, {failed} failed -> {tokens_out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download JianuoCao/TEXEDO, flatten it into the CustomCombined "
        "training layout, and regenerate FSQ tokens.",
    )
    parser.add_argument(
        "--repo-id",
        default="JianuoCao/TEXEDO",
        help="Hugging Face dataset repo id (default: %(default)s).",
    )
    parser.add_argument(
        "--texedo-dir",
        type=Path,
        default=data("texedo"),
        help="Where the raw TEXEDO HF snapshot is downloaded to "
        "(default: ${TSD_DATA}/texedo = %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data("CustomCombined"),
        help="Where the flattened training layout is written "
        "(default: ${TSD_DATA}/CustomCombined = %(default)s).",
    )
    parser.add_argument(
        "--fsq-checkpoint",
        type=Path,
        default=assets("tokenizer", "checkpoint_epoch_95.pt"),
        help="FSQ tokenizer checkpoint used to regenerate tokens "
        "(default: ${TSD_ASSETS}/tokenizer/checkpoint_epoch_95.pt = %(default)s).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing --texedo-dir snapshot instead of downloading.",
    )
    parser.add_argument(
        "--skip-tokens",
        action="store_true",
        help="Flatten only; skip FSQ token regeneration.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples from the index (smoke testing).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for FSQ token encoding (default: %(default)s).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print(f"TSD_DATA={DATA_ROOT}  TSD_ASSETS={ASSETS_ROOT}")

    if args.skip_download:
        print(f"Skipping download, using existing snapshot at {args.texedo_dir}")
    else:
        download_texedo(args.texedo_dir, repo_id=args.repo_id)

    ids = flatten_texedo(args.texedo_dir, args.output_dir, limit=args.limit)

    if args.skip_tokens:
        print("Skipping FSQ token regeneration (--skip-tokens).")
        return

    encode_tokens(args.output_dir, ids, args.fsq_checkpoint, device=args.device)


if __name__ == "__main__":
    main()
