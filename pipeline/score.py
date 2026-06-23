"""Score candidate motions with both verifiers.

Two scorers, each consuming a ``(T, 36)`` motion array:
- ``DynamicScorer``  -> physical-plausibility reward in [0, 1] (higher = better),
  mirroring the normalization used by ``verifiers/dynamic/predict_rewards.py``.
- ``SemanticScorer`` -> text-motion matching distance (lower = better), via the
  semantic verifier's ``verifiers/semantic/inference.py`` API.

CLI: score a directory of ``{id}.npy`` candidates for one caption and write a CSV.

    python -m pipeline.score --motion-dir <dir> --caption "a person waves" --output scores.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (python pipeline/score.py) as well as -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.paths import assets


class DynamicScorer:
    """DynamicVerifier reward for a (T, 36) motion (higher = more plausible)."""

    def __init__(self, checkpoint: Path | None = None, norm_stats: Path | None = None,
                 device: str | None = None, T_max: int = 1024,
                 d_model: int = 256, n_heads: int = 4, d_ff: int = 1024, n_layers: int = 4):
        import torch
        from verifiers.dynamic.model import DynamicVerifier
        from verifiers.dynamic.dataset import transform_36_to_94, INPUT_DIM

        self._torch = torch
        self._transform = transform_36_to_94
        self._INPUT_DIM = INPUT_DIM
        self.T_max = T_max
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = Path(checkpoint or assets("verifiers/dynamic/checkpoint_last.pt"))
        norm_stats = Path(norm_stats or assets("verifiers/dynamic/norm_stats.npz"))
        stats = np.load(norm_stats)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

        self.model = DynamicVerifier(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                     n_layers=n_layers, dropout=0.0).to(self.device)
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def score(self, motion_36d: np.ndarray) -> dict:
        torch = self._torch
        feats = self._transform(np.asarray(motion_36d, dtype=np.float32))
        T1 = min(feats.shape[0], self.T_max)
        feats = (feats[:T1] - self.mean) / self.std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = np.clip(feats, -10.0, 10.0)
        mask = np.zeros(self.T_max, dtype=bool)
        if T1 < self.T_max:
            feats = np.concatenate([feats, np.zeros((self.T_max - T1, self._INPUT_DIM), np.float32)], 0)
            mask[T1:] = True
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(feats).unsqueeze(0).to(self.device),
                torch.from_numpy(mask).unsqueeze(0).to(self.device),
            )
        return {k: float(out[k].item()) for k in
                ("reward_hat", "success_prob", "dynamics_hat", "progress_hat")}


class SemanticScorer:
    """Text-motion matching distance for a (T, 36) motion + caption (lower = better)."""

    def __init__(self, checkpoint: Path | None = None, meta_dir: Path | None = None,
                 glove_dir: Path | None = None, device: str | None = None):
        from verifiers.semantic.inference import load_evaluator
        import torch

        root = assets("verifiers/semantic/t2m_custom36_combinedv2/custom36/t2m")
        checkpoint = Path(checkpoint or root / "text_mot_match/model/finest.tar")
        meta_dir = Path(meta_dir or root / "Comp_v6_KLD01/meta")
        glove_dir = Path(glove_dir or assets("glove"))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = load_evaluator(str(checkpoint), str(meta_dir), str(glove_dir), device=device)

    def score(self, motion_36d: np.ndarray, caption: str) -> float:
        return float(self.evaluator.score(np.asarray(motion_36d, dtype=np.float32), caption))


def score_directory(motion_dir: Path, caption: str, output_csv: Path,
                    dynamic: bool = True, semantic: bool = True) -> None:
    import pandas as pd

    motion_dir = Path(motion_dir)
    files = sorted(motion_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy candidates in {motion_dir}")

    dyn = DynamicScorer() if dynamic else None
    sem = SemanticScorer() if semantic else None

    rows = []
    for f in files:
        motion = np.load(f)
        row = {"motion_key": f.stem, "caption": caption}
        if dyn is not None:
            row.update({f"R_dyn_{k}": v for k, v in dyn.score(motion).items()})
        if sem is not None:
            row["R_sem_matching_dist"] = sem.score(motion, caption)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows -> {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Score candidate motions with the verifiers")
    p.add_argument("--motion-dir", required=True, type=Path, help="Dir of {id}.npy candidates")
    p.add_argument("--caption", required=True, help="Text prompt the candidates were generated for")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path")
    p.add_argument("--no-dynamic", action="store_true", help="Skip the dynamic verifier")
    p.add_argument("--no-semantic", action="store_true", help="Skip the semantic verifier")
    args = p.parse_args()
    score_directory(args.motion_dir, args.caption, args.output,
                    dynamic=not args.no_dynamic, semantic=not args.no_semantic)


if __name__ == "__main__":
    main()
