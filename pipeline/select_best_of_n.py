"""Best-of-N selection from a verifier-score CSV (produced by pipeline/score.py).

Combines the dynamic reward (higher = better) and the semantic matching distance
(lower = better) into a single rank and picks the best candidate.

    python -m pipeline.select_best_of_n --scores scores.csv \
        --alpha 1.0 --beta 1.0 --copy-best-to best/

Combined score (higher = better):
    combined = alpha * R_dyn_reward_hat  -  beta * z(R_sem_matching_dist)

where z(.) is per-file z-normalization of the matching distance so the two
verifiers are on comparable scales. With one of the verifiers absent, selection
falls back to whichever score is present.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def select(scores_csv: Path, alpha: float = 1.0, beta: float = 1.0):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(scores_csv).copy()
    has_dyn = "R_dyn_reward_hat" in df.columns
    has_sem = "R_sem_matching_dist" in df.columns
    if not (has_dyn or has_sem):
        raise ValueError("scores CSV has neither R_dyn_reward_hat nor R_sem_matching_dist")

    combined = np.zeros(len(df), dtype=float)
    if has_dyn:
        combined += alpha * df["R_dyn_reward_hat"].to_numpy(dtype=float)
    if has_sem:
        d = df["R_sem_matching_dist"].to_numpy(dtype=float)
        z = (d - d.mean()) / (d.std() + 1e-8)
        combined -= beta * z  # lower distance -> higher combined
    df["combined_score"] = combined
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Best-of-N selection from a verifier-score CSV")
    p.add_argument("--scores", required=True, type=Path, help="CSV from pipeline/score.py")
    p.add_argument("--alpha", type=float, default=1.0, help="Weight on dynamic reward")
    p.add_argument("--beta", type=float, default=1.0, help="Weight on semantic match")
    p.add_argument("--output", type=Path, default=None, help="Optional ranked CSV output")
    p.add_argument("--motion-dir", type=Path, default=None,
                   help="Dir of {motion_key}.npy (needed for --copy-best-to)")
    p.add_argument("--copy-best-to", type=Path, default=None,
                   help="Copy the winning {motion_key}.npy into this dir")
    args = p.parse_args()

    df = select(args.scores, args.alpha, args.beta)
    best_key = df.iloc[0]["motion_key"]
    print(f"Best candidate: {best_key}  (combined={df.iloc[0]['combined_score']:.4f})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Ranked CSV -> {args.output}")

    if args.copy_best_to:
        if args.motion_dir is None:
            raise SystemExit("--copy-best-to requires --motion-dir")
        args.copy_best_to.mkdir(parents=True, exist_ok=True)
        src = Path(args.motion_dir) / f"{best_key}.npy"
        dst = args.copy_best_to / f"{best_key}.npy"
        shutil.copy2(src, dst)
        print(f"Copied best -> {dst}")


if __name__ == "__main__":
    main()
