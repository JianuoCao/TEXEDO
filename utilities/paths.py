"""Central path resolution for the TEXEDO repo.

Every script and config resolves large-asset and dataset locations through this module
(or the equivalent OmegaConf env interpolation ``${oc.env:TSD_ASSETS}`` /
``${oc.env:TSD_DATA}``) so that **no absolute user paths are baked into the code**.

Resolution order
----------------
- ``REPO_ROOT``   : the repository root (this file is ``<repo>/utilities/paths.py``).
- ``ASSETS_ROOT`` : ``$TSD_ASSETS`` if set, else ``<repo>/assets``. Holds downloaded checkpoints.
- ``DATA_ROOT``   : ``$TSD_DATA``   if set, else ``<repo>/data``.   Holds datasets (TEXEDO + prepared).

So that OmegaConf interpolation always works, importing this module exports
``TSD_ASSETS`` / ``TSD_DATA`` into the environment if they are not already set.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    root = Path(value).expanduser().resolve() if value else default
    # Re-export a concrete absolute path so OmegaConf ${oc.env:...} resolves identically.
    os.environ.setdefault(env_var, str(root))
    return root


ASSETS_ROOT = _resolve("TSD_ASSETS", REPO_ROOT / "assets")
DATA_ROOT = _resolve("TSD_DATA", REPO_ROOT / "data")


def assets(*parts: str) -> Path:
    """Path under the assets root (downloaded checkpoints)."""
    return ASSETS_ROOT.joinpath(*parts)


def data(*parts: str) -> Path:
    """Path under the data root (datasets)."""
    return DATA_ROOT.joinpath(*parts)


def repo(*parts: str) -> Path:
    """Path under the repository root."""
    return REPO_ROOT.joinpath(*parts)


__all__ = [
    "REPO_ROOT",
    "ASSETS_ROOT",
    "DATA_ROOT",
    "assets",
    "data",
    "repo",
]
