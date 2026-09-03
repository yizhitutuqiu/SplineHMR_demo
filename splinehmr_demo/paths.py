from __future__ import annotations

import sys
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = DEMO_ROOT.parent
SPLINEHMR_ROOT = WORK_ROOT / "SplineHMR"
INPUTS_ROOT = SPLINEHMR_ROOT / "inputs"
OUTPUTS_ROOT = DEMO_ROOT / "outputs"
CACHE_ROOT = DEMO_ROOT / "cache"


def add_splinehmr_to_path() -> Path:
    """Make the sibling SplineHMR repository importable."""
    root = SPLINEHMR_ROOT.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def require_splinehmr_repo() -> Path:
    if not SPLINEHMR_ROOT.exists():
        raise FileNotFoundError(f"Sibling SplineHMR repo not found: {SPLINEHMR_ROOT}")
    if not (SPLINEHMR_ROOT / "splinehmr").exists():
        raise FileNotFoundError(f"Invalid SplineHMR repo: {SPLINEHMR_ROOT}")
    return add_splinehmr_to_path()

