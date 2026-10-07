"""Re-export shim onto the shipped implementation.

The reprojection math moved to `openreframe/_warp.py` when the package was
promoted out of `prototypes/`. This shim keeps `prototypes/04_reprojection_math.py`
pointing at the code that actually ships — otherwise the validation suite would
protect a file nothing imports, which would defeat the point of the
pre-registered protocol in docs/decisions.md.

Do not re-add implementation here. Edit `openreframe/_warp.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openreframe._warp import (  # noqa: E402,F401
    EPS_Z, backward_resample, build_intrinsics, classify_holes,
    depth_rel_to_pseudo_z, fill_channels, fill_pseudo_depth, forward_warp,
    forward_warp_full, make_disocclusion_mask, render_preview_with_gaps,
    slider_to_center, validate_warp,
)

__all__ = [
    "EPS_Z", "backward_resample", "build_intrinsics", "classify_holes",
    "depth_rel_to_pseudo_z", "fill_channels", "fill_pseudo_depth",
    "forward_warp", "forward_warp_full", "make_disocclusion_mask",
    "render_preview_with_gaps", "slider_to_center", "validate_warp",
]