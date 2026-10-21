"""OpenReframe — single-image 3D reframing by depth-aware reprojection.

Import order matters: `_paths` redirects the HuggingFace caches into the repo
before any HF library is imported, so it must come first.
"""
from __future__ import annotations

from . import _paths  # noqa: F401  (side effect: HF cache redirection)

__version__ = "0.1.0"
__all__ = ["_paths", "config", "pipeline", "_quality", "_smart", "_variants"]