"""Cache redirection. MUST be imported before huggingface_hub/transformers.

`huggingface_hub` reads HF_HUB_CACHE into a module constant at import time, so
setting the variable later has no effect. Every entry point therefore imports
this module first; it does nothing else.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
STATIC_DIR = REPO_ROOT / "static"
ASSETS_DIR = REPO_ROOT / "assets"
BENCHMARKS_MD = REPO_ROOT / "docs" / "benchmarks.md"

os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("HF_HOME", str(MODELS_DIR))