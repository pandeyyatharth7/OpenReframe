"""Frozen configuration for OpenReframe.

Every constant here is a locked decision. Provenance lives in
`docs/decisions.md`; measurements live in `docs/benchmarks.md`. Changing a
value here invalidates the benchmark rows that were recorded under it, so any
edit must be followed by a re-profile before it is committed.
"""
from __future__ import annotations

import torch

from . import _paths

REPO_ROOT = _paths.REPO_ROOT

# --- depth: ViT-L relative-depth checkpoint, adopted in R1 -----------------
# Relative (disparity-like) depth; pseudo-depth is derived as z = 1/(d+eps) and
# median-normalized, so absolute scale is never needed (D0.3).
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"

# --- camera / reprojection --------------------------------------------------
# Translation-only camera: no rotation, no focal change. Sliders are offsets in
# units of median scene depth (D0.3 locked scope decision).
FOV_H_DEG = 55.0
CANON_SHIFT = (0.18, -0.06, 0.12)   # frozen eval shift; do not edit post-hoc

# --- generation: identical across every candidate so comparisons stay valid ---
SEED = 0
STEPS = 25
GUIDANCE = 5.5
CONTROLNET_SCALE = 0.9
NEUTRAL_PROMPT = "a high quality, detailed photograph"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text"

# --- inpainting backend: candidate B, locked for v1 in R3 -------------------
# SD1.5-inpaint + depth-ControlNet. Selected under the D0.1 escape clause; see
# docs/decisions.md R3 for the documented gap (candidates A and C unprofiled).
BACKEND_BASE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
BACKEND_CONTROLNET = "lllyasviel/control_v11f1p_sd15_depth"
PIPELINE_CLASS = "sd15_cn"
CAP_PX = 768                        # SD1.5's sensible operating long side
RES_MULTIPLE = 64                   # both SD1.5 and SDXL want /64 sizes

# --- pre-committed budgets (D0) ---------------------------------------------
VRAM_CEIL_MB = 6.8 * 1024            # 6959 MiB hard ceiling per stage
PREVIEW_SLA_S = 2.0                  # geometry-only preview budget
DEPTH_BUDGET_S = 1.2                 # warm p95 allowance for depth inference

# --- eval set (frozen) ------------------------------------------------------
EVAL_IMAGES = ["ind01.jpg", "out02.jpg", "mix01.jpg",
               "mix05.jpg", "out04.jpg", "mix06.jpg"]

# --- OOM watchdog (Phase 3) -------------------------------------------------
# WDDM CUDA sysmem fallback is ACTIVE on the target machine, so a real OOM
# surfaces as a large slowdown rather than an OutOfMemoryError. Detection has
# to be time-based. Measured warm p95 for candidate B is ~10.6 s.
OOM_SUSPECT_FACTOR = 4.0
OOM_ABSOLUTE_FLOOR_S = 45.0


def device() -> str:
    """Compute device. Falls back to CPU so geometry code stays runnable."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def local_or_repo(repo_id: str) -> str:
    """Prefer directly-downloaded weights over a Hub id.

    `models/direct/` is populated by scripts/download_direct.py, which bypasses
    hf-xet stalls. Falls back to the Hub id when the local copy is absent.
    """
    p = _paths.MODELS_DIR / "direct" / repo_id.replace("/", "__")
    return str(p) if (p / "model_index.json").exists() or (p / "config.json").exists() \
        else repo_id


def fit_resolution(w: int, h: int, cap: int = CAP_PX,
                   mult: int = RES_MULTIPLE) -> tuple[int, int]:
    """Scale (w,h) down to fit the cap, rounded to a multiple of `mult`."""
    s = min(1.0, cap / max(w, h))
    return (max(mult, int(round(w * s / mult)) * mult),
            max(mult, int(round(h * s / mult)) * mult))