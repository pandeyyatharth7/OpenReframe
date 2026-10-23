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
# Canonical eval shift. Scaled down 3x from the original (0.18, -0.06, 0.12):
# at 0.18 the reprojection masks covered 18-53% of the frame, which is far more
# disocclusion than any inpainter fills plausibly and is not representative of
# the product (users want subtle parallax). See docs/decisions.md R5.
CANON_SHIFT = (0.06, -0.02, 0.04)   # frozen eval shift; do not edit post-hoc

# Largest camera offset the pipeline accepts, in units of median scene depth.
# ~4x the canonical shift; beyond this the disocclusion masks stop being
# fillable by the inpainter.
MAX_SHIFT = 0.25

# Edge-preserving smoothing of the relative-depth map before reprojection.
# Leaf- and branch-scale depth noise is what creates the thousands of tiny
# isolated disocclusions that a diffusion model fills with one flat blob each
# (observed as scattered grey cubes). Bilateral smoothing removes that noise
# while preserving depth discontinuities at object silhouettes.
DEPTH_SMOOTH_SIGMA_SPACE = 8.0
DEPTH_SMOOTH_SIGMA_RANGE = 0.05

# --- generation: identical across every candidate so comparisons stay valid ---
SEED = 0
STEPS = 25
GUIDANCE = 5.5
CONTROLNET_SCALE = 0.9
NEUTRAL_PROMPT = "a high quality, detailed photograph"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text"

# --- inpainting backend: candidate A, selected in R5 ------------------------
# SDXL-inpaint, no ControlNet. Candidate C (SDXL + depth-ControlNet) is the
# higher-fidelity rank but exceeds both the VRAM and latency gates on this
# hardware; candidate B (SD1.5 + depth-ControlNet) passes the measurable gates
# but leaves visible netting on textured surfaces. See docs/decisions.md R5.
BACKEND_BASE = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
BACKEND_CONTROLNET = None            # candidate A is ControlNet-free
PIPELINE_CLASS = "sdxl_inpaint"
CAP_PX = 1024                        # SDXL's operating long side
RES_MULTIPLE = 64                   # both SD1.5 and SDXL want /64 sizes
# Measured warm p95 for candidate A over the 6-image eval set; drives the
# stall watchdog.
GEN_P95_S = 21.4

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
# to be time-based. Candidate C demonstrated this directly: it exceeded
# dedicated VRAM and ran 18x over budget instead of raising.
OOM_SUSPECT_FACTOR = 2.5
OOM_ABSOLUTE_FLOOR_S = 30.0


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