"""Quality evaluation for a generated reframed image.

Every number reported here is measured from data the pipeline already
produces: the reprojected frame (`warped`), the disocclusion mask (`mask`) and
the generated image (`filled`). There is no learned evaluator and no extra
model download. A metric that cannot be computed is reported as *unavailable*
rather than replaced by a fabricated number.

What each metric means
----------------------
preservation  SSIM between the reprojected frame and the output, restricted to
              the region the generator was NOT allowed to touch. It catches the
              model damaging content that should have survived. Comparing
              against the *reprojected* frame (not the original photograph) is
              what makes the comparison valid under camera movement.
boundary      Colour and gradient discontinuity across the mask edge. A low
              value means a visible seam. It does NOT prove that the generated
              content is semantically plausible.
coverage      Fraction of the missing area that the generator actually
              replaced, measured against the pre-inpaint placeholder. Filling
              a region is not the same as filling it correctly.
composition   Heuristic built from the geometry: how much original content
              survived the move, how much was cropped away, how large the
              invented area is and how concentrated it is.
subject       Unavailable - the project ships no object detector, so subject
              preservation cannot be measured honestly.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CANON_SHIFT

# --- composition scoring ---------------------------------------------------
# Weights per strategy. Each row sums to 1.0 so the score stays on 0-100.
# `preserve` favours a small, safe move; `explore` rewards a larger one.
STRATEGIES: dict[str, dict[str, float]] = {
    "preserve": {"retention": 0.45, "exposure": 0.35, "crop": 0.15,
                 "spread": 0.05, "motion": 0.00},
    "balanced": {"retention": 0.35, "exposure": 0.30, "crop": 0.20,
                 "spread": 0.15, "motion": 0.00},
    "explore": {"retention": 0.20, "exposure": 0.25, "crop": 0.10,
                "spread": 0.10, "motion": 0.35},
}
DEFAULT_STRATEGY = "balanced"

# Fraction of the output frame the inpainter has to invent. Above this the
# diffusion model stops producing plausible content, so the same number is
# used as the "too aggressive" rejection threshold in the recovery logic.
MAX_EXPOSURE_FRAC = 0.20

# Fraction of the original frame that may be cropped away before a shift is
# considered over-aggressive.
MAX_CROP_FRAC = 0.25

# Motion is normalised against the exploratory tier magnitude, so a shift as
# large as that tier scores 1.0 on the motion factor.
MOTION_REFERENCE = 1.7

# --- acceptance floors -----------------------------------------------------
# Each floor is applied to its own metric. Acceptance is deliberately NOT a
# single aggregate score: a perfect expansion coverage must not be able to
# hide severe damage to the original image.
ACCEPT_FLOORS: dict[str, float] = {
    "preservation": 75.0,
    "coverage": 90.0,
    "boundary": 55.0,
}

# Boundary tuning. A mean colour jump of 0.15 (about 38/255) across the seam
# scores zero; a gradient gap of 0.5 (half of the local gradient energy)
# scores zero.
BOUNDARY_COLOR_CAP = 0.15
BOUNDARY_GRAD_CAP = 0.50

# Side of the SSIM window. The window reaches this far, so the preservation
# metric excludes the same distance around the mask.
SSIM_WINDOW = 11
ssim_window_radius = SSIM_WINDOW // 2


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass
class Metric:
    """One reported metric plus how it was obtained."""

    name: str
    value: float | None
    status: str          # measured | heuristic | unavailable | failed
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "status": self.status,
                "note": self.note}


# --- small helpers ---------------------------------------------------------

def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> (3,H,W) float32 in [0,1]."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def _resize_like(t: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Resample a (3,H,W) tensor onto (H,W)."""
    if tuple(t.shape[-2:]) == tuple(hw):
        return t
    return F.interpolate(t[None], size=hw, mode="bilinear",
                         align_corners=False)[0]


def mask_bool(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Normalise a mask to a boolean (H,W) array."""
    if isinstance(mask, Image.Image):
        return np.asarray(mask.convert("L")) > 127
    arr = np.asarray(mask)
    return arr > 127 if arr.dtype != bool else arr


def motion_ratio(shift: tuple[float, float, float]) -> float:
    """Length of a camera shift relative to the canonical shift."""
    base = float(np.linalg.norm(CANON_SHIFT))
    if base <= 0:
        return 0.0
    return float(np.linalg.norm(np.asarray(shift, dtype=float))) / base


def mask_concentration(mask: Image.Image | np.ndarray) -> float:
    """Share of the masked area that sits in the single largest blob.

    One coherent region is far easier for the inpainter to fill than the same
    area scattered into hundreds of specks, so this is the "spread" factor of
    the composition score.
    """
    m = mask_bool(mask)
    area = int(m.sum())
    if area == 0:
        return 1.0
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 1.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest / area


# --- SSIM ------------------------------------------------------------------

def _gaussian_window(channels: int, size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = torch.outer(g, g)
    return win.expand(channels, 1, size, size).contiguous()


def ssim_map(a: torch.Tensor, b: torch.Tensor, window: int = 11,
             sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel SSIM between two (3,H,W) float tensors in [0,1].

    Standard Wang et al. formulation with a Gaussian window. Computed on CPU
    in float32 so the GPU stays free for the next generation.
    """
    if a.shape != b.shape:
        raise ValueError("ssim_map needs two tensors with the same shape")
    h, w = a.shape[-2:]
    win_size = min(window, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = win_size // 2
    groups = a.shape[0]
    win = _gaussian_window(groups, win_size, sigma)

    x = F.pad(a[None], (pad,) * 4, mode="reflect")
    y = F.pad(b[None], (pad,) * 4, mode="reflect")
    mu_x = F.conv2d(x, win, groups=groups)
    mu_y = F.conv2d(y, win, groups=groups)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = F.conv2d(x * x, win, groups=groups) - mu_xx
    sigma_y = F.conv2d(y * y, win, groups=groups) - mu_yy
    sigma_xy = F.conv2d(x * y, win, groups=groups) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sigma_x + sigma_y + c2)
    return (num / den).squeeze(0).mean(0)      # average the colour channels


# --- the four metrics ------------------------------------------------------

def preservation_metric(warped: Image.Image | None,
                         filled: Image.Image | None,
                         mask: Image.Image | None) -> tuple[Metric, dict]:
    """How much of the untouched region survived the generation."""
    if warped is None or filled is None:
        return Metric("Original-Region Preservation", None, "unavailable",
                      "no images to compare"), {}
    if mask is None:
        return Metric("Original-Region Preservation", None, "unavailable",
                      "no disocclusion mask was supplied, so the region the "
                      "generator was allowed to touch is unknown"), {}

    ref = _to_tensor(warped)
    out = _to_tensor(filled)
    if ref.shape != out.shape:
        # Comparing different sizes directly would score garbage, so the
        # reference is resampled onto the output grid first.
        ref = _resize_like(ref, tuple(out.shape[-2:]))
    m = mask_bool(mask)
    if m.shape != out.shape[-2:]:
        m = cv2.resize(m.astype(np.uint8), (out.shape[-1], out.shape[-2]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)

    # The generated area is excluded, and so is a narrow band around it: the
    # generator is allowed to blend into the seam, and the SSIM window reaches
    # across it.
    band = max(1, ssim_window_radius)
    keep = ~cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=band).astype(bool)
    n_keep = int(keep.sum())
    if n_keep < 64:
        return Metric("Original-Region Preservation", None, "unavailable",
                      "almost the whole frame was masked, so there is no "
                      "preserved region to measure"), {"compared_px": 0}

    smap = ssim_map(ref, out)
    value = float(smap[keep].mean()) * 100.0
    return (Metric("Original-Region Preservation", round(value, 1), "measured",
                   f"SSIM over {n_keep} pixels the generator was not allowed "
                   f"to change (a {band}px band around the mask is excluded)"),
            {"compared_px": n_keep})


def boundary_metric(warped: Image.Image | None,
                    filled: Image.Image | None,
                    mask: Image.Image | None) -> tuple[Metric, dict]:
    """Colour and gradient discontinuity across the mask edge."""
    if filled is None or mask is None:
        return Metric("Boundary Consistency", None, "unavailable",
                      "no generated image or mask to compare"), {}
    m = mask_bool(mask)
    out_u8 = np.asarray(filled.convert("RGB"), dtype=np.uint8)
    if m.shape != out_u8.shape[:2]:
        # The mask has to live on the same grid as the image it describes.
        m = cv2.resize(m.astype(np.uint8), (out_u8.shape[1], out_u8.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    if not m.any():
        return Metric("Boundary Consistency", None, "unavailable",
                      "the mask is empty, so there is no seam"), {}
    if m.all():
        return Metric("Boundary Consistency", None, "unavailable",
                      "the mask covers the whole frame, so there is no seam"), {}

    k = np.ones((3, 3), np.uint8)
    mu8 = m.astype(np.uint8)
    inside_band = m & ~cv2.erode(mu8, k).astype(bool)
    outside_band = cv2.dilate(mu8, k).astype(bool) & ~m
    if int(inside_band.sum()) < 20 or int(outside_band.sum()) < 20:
        return Metric("Boundary Consistency", None, "unavailable",
                      "the mask is too thin to have a measurable seam"), {}

    # For every generated pixel on the seam, find the nearest original pixel
    # on the other side; the colour jump between them is the discontinuity.
    from scipy import ndimage
    _dist, ind = ndimage.distance_transform_edt(m, return_indices=True)
    ys, xs = np.nonzero(inside_band)
    ny, nx = ind[0][ys, xs], ind[1][ys, xs]

    out_f = out_u8.astype(np.float32) / 255.0
    color_diff = float(np.abs(out_f[ys, xs] - out_f[ny, nx]).mean())

    gray = cv2.cvtColor(out_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    g_in = float(grad[inside_band].mean())
    g_out = float(grad[outside_band].mean())
    grad_gap = abs(g_in - g_out) / (g_out + 0.02)

    color_score = 100.0 * _clamp01(1.0 - color_diff / BOUNDARY_COLOR_CAP)
    grad_score = 100.0 * _clamp01(1.0 - grad_gap / BOUNDARY_GRAD_CAP)
    value = 0.7 * color_score + 0.3 * grad_score
    return (Metric("Boundary Consistency", round(value, 1), "measured",
                   "colour and gradient jump between the generated region and "
                   "the adjacent original pixels; a low value means a visible "
                   "seam, not that the content is realistic"),
            {"color_jump": round(color_diff, 4),
             "gradient_gap": round(grad_gap, 4),
             "seam_px": int(inside_band.sum())})


def coverage_metric(warped: Image.Image | None,
                     filled: Image.Image | None,
                     mask: Image.Image | None) -> tuple[Metric, dict]:
    """How much of the missing area the generator actually replaced."""
    if warped is None or filled is None or mask is None:
        return Metric("Expansion Coverage", None, "unavailable",
                      "no warped frame, result or mask to compare"), {}
    m = mask_bool(mask)
    total = int(m.sum())
    if total == 0:
        return (Metric("Expansion Coverage", 100.0, "measured",
                       "the mask is empty, so there was nothing to expand"),
                {"expected_missing_px": 0, "filled_px": 0,
                 "remaining_unfilled_px": 0})

    a = np.asarray(warped.convert("RGB"), dtype=np.float32)
    b = np.asarray(filled.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    # Inside the mask the warped frame still holds the nearest-valid-pixel
    # placeholder. A pixel the inpainter really synthesised differs from it.
    diff = np.abs(a - b).max(axis=2)
    unchanged = int((diff[m] <= 1.0).sum())
    filled_px = total - unchanged
    value = 100.0 * filled_px / total
    return (Metric("Expansion Coverage", round(value, 1), "measured",
                   f"{filled_px} of {total} missing pixels were replaced; "
                   f"replaced is not the same as correct"),
            {"expected_missing_px": total, "filled_px": filled_px,
             "remaining_unfilled_px": unchanged,
             "expected_missing_pct": round(100.0 * total / m.size, 2)})


def composition_factors(geom: dict) -> dict:
    """Per-factor composition scores in [0,1]; 1 is always the better end."""
    exposure = float(geom.get("exposure_frac", 0.0))
    crop = float(geom.get("crop_frac", 0.0))
    return {
        # how much of the new frame comes from the original photograph
        "retention": _clamp01(geom.get("retention", 0.0)),
        # penalty for inventing a large area
        "exposure": _clamp01(1.0 - min(exposure / MAX_EXPOSURE_FRAC, 1.0)),
        # penalty for cropping the original away
        "crop": _clamp01(1.0 - min(crop / MAX_CROP_FRAC, 1.0)),
        # reward for a coherent (not speckled) invented area
        "spread": _clamp01(geom.get("spread", 1.0)),
        # reward for actually moving the camera
        "motion": _clamp01(float(geom.get("motion", 0.0)) / MOTION_REFERENCE),
    }


def composition_score(geom: dict,
                      strategy: str = DEFAULT_STRATEGY) -> tuple[float, dict]:
    """Weighted composition score in 0-100 plus the individual factors."""
    weights = STRATEGIES.get(strategy, STRATEGIES[DEFAULT_STRATEGY])
    factors = composition_factors(geom)
    score = 100.0 * sum(weights[k] * factors[k] for k in weights)
    return score, factors


def is_acceptable(metrics: dict) -> bool:
    """Per-metric acceptance check.

    A metric that could not be measured does not block acceptance - unknown
    is not the same as bad - but every measured metric has to clear its own
    floor.
    """
    for key, floor in ACCEPT_FLOORS.items():
        entry = metrics.get(key) or {}
        value = entry.get("value")
        if value is not None and value < floor:
            return False
    return True


def evaluate(warped: Image.Image | None, filled: Image.Image | None,
             mask: Image.Image | None, geom: dict | None = None,
             strategy: str = DEFAULT_STRATEGY) -> dict:
    """Build the full quality report for one generated image.

    `geom` carries the geometric facts of the move (retention, crop, exposure,
    spread, motion). Without it the composition score is reported as
    unavailable rather than guessed.
    """
    metrics: dict[str, dict] = {}
    details: dict = {}

    pres, d = preservation_metric(warped, filled, mask)
    metrics["preservation"] = pres.to_dict()
    details["preservation"] = d

    bnd, d = boundary_metric(warped, filled, mask)
    metrics["boundary"] = bnd.to_dict()
    details["boundary"] = d

    cov, d = coverage_metric(warped, filled, mask)
    metrics["coverage"] = cov.to_dict()
    details["coverage"] = d

    if geom is None:
        metrics["composition"] = Metric(
            "Composition Score", None, "unavailable",
            "no geometry statistics were supplied").to_dict()
    else:
        score, factors = composition_score(geom, strategy)
        metrics["composition"] = Metric(
            "Composition Score", round(score, 1), "heuristic",
            "weighted mix of original-content retention, invented area, "
            "cropping, mask concentration and camera movement").to_dict()
        details["composition"] = {"factors": factors,
                                  "weights": STRATEGIES.get(
                                      strategy, STRATEGIES[DEFAULT_STRATEGY])}
        details["geometry"] = geom

    metrics["subject"] = Metric(
        "Subject Preservation", None, "unavailable",
        "the project ships no object detector, so subject preservation "
        "cannot be measured").to_dict()

    return {"metrics": metrics, "details": details,
            "acceptable": is_acceptable(metrics)}


__all__ = [
    "ACCEPT_FLOORS", "DEFAULT_STRATEGY", "MAX_CROP_FRAC", "MAX_EXPOSURE_FRAC",
    "Metric", "STRATEGIES", "composition_factors", "composition_score",
    "evaluate", "is_acceptable", "mask_bool", "mask_concentration",
    "motion_ratio", "ssim_map",
]
