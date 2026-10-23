"""Multi-variant generation with bounded, targeted recovery.

A *variant* is one camera movement carried through the existing pipeline:
reprojection -> SDXL inpainting -> quality report. Variants run one at a
time because the target GPU has 8 GB and the server owns a single worker.

Recovery is deliberately small. A handful of problems are detected, each has
exactly one supported correction, and at most `max_retries` corrections are
applied per variant. A variant that fails never invalidates the variants
that succeeded.

Corrections come in two kinds:

  re-run      the inputs change and the image is generated again
              (smaller camera move, widened generation mask, plain retry)
  post-fix   the image that already exists is corrected in place
              (feathered seam blend, paste-back of the preserved region)

Acceptance is decided per metric, never by one aggregate score: a perfect
expansion coverage must not be able to hide severe damage to the original.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from ._quality import (ACCEPT_FLOORS, DEFAULT_STRATEGY, MAX_EXPOSURE_FRAC,
                       evaluate, is_acceptable, mask_bool)

# How much the camera move is pulled back when it exposes too much area.
SHRINK_FACTOR = 0.6

# Width of the feather band used by the seam and paste-back corrections.
FEATHER_PX = 3

# The VAE encodes the frame at 1/8 resolution, so the mask the diffusion model
# actually sees is eight times smaller. A masked region that is only a couple
# of generation-resolution pixels wide can disappear there and leave the
# placeholder in the output; widening the mask by one latent pixel fixes that.
LATENT_MASK_PAD_PX = 8


@dataclass
class VariantSpec:
    """One candidate to generate."""

    label: str = "Variant"
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strategy: str = DEFAULT_STRATEGY
    direction: str = ""


@dataclass
class VariantResult:
    """What one variant produced, including any recovery that was applied."""

    label: str = "Variant"
    direction: str = ""
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effective_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    filled: Image.Image | None = None
    warped: Image.Image | None = None
    mask: Image.Image | None = None
    overlay: Image.Image | None = None
    stats: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    recovery: list = field(default_factory=list)
    error: str | None = None
    attempts: int = 0


# --- problem detection -----------------------------------------------------

def detect_problems(out: dict, report: dict) -> list[dict]:
    """Problems that have a supported correction.

    Only problems that can be identified reliably from the pipeline output are
    reported; semantic failures (a duplicated object, a nonsensical fill) are
    deliberately not attempted.
    """
    problems: list[dict] = []
    stats = out.get("stats", {}) or {}
    metrics = (report or {}).get("metrics", {}) or {}

    exposure = float(stats.get("hole_pct", 0.0)) / 100.0
    if exposure > MAX_EXPOSURE_FRAC:
        problems.append({"id": "excessive_disocclusion",
                         "detail": f"{exposure*100:.1f}% of the frame would "
                                   f"have to be invented"})

    value = (metrics.get("coverage") or {}).get("value")
    if value is not None and value < ACCEPT_FLOORS["coverage"]:
        problems.append({"id": "unfilled_regions",
                         "detail": f"expansion coverage {value:.1f}%"})

    value = (metrics.get("boundary") or {}).get("value")
    if value is not None and value < ACCEPT_FLOORS["boundary"]:
        problems.append({"id": "boundary_discontinuity",
                         "detail": f"boundary consistency {value:.1f}"})

    value = (metrics.get("preservation") or {}).get("value")
    if value is not None and value < ACCEPT_FLOORS["preservation"]:
        problems.append({"id": "preservation_loss",
                         "detail": f"original-region preservation {value:.1f}"})

    if (stats.get("watchdog") or {}).get("tripped"):
        problems.append({"id": "generation_stalled",
                         "detail": "the diffusion stage was stopped by the "
                                   "stall watchdog"})

    return problems


# --- corrections ------------------------------------------------------------

def _seam_weights(mask: np.ndarray, feather: int) -> tuple[np.ndarray, np.ndarray]:
    """Weights for the two sides of the mask boundary.

    `inside_w`  how strongly a generated pixel is pulled towards the nearest
                original pixel across the seam.
    `outside_w` how strongly an original pixel is restored to its reprojected
                value.
    """
    from scipy import ndimage

    inside_w = np.zeros(mask.shape, np.float32)
    outside_w = np.zeros(mask.shape, np.float32)
    if not mask.any():
        return inside_w, outside_w

    # Distance to the mask edge, measured from each side. The transform of
    # the mask itself gives the inward distance (1 on the first row inside),
    # the transform of its complement the outward one.
    dist_in_all = ndimage.distance_transform_edt(mask)
    dist_out_all = ndimage.distance_transform_edt(~mask)

    ys, xs = np.nonzero(mask)
    dist_in = dist_in_all[ys, xs]
    w = np.clip(0.5 * (1.0 - (dist_in - 1.0) / max(feather, 1)), 0.0, 0.5)
    inside_w[ys, xs] = w

    yo, xo = np.nonzero(~mask)
    d = dist_out_all[yo, xo]
    outside_w[yo, xo] = np.clip(1.0 - (d - 1.0) / max(feather, 1), 0.0, 1.0)
    return inside_w, outside_w


def blend_seam(filled: Image.Image, warped: Image.Image, mask: Image.Image,
               feather: int = FEATHER_PX) -> Image.Image:
    """Feather the generated content into the original across the mask edge.

    The diffusion model decodes the whole frame, so the pixels right at the
    seam carry generated colour. Pulling the neighbouring original pixels into
    a narrow band inside the mask (and restoring the original just outside it)
    removes that fringe.
    """
    m = mask_bool(mask)
    if not m.any():
        return filled

    a = np.asarray(filled.convert("RGB"), dtype=np.float32)
    b = np.asarray(warped.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    inside_w, outside_w = _seam_weights(m, feather)
    if inside_w.any():
        from scipy import ndimage
        _dist, ind = ndimage.distance_transform_edt(m, return_indices=True)
        nearest = b[ind[0], ind[1]]
        w = inside_w[..., None]
        a = a * (1.0 - w) + nearest * w
    if outside_w.any():
        w = outside_w[..., None]
        a = a * (1.0 - w) + b * w
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def paste_back(filled: Image.Image, warped: Image.Image, mask: Image.Image,
               feather: int = FEATHER_PX) -> Image.Image:
    """Restore the whole preserved region from the reprojected frame.

    The generator was only allowed to change the masked area, so anything it
    did outside that area is damage. The reprojected pixels are the truth
    there; the seam is feathered so the restored region does not introduce a
    new hard edge.
    """
    m = mask_bool(mask)
    if not m.any():
        return filled

    a = np.asarray(filled.convert("RGB"), dtype=np.float32)
    b = np.asarray(warped.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    inside_w, _outside_w = _seam_weights(m, feather)
    restore = (~m).astype(np.float32)
    if inside_w.any():
        from scipy import ndimage
        _dist, ind = ndimage.distance_transform_edt(m, return_indices=True)
        nearest = b[ind[0], ind[1]]
        w = inside_w[..., None]
        a = a * (1.0 - w) + nearest * w
    w = restore[..., None]
    return Image.fromarray(
        np.clip(a * (1.0 - w) + b * w, 0, 255).astype(np.uint8))


def _apply_post_fixes(out: dict, report: dict,
                      problems: list[dict]) -> tuple[dict, dict]:
    """Correct the image we already have, if that improves it."""
    ids = {p["id"] for p in problems}
    warped, filled, mask = out.get("warped"), out.get("filled"), out.get("mask")
    if filled is None or warped is None or mask is None:
        return out, report

    if "preservation_loss" in ids:
        candidate = dict(out)
        candidate["filled"] = paste_back(filled, warped, mask)
        cand_report = evaluate(warped, candidate["filled"], mask,
                               out.get("geom"))
        if _rank(cand_report) > _rank(report):
            out, report = candidate, cand_report

    if "boundary_discontinuity" in ids:
        candidate = dict(out)
        candidate["filled"] = blend_seam(out["filled"], warped, mask)
        cand_report = evaluate(warped, candidate["filled"], mask,
                               out.get("geom"))
        if _rank(cand_report) > _rank(report):
            out, report = candidate, cand_report

    return out, report


def _plan_rerun(problems: list[dict], shift: tuple[float, float, float],
                mask_dilate_px: int) -> dict | None:
    """Choose the correction for the next generation attempt.

    Only one correction is applied per attempt so the cause stays traceable.
    """
    ids = {p["id"] for p in problems}

    if "excessive_disocclusion" in ids:
        # Pull the camera back: the move exposed more area than the inpainter
        # can plausibly fill, and a smaller move exposes less.
        new = tuple(float(v) * SHRINK_FACTOR for v in shift)
        return {"action": "reduce_shift", "shift": new,
                "mask_dilate_px": mask_dilate_px,
                "note": f"camera movement reduced to "
                        f"({new[0]:.3f}, {new[1]:.3f}, {new[2]:.3f})"}

    if "unfilled_regions" in ids:
        return {"action": "repair_mask", "shift": shift,
                "mask_dilate_px": mask_dilate_px + LATENT_MASK_PAD_PX,
                "note": "generation mask widened by one latent pixel so the "
                        "inpainter owns the seam instead of leaving the "
                        "placeholder"}

    if "generation_stalled" in ids:
        return {"action": "retry_generation", "shift": shift,
                "mask_dilate_px": mask_dilate_px,
                "note": "the diffusion stage stalled, running it again"}

    return None


# --- ranking ----------------------------------------------------------------

def _rank(report: dict) -> tuple:
    """Ordering key used to decide which attempt to keep.

    Acceptance itself is per metric (see `is_acceptable`); this key only
    decides which of two attempts is the better one to show.
    """
    metrics = (report or {}).get("metrics", {}) or {}
    ok = 1 if is_acceptable(metrics) else 0
    passed = 0
    for key, floor in ACCEPT_FLOORS.items():
        value = (metrics.get(key) or {}).get("value")
        if value is not None and value >= floor:
            passed += 1
    values = [(metrics.get(k) or {}).get("value")
              for k in ("preservation", "boundary", "coverage", "composition")]
    values = [v for v in values if v is not None]
    mean = sum(values) / len(values) if values else 0.0
    # Tie-break: prefer the attempt that invents less. Two attempts can score
    # identically on the metrics and still differ in how much of the frame the
    # inpainter had to fill, and less invented area is the safer image.
    geom = (report or {}).get("details", {}).get("geometry") or {}
    exposure = float(geom.get("exposure_frac", 0.0))
    return (ok, passed, round(mean, 3), -round(exposure, 4))


# --- orchestration ----------------------------------------------------------

def run_variant(reframer, img: Image.Image, spec: VariantSpec,
                on_step=None, cancel=None, max_retries: int = 1) -> VariantResult:
    """Generate one variant, evaluate it, and repair it when that is cheap.

    `reframer` only has to provide `reframe(...)`, which keeps the expensive
    model out of the unit tests.
    """
    from ._inpaint import GenerationCancelled

    shift = tuple(float(v) for v in spec.shift)
    mask_dilate = 0
    attempts: list[dict] = []
    produced: list[tuple[dict, dict, tuple]] = []
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            out = reframer.reframe(img, shift, on_step=on_step, cancel=cancel,
                                   mask_dilate_px=mask_dilate)
        except GenerationCancelled:
            raise
        except ValueError as exc:
            # A rejected configuration is deterministic - the same request
            # will fail the same way, so do not spend a retry on it.
            result = _empty_result(spec, shift)
            result.error = f"{type(exc).__name__}: {exc}"
            result.recovery = attempts
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt + 1, "action": "retry",
                             "problem": "generation_failed", "detail": last_error})
            if attempt >= max_retries:
                break
            continue

        report = out.get("report") or evaluate(out.get("warped"), out.get("filled"),
                                               out.get("mask"), out.get("geom"))
        problems = detect_problems(out, report)
        attempts.append({"attempt": attempt + 1,
                         "shift": [round(v, 4) for v in shift],
                         "problems": [p["id"] for p in problems],
                         "metrics": _metric_values(report)})
        produced.append((out, report, shift))

        if not problems or attempt >= max_retries:
            break
        fix = _plan_rerun(problems, shift, mask_dilate)
        if fix is None:
            break
        attempts[-1]["correction"] = fix["action"]
        attempts[-1]["note"] = fix["note"]
        shift, mask_dilate = fix["shift"], fix["mask_dilate_px"]

    if not produced:
        result = _empty_result(spec, shift)
        result.error = last_error or "generation produced no image"
        result.recovery = attempts
        return result

    # Keep the best attempt. The comparison uses the RAW metrics of each
    # attempt: a repair applied to one attempt must not make it outrank an
    # attempt that never needed one.
    out, report, used_shift = max(produced, key=lambda item: _rank(item[1]))

    # Repair the winner. The delivered image is the repaired one, so the
    # report describes what the user actually gets.
    problems = detect_problems(out, report)
    out, report = _apply_post_fixes(out, report, problems)

    result = _empty_result(spec, used_shift)
    result.filled = out.get("filled")
    result.warped = out.get("warped")
    result.mask = out.get("mask")
    result.overlay = out.get("overlay")
    result.stats = out.get("stats", {})
    result.report = report
    result.recovery = attempts
    result.attempts = len(attempts)
    return result


def _metric_values(report: dict) -> dict:
    """The numeric part of a report, for the recovery log."""
    return {k: v.get("value") for k, v in (report.get("metrics") or {}).items()
            if k != "subject"}


def _empty_result(spec: VariantSpec, shift: tuple[float, float, float]) -> VariantResult:
    return VariantResult(label=spec.label, direction=spec.direction,
                         shift=tuple(float(v) for v in spec.shift),
                         effective_shift=tuple(float(v) for v in shift))


__all__ = [
    "FEATHER_PX", "MAX_EXPOSURE_FRAC", "SHRINK_FACTOR", "VariantResult",
    "VariantSpec", "blend_seam", "detect_problems", "paste_back",
    "run_variant",
]
