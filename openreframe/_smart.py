"""Smart reframing: propose camera movements and score them geometrically.

The planner never touches a diffusion model. It reuses the pseudo-depth map
that `DepthEstimator` already cached for the image and the same geometry-only
warp that the preview path uses, so planning costs one splat per candidate
(tens of milliseconds) instead of a full generation.

Candidates are built from three magnitude tiers crossed with a handful of
directions. That guarantees the three characters the user cares about -
conservative, balanced and exploratory - differ in the actual camera
configuration and not only in their label.

Scoring is a transparent weighted mix of geometric facts:

  retention  share of the new frame that still comes from the original photo
  exposure  penalty for the share of the frame the inpainter has to invent
  crop      penalty for the share of the original frame that leaves the view
  spread    reward for an invented area that is one coherent region
  motion    reward for actually moving the camera

No semantic claim is made anywhere here: the repository has no object
detector, so the planner cannot and does not try to find a subject.
"""
from __future__ import annotations

import numpy as np
import torch

from ._quality import (DEFAULT_STRATEGY, STRATEGIES, composition_score,
                        mask_concentration, motion_ratio)
from ._warp import build_intrinsics, forward_warp_full, slider_to_center
from .config import CANON_SHIFT, FOV_H_DEG, MAX_SHIFT

# Number of variants the UI generates by default and the hard ceiling.
DEFAULT_VARIANTS = 3
MAX_VARIANTS = 6

# Magnitude tiers, expressed as multiples of the canonical shift. The three
# tiers are what the "conservative / balanced / exploratory" labels mean.
TIERS = (("Conservative", 0.5), ("Balanced", 1.0), ("Exploratory", 1.7))

# Directions the camera may move, as unit-ish vectors in the slider
# convention: X right, Y down, Z forward. Normalised before scaling.
DIRECTIONS = (
    ("lateral", (1.00, 0.00, 0.35)),
    ("diagonal", (0.75, -0.35, 0.50)),
    ("vertical", (0.20, -0.90, 0.30)),
    ("forward", (0.15, 0.00, 1.00)),
    ("counter-left", (-1.00, 0.00, 0.35)),
)

# How the tiers are ordered when fewer than three variants are requested.
TIER_ORDER = {
    "preserve": ("Conservative", "Balanced", "Exploratory"),
    "balanced": ("Balanced", "Conservative", "Exploratory"),
    "explore": ("Exploratory", "Balanced", "Conservative"),
}


def seed_shifts() -> list[tuple[str, str, tuple[float, float, float]]]:
    """The candidate list as (tier, direction, shift) triples.

    The frozen canonical shift is included once, in the balanced tier, so the
    evaluated protocol stays reachable from the planner.
    """
    canon = tuple(float(v) for v in CANON_SHIFT)
    mag = float(np.linalg.norm(canon))
    out: list[tuple[str, str, tuple[float, float, float]]] = []
    for tier, scale in TIERS:
        for name, vec in DIRECTIONS:
            if name == "canonical":
                continue
            v = np.asarray(vec, dtype=float)
            v = v / np.linalg.norm(v) * mag * scale
            out.append((tier, name, (float(v[0]), float(v[1]), float(v[2]))))
    out.append(("Balanced", "canonical", canon))
    return out


def geometry_of(res: dict, shift: tuple[float, float, float]) -> dict:
    """Collect the geometric facts of one warped candidate."""
    mask = res["mask"]
    exposure = float((mask > 0).mean())
    return {
        "exposure_frac": exposure,
        "retention": float(res["covered_frac"]),
        "crop_frac": float(res.get("out_of_frame_frac", 0.0)),
        "spread": mask_concentration(mask),
        "motion": motion_ratio(shift),
        "hole_pct": round(exposure * 100.0, 1),
        "covered_frac": round(float(res["covered_frac"]), 4),
        "crop_pct": round(float(res.get("out_of_frame_frac", 0.0)) * 100.0, 1),
        "flow_med_px": round(float(res["flow_med_px"]), 1),
    }


def score_all(rgb: torch.Tensor, z: torch.Tensor,
              strategy: str = DEFAULT_STRATEGY) -> list[dict]:
    """Warp and score every candidate movement. Best first.

    This is the expensive part of planning: one geometry-only warp per
    candidate, no diffusion involved.
    """
    if strategy is None:
        strategy = DEFAULT_STRATEGY
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}'; "
                         f"expected one of {sorted(STRATEGIES)}")

    h, w = int(z.shape[-2]), int(z.shape[-1])
    K = build_intrinsics(w, h, FOV_H_DEG)
    z_med = float(z.median())

    rows: list[dict] = []
    for tier, direction, shift in seed_shifts():
        if max(abs(v) for v in shift) > MAX_SHIFT:
            continue                      # never propose a move the app rejects
        res = forward_warp_full(rgb, z, K, slider_to_center(shift, z_med))
        geom = geometry_of(res, shift)
        score, factors = composition_score(geom, strategy)
        rows.append({
            "tier": tier,
            "label": tier,
            "direction": direction,
            "shift": [round(shift[0], 4), round(shift[1], 4), round(shift[2], 4)],
            "score": round(score, 1),
            "factors": {k: round(v, 4) for k, v in factors.items()},
            "stats": geom,
            "reason": (f"keeps {geom['covered_frac']*100:.0f}% of the frame, "
                       f"invents {geom['hole_pct']:.1f}%, "
                       f"crops {geom['crop_pct']:.1f}% away"),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


def plan_candidates(rgb: torch.Tensor, z: torch.Tensor,
                    count: int = DEFAULT_VARIANTS,
                    strategy: str = DEFAULT_STRATEGY) -> dict:
    """Score every candidate movement and pick the ones to generate.

    `rgb` is (3,H,W) float32 in [0,1] and `z` is the matching (H,W)
    pseudo-depth map - the same pair the preview path uses, so the plan
    reflects the geometry the user will actually see.

    Returns the selected `candidates` and the full `ranking` they were picked
    from, so the UI can show what else was considered.
    """
    count = max(1, min(int(count or DEFAULT_VARIANTS), MAX_VARIANTS))
    if strategy is None:
        strategy = DEFAULT_STRATEGY
    rows = score_all(rgb, z, strategy)
    return {"candidates": select_candidates(rows, count, strategy),
            "ranking": rows}



def select_candidates(ranked: list[dict], count: int,
                      strategy: str = DEFAULT_STRATEGY) -> list[dict]:
    """Pick the variants to generate from the scored candidates.

    The best candidate of every tier is taken first so the set always spans
    the conservative/balanced/exploratory characters; the remaining slots are
    filled with the globally best candidates that are not already in.
    """
    chosen: list[dict] = []
    used: set[int] = set()

    for tier in TIER_ORDER.get(strategy, TIER_ORDER[DEFAULT_STRATEGY]):
        for i, row in enumerate(ranked):
            if i not in used and row["tier"] == tier:
                chosen.append(row)
                used.add(i)
                break

    for i, row in enumerate(ranked):
        if len(chosen) >= count:
            break
        if i not in used:
            chosen.append(row)
            used.add(i)
    return chosen[:count]



__all__ = [
    "DEFAULT_VARIANTS", "DIRECTIONS", "MAX_VARIANTS", "TIERS", "TIER_ORDER",
    "geometry_of", "plan_candidates", "score_all", "seed_shifts",
    "select_candidates",
]
