"""Orchestration: the two public operations the app exposes.

`preview` is geometry-only — depth + warp, no diffusion. It is the operation
that must fit the 2 s SLA (R1 left depth at 119 ms warm p95 against a 2 s
budget), and it renders disocclusions as magenta gaps so the user sees the
geometry cost of a shift before paying for inpainting.

`reframe` is the full chain and takes seconds, so it is queued and cancellable.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image

from . import config
from ._depth import DepthEstimator, rgb_to_tensor
from ._inpaint import GenerationCancelled, Inpainter
from ._quality import evaluate, mask_concentration, motion_ratio
from ._smart import plan_candidates
from ._warp import (build_intrinsics, forward_warp_full,
                    render_preview_with_gaps, slider_to_center, validate_warp)

MAX_SHIFT = config.MAX_SHIFT


def _to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _cond_to_u8(dense_z: torch.Tensor) -> np.ndarray:
    """Percentile-stretched 8-bit depth map for ControlNet conditioning."""
    cn = dense_z.cpu().numpy()
    lo, hi = np.percentile(cn, 2), np.percentile(cn, 98)
    return (np.clip((cn - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


class Reframer:
    """Owns the models and the GPU lock. One instance per process."""

    def __init__(self) -> None:
        self.depth = DepthEstimator()
        self.inpaint = Inpainter()
        self._gpu_lock = threading.RLock()
        self._dev = config.device()

    # -- lifecycle ---------------------------------------------------------
    def load(self, with_inpaint: bool = True) -> dict:
        t0 = time.perf_counter()
        self.depth.load()
        if with_inpaint:
            self.inpaint.load()
        return {"device": self._dev, "load_s": round(time.perf_counter() - t0, 2),
                "depth_loaded": self.depth.loaded,
                "inpaint_loaded": self.inpaint.loaded}

    @property
    def ready(self) -> bool:
        return self.depth.loaded

    def health(self) -> dict:
        cn = config.BACKEND_CONTROLNET or "none"
        h = {"device": self._dev, "depth_loaded": self.depth.loaded,
             "inpaint_loaded": self.inpaint.loaded,
             "backend": f"{config.BACKEND_BASE}+{cn}",
             "cap_px": config.CAP_PX,
             "vram_ceil_mb": config.VRAM_CEIL_MB}
        if self._dev == "cuda" and torch.cuda.is_available():
            h["vram_alloc_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            h["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
        return h

    # -- shared front half -------------------------------------------------
    def depth_for(self, img: Image.Image) -> torch.Tensor:
        """Pseudo-depth for an image, served from the estimator's cache.

        The cache is keyed by image content, so a slider drag or a candidate
        plan costs a splat, not another ViT-L forward pass.
        """
        self.depth.load()
        return self.depth.pseudo_z(img)

    def _warp(self, img: Image.Image, z: torch.Tensor,
              shift: tuple[float, float, float]) -> dict:
        w, h = img.size
        C = slider_to_center(shift, float(z.median()))
        rgb = rgb_to_tensor(img)
        res = forward_warp_full(rgb, z, build_intrinsics(w, h, config.FOV_H_DEG), C)
        ok, why = validate_warp(res)
        if not ok:
            raise ValueError(f"shift rejected: {why}")
        return res

    @staticmethod
    def _triplet(res: dict, nw: int, nh: int):
        """Downscale warped/mask/cond to the generation size."""
        warped = Image.fromarray(_to_u8(res["warped_filled"])).resize(
            (nw, nh), Image.LANCZOS)
        mask = Image.fromarray(res["mask"]).resize((nw, nh), Image.NEAREST)
        cond = Image.fromarray(_cond_to_u8(res["dense_z"])).resize(
            (nw, nh), Image.BILINEAR)
        return warped, mask, cond

    # -- public ops --------------------------------------------------------
    def preview(self, img: Image.Image, shift: tuple[float, float, float]) -> dict:
        """Geometry-only render. No diffusion — must fit the 2 s SLA."""
        if max(abs(s) for s in shift) > MAX_SHIFT:
            raise ValueError(f"shift out of range (max |component| = {MAX_SHIFT})")
        t0 = time.perf_counter()
        with self._gpu_lock:
            z = self.depth_for(img)
            res = self._warp(img, z, shift)
        elapsed = time.perf_counter() - t0
        warped_u8 = _to_u8(res["warped_filled"])
        return {
            "warped": Image.fromarray(warped_u8),
            "mask": Image.fromarray(res["mask"]),
            "overlay": Image.fromarray(
                render_preview_with_gaps(res["warped_filled"], res["mask"])),
            "stats": {
                "hole_pct": round(float((res["mask"] > 0).mean()) * 100, 1),
                "covered_frac": round(res["covered_frac"], 4),
                "behind_frac": round(res["behind_frac"], 4),
                "flow_med_px": round(res["flow_med_px"], 1),
                "flow_max_px": round(res["flow_max_px"], 1),
                "elapsed_s": round(elapsed, 3),
                "within_sla": elapsed <= config.PREVIEW_SLA_S,
            },
        }

    def plan(self, img: Image.Image, count: int = 3,
             strategy: str = "balanced") -> dict:
        """Propose and rank camera movements. Geometry only, no diffusion.

        Reuses the cached depth map, so planning costs one warp per candidate
        and no additional model inference.
        """
        t0 = time.perf_counter()
        with self._gpu_lock:
            z = self.depth_for(img)
            rgb = rgb_to_tensor(img)
            # Score every candidate once, then pick the ones to generate from
            # that ranking.
            plan = plan_candidates(rgb, z, count=count, strategy=strategy)
        return {
            "candidates": plan["candidates"],
            "ranking": plan["ranking"],
            "stats": {"count": len(plan["candidates"]), "strategy": strategy,
                      "elapsed_s": round(time.perf_counter() - t0, 3)},
        }

    def reframe(self, img: Image.Image, shift: tuple[float, float, float],
                on_step=None, cancel: threading.Event | None = None,
                watchdog_enabled: bool = True,
                mask_dilate_px: int = 0) -> dict:
        """Full chain: depth -> warp -> conditioned inpainting."""
        if max(abs(s) for s in shift) > MAX_SHIFT:
            raise ValueError(f"shift out of range (max |component| = {MAX_SHIFT})")
        with self._gpu_lock:
            z = self.depth_for(img)
            res = self._warp(img, z, shift)
            nw, nh = config.fit_resolution(img.size[0], img.size[1], config.CAP_PX)
            warped, mask, cond = self._triplet(res, nw, nh)
            if mask_dilate_px > 0:
                # Widening the mask hands the seam ring to the inpainter
                # instead of leaving the nearest-valid placeholder there.
                # The VAE sees the mask at 1/8 resolution, so the widening has
                # to be several pixels to survive the downscale.
                grown = cv2.dilate((np.asarray(mask) > 127).astype(np.uint8),
                                   np.ones((3, 3), np.uint8),
                                   iterations=mask_dilate_px)
                mask = Image.fromarray((grown * 255).astype(np.uint8))
            self.inpaint.load()
            gen = self.inpaint.generate(
                warped, mask, cond, nw, nh, on_step=on_step, cancel=cancel,
                watchdog_enabled=watchdog_enabled)
        hole_pct = float((np.asarray(mask) > 127).mean()) * 100.0
        # The geometric facts the composition score is built from.
        geom = {
            "exposure_frac": hole_pct / 100.0,
            "retention": float(res["covered_frac"]),
            "crop_frac": float(res.get("out_of_frame_frac", 0.0)),
            "spread": mask_concentration(mask),
            "motion": motion_ratio(shift),
        }
        stats = {
            "hole_pct": round(hole_pct, 1),
            "width": gen["width"], "height": gen["height"],
            "steps_run": gen["steps_run"], "total_steps": gen["total_steps"],
            "elapsed_s": gen["elapsed_s"], "watchdog": gen["watchdog"],
            "cancelled": False,
            "covered_frac": round(float(res["covered_frac"]), 4),
            "crop_pct": round(float(res.get("out_of_frame_frac", 0.0)) * 100, 1),
            "seed": config.SEED,
        }
        report = evaluate(warped, gen["image"], mask, geom)
        return {
            "filled": gen["image"],
            "warped": warped,
            "mask": mask,
            "overlay": Image.fromarray(render_preview_with_gaps(
                res["warped_filled"], res["mask"])),
            "stats": stats,
            "geom": geom,
            "report": report,
        }


__all__ = ["Reframer", "GenerationCancelled", "MAX_SHIFT"]
