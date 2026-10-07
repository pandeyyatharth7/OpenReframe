"""Relative-depth estimation (Depth Anything V2 ViT-L).

Adopted in R1. The checkpoint emits *disparity-like* relative depth, which is
converted to unitless pseudo-depth by `depth_rel_to_pseudo_z` and median
normalized, so absolute metric scale is never required (D0.3).
"""
from __future__ import annotations

import hashlib
import threading

import cv2
import numpy as np
import torch
from PIL import Image

from ._warp import depth_rel_to_pseudo_z
from .config import (DEPTH_MODEL, DEPTH_SMOOTH_SIGMA_RANGE,
                     DEPTH_SMOOTH_SIGMA_SPACE, device)


class DepthEstimator:
    """Lazily-loaded ViT-L depth model with a small per-image result cache.

    Slider drags re-warp the same image many times per second, so depth is
    cached by content hash; a drag costs one splat, not one ViT-L forward pass.
    """

    def __init__(self, model_id: str = DEPTH_MODEL, cache_size: int = 4) -> None:
        self.model_id = model_id
        self.cache_size = cache_size
        self._proc = None
        self._model = None
        self._dev = device()
        self._lock = threading.Lock()
        self._cache: dict[str, torch.Tensor] = {}
        self._order: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Idempotent. Uses fp16 only on CUDA; CPU wants fp32."""
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            dtype = torch.float16 if self._dev == "cuda" else torch.float32
            self._proc = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                self.model_id, dtype=dtype).to(self._dev).eval()

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._proc = None
            self._cache.clear()
            self._order.clear()
            if self._dev == "cuda":
                torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------
    def _key(self, img: Image.Image) -> str:
        return hashlib.sha256(img.tobytes()).hexdigest()

    def pseudo_z(self, img: Image.Image) -> torch.Tensor:
        """(H,W) float32 pseudo-depth for an RGB image, median-normalized."""
        k = self._key(img)
        with self._lock:
            hit = self._cache.get(k)
        if hit is not None:
            return hit

        self.load()
        w, h = img.size
        inp = self._proc(images=img, return_tensors="pt").to(self._dev)
        with torch.inference_mode():
            pred = self._model(**inp).predicted_depth.squeeze().float().cpu()
        # DA-V2 emits a lower-resolution map; bicubic upsample to image size.
        full = torch.nn.functional.interpolate(
            pred[None, None], size=(h, w), mode="bicubic",
            align_corners=False).squeeze()
        z = depth_rel_to_pseudo_z(smooth_relative_depth(full))

        with self._lock:
            self._cache[k] = z
            self._order.append(k)
            while len(self._order) > self.cache_size:
                self._cache.pop(self._order.pop(0), None)
        return z

    def warm(self, img: Image.Image, runs: int = 1) -> float:
        """Run `runs` warm-up inferences and return the last latency in ms.

    The first call pays cuDNN/kernel autotune; R1's 119 ms warm p95 only holds
        after warm-up, so the UI should call this once on image load.
        """
        import time
        for _ in range(max(1, runs)):
            self.load()
            inp = self._proc(images=img, return_tensors="pt").to(self._dev)
            if self._dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                self._model(**inp)
            if self._dev == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0


def rgb_to_tensor(img: Image.Image) -> torch.Tensor:
    """(3,H,W) float32 in [0,1]."""
    return torch.from_numpy(
        np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def smooth_relative_depth(d: torch.Tensor) -> torch.Tensor:
    """Edge-preserving smoothing of a relative-depth map, normalized to [0,1].

    A forward splat turns leaf- and branch-scale depth noise into large numbers
    of tiny isolated holes. A diffusion inpainter fills each such hole with a
    single flat blob, which is the scattered-cube artifact observed on foliage
    in `outputs/inpaint/A_sdxl_inpaint/`. Bilateral smoothing suppresses the
    noise while leaving true depth discontinuities at silhouettes intact, which
    a Gaussian blur would destroy.
    """
    span = (d.max() - d.min()).clamp_min(1e-6)
    dn = ((d - d.min()) / span).numpy().astype(np.float32)
    out = cv2.bilateralFilter(dn, 0, DEPTH_SMOOTH_SIGMA_RANGE,
                              DEPTH_SMOOTH_SIGMA_SPACE)
    return torch.from_numpy(out)