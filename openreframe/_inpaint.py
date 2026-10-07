"""Inpainting backend — candidate A (SDXL-inpaint), selected under D0.1 / R5.

SDXL-inpaint with no ControlNet. Candidate C (SDXL + depth-ControlNet) is the
higher fidelity rank in the pre-registered table but exceeds both the VRAM and
the latency gate on this hardware: it ran at a 378.7 s p95 against a 30 s limit
after silently spilling into system RAM. Candidate B (SD1.5 + depth-ControlNet)
fits comfortably but leaves visible netting on textured surfaces.

Two mechanisms cooperate here:
- cancellation, via `pipe._interrupt` observed at a step boundary (the pattern
  proved in prototypes/02_inpaint_sdxl.py);
- stall detection, via `_watchdog.StageWatchdog`, because WDDM sysmem fallback
  means a real OOM never raises — it just crawls.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch
from PIL import Image

from ._watchdog import StageWatchdog
from .config import (BACKEND_BASE, BACKEND_CONTROLNET, CAP_PX,
                     CONTROLNET_SCALE, GEN_P95_S, GUIDANCE, NEGATIVE_PROMPT,
                     NEUTRAL_PROMPT, STEPS, device, local_or_repo)


class GenerationCancelled(Exception):
    """Raised when a caller-requested cancellation stopped denoising."""


class Inpainter:
    """Lazily-loaded candidate A pipeline (ControlNet optional)."""

    def __init__(self, base: str = BACKEND_BASE,
                 controlnet: str | None = BACKEND_CONTROLNET,
                 cap_px: int = CAP_PX) -> None:
        self.base = base
        self.controlnet = controlnet
        self.cap_px = cap_px
        self._pipe = None
        self._lock = threading.Lock()
        self._dev = device()

    # -- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        """Idempotent. cpu-offload is what makes an 8 GB card viable here."""
        with self._lock:
            if self._pipe is not None:
                return
            from diffusers import (ControlNetModel,
                                   DPMSolverMultistepScheduler,
                                   StableDiffusionControlNetInpaintPipeline,
                                   StableDiffusionXLControlNetInpaintPipeline,
                                   StableDiffusionXLInpaintPipeline)
            # fp16 on CPU is both unsupported for some kernels and extremely
            # slow, so the CPU path keeps fp32 (and skips the fp16 variant).
            cuda = self._dev == "cuda"
            dt = torch.float16 if cuda else torch.float32
            kw = dict(torch_dtype=dt)
            if cuda:
                kw["variant"] = "fp16"

            cn = None
            cls = StableDiffusionXLInpaintPipeline
            extra: dict = {}
            if self.controlnet:
                cls = StableDiffusionXLControlNetInpaintPipeline
                cn = ControlNetModel.from_pretrained(
                    local_or_repo(self.controlnet), **kw)
                extra["controlnet"] = cn
            elif "stable-diffusion-v1-5" in self.base:
                cls = StableDiffusionControlNetInpaintPipeline
                cn = ControlNetModel.from_pretrained(
                    local_or_repo("lllyasviel/control_v11f1p_sd15_depth"), **kw)
                extra["controlnet"] = cn

            pipe = cls.from_pretrained(local_or_repo(self.base), **extra,
                                       safety_checker=None,
                                       requires_safety_checker=False, **kw)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True,
                algorithm_type="dpmsolver++")
            if cuda:
                # accelerate's cpu-offload requires a CUDA device.
                pipe.enable_model_cpu_offload()
            for opt in ("enable_vae_tiling", "enable_vae_slicing"):
                if hasattr(pipe, opt):
                    getattr(pipe, opt)()
            self._pipe = pipe

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            if self._dev == "cuda":
                torch.cuda.empty_cache()

    # -- generation --------------------------------------------------------
    def generate(self, warped: Image.Image, mask: Image.Image,
                 cond: Image.Image | None, width: int, height: int,
                 on_step=None, cancel: threading.Event | None = None,
                 expected_p95_s: float = GEN_P95_S,
                 watchdog_enabled: bool = True) -> dict:
        """Fill `mask` in `warped`, optionally depth-conditioned by `cond`.

        `on_step(step_index, total_steps)` is called after every denoising step.
        Setting `cancel` stops at the next step boundary. Raises
        GenerationCancelled; returns a dict describing the run.
        """
        if self._pipe is None:
            raise RuntimeError("pipeline not loaded; call load() first")
        pipe = self._pipe
        uses_cn = getattr(pipe.config, "_class_name", "").find("ControlNet") >= 0
        if uses_cn and cond is None:
            raise ValueError("this pipeline is ControlNet-conditioned; "
                             "a conditioning depth map is required")

        total = STEPS
        state = {"n": 0, "cancelled": False}

        def cb(p, i, _t, kw):
            state["n"] = i + 1
            if on_step is not None:
                on_step(i + 1, total)
            if cancel is not None and cancel.is_set():
                state["cancelled"] = True
                p._interrupt = True
            return kw

        gen = torch.Generator("cpu").manual_seed(0)
        pipe._interrupt = False

        kw = dict(prompt=NEUTRAL_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                  image=warped, mask_image=mask, width=width, height=height,
                  num_inference_steps=total, guidance_scale=GUIDANCE,
                  generator=gen, callback_on_step_end=cb)
        if uses_cn:
            kw["control_image"] = cond
            kw["controlnet_conditioning_scale"] = CONTROLNET_SCALE

        t0 = time.perf_counter()
        with StageWatchdog(expected_p95_s, on_exceed=lambda: setattr(
                pipe, "_interrupt", True), enabled=watchdog_enabled) as wd:
            try:
                out = pipe(**kw).images[0]
            except torch.cuda.OutOfMemoryError:
                # Only reachable once WDDM sysmem fallback is disabled.
                if self._dev == "cuda":
                    torch.cuda.empty_cache()
                raise
            finally:
                pipe._interrupt = False
        elapsed = time.perf_counter() - t0

        if state["cancelled"]:
            raise GenerationCancelled(f"stopped after {state['n']}/{total} steps")

        hole_pct = float((np.asarray(mask.convert("L")) > 127).mean()) * 100.0
        return {"image": out, "steps_run": state["n"], "total_steps": total,
                "width": width, "height": height, "hole_pct": round(hole_pct, 1),
                "elapsed_s": round(elapsed, 3), "watchdog": wd.summary()}