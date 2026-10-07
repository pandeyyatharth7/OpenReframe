"""02_inpaint_sdxl.py — profile SDXL inpainting candidates on the RTX 4060.

Variant A (--variant inpaint):     SDXL-inpaint, no ControlNet.
Variant C (--variant controlnet):  SDXL-inpaint + SDXL depth-ControlNet.
--slice enables attention slicing for the empirical mitigation comparison.
Also proves mid-run cancellation (interrupt between denoising steps) and runs
the deliberate OOM probe (variant A only).

Pre-committed selection rule lives in docs/decisions.md D0.1 — not modified here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import OUTPUTS_DIR, Bench, deliberate_oom
from _prepwarp import (GUIDANCE, NEGATIVE_PROMPT, NEUTRAL_PROMPT, SEED,
                       STEPS, fit_resolution, get_pairs, load_triplet,
                       local_or_repo)

REPO = Path(__file__).resolve().parents[1]
BASE = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CN_SDXL = "diffusers/controlnet-depth-sdxl-1.0"
OUT_DIR = OUTPUTS_DIR / "inpaint"


def build_pipeline(variant: str):
    from diffusers import (ControlNetModel, StableDiffusionXLControlNetInpaintPipeline,
                           StableDiffusionXLInpaintPipeline)
    base_path = local_or_repo(BASE)
    if variant == "inpaint":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base_path, torch_dtype=torch.float16, variant="fp16")
        return pipe
    cn = ControlNetModel.from_pretrained(local_or_repo(CN_SDXL),
                                         torch_dtype=torch.float16,
                                         variant="fp16")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_path, controlnet=cn, torch_dtype=torch.float16, variant="fp16")
    return pipe


def run(variant: str, use_slice: bool) -> None:
    tag = {"inpaint": "A_sdxl_inpaint",
           "controlnet": "C_sdxl_cn"}[variant] + ("_slice" if use_slice else "")
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = Bench(script="02_inpaint_sdxl", model=BASE + ("+CN-depth" if variant == "controlnet" else ""),
                  precision="fp16", steps=str(STEPS),
                  scheduler="DPMSolverMultistep+Karras")

    with bench.stage("load"):
        pipe = build_pipeline(variant)
        pipe.scheduler = __import__("diffusers").DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        if use_slice:
            pipe.enable_attention_slicing("auto")

    pairs = get_pairs()
    times = []
    for name, pair in pairs.items():
        warped_pil = Image.open(pair["warped"])
        w, h = warped_pil.size
        nw, nh = fit_resolution(w, h, cap=1024)
        img_pil, mask_pil, cond_pil = load_triplet(pair, nw, nh)
        gen = torch.Generator("cpu").manual_seed(SEED)
        kw = dict(prompt=NEUTRAL_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                  image=img_pil, mask_image=mask_pil, width=nw, height=nh,
                  num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                  generator=gen)
        if variant == "controlnet":
            kw["control_image"] = cond_pil
            kw["controlnet_conditioning_scale"] = 0.9
        with bench.stage(f"gen_{name}", resolution=f"{nw}x{nh}") as st:
            out = pipe(**kw).images[0]
            st.update(hole_pct=round(100 * (np.asarray(mask_pil) > 127).mean(), 1))
        out.save(out_dir / f"{Path(name).stem}.png")
        times.append(bench.rows[-1]["wall_s"])

    t = np.array(times)
    print(f"[cand {tag}] p50={np.percentile(t, 50):.1f}s p95={np.percentile(t, 95):.1f}s "
          f"over {len(t)} imgs @<=1024px")

    if variant == "inpaint" and not use_slice:
        # cancellation proof: interrupt between denoising steps
        name, pair = next(iter(pairs.items()))
        warped_pil = Image.open(pair["warped"])
        nw, nh = fit_resolution(*warped_pil.size, cap=1024)
        img_pil, mask_pil, _ = load_triplet(pair, nw, nh)
        gen = torch.Generator("cpu").manual_seed(SEED)

        state = {"n": 0}
        def cb(pipe, i, t_, kw):
            state["n"] = i + 1
            if i == 9:
                pipe._interrupt = True
            return kw

        with bench.stage("cancel_test", resolution=f"{nw}x{nh}") as st:
            t0 = time.perf_counter()
            pipe(prompt=NEUTRAL_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                 image=img_pil, mask_image=mask_pil, width=nw, height=nh,
                 num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                 generator=gen, callback_on_step_end=cb)
            elapsed = time.perf_counter() - t0
            st.update(steps_ran=state["n"], full_run_p50_s=round(float(np.percentile(t, 50)), 2))
        print(f"[cancel] stopped after {state['n']}/{STEPS} steps in {elapsed:.1f}s "
              f"(full run ~{np.percentile(t, 50):.1f}s) -> cancellation feasible")
        torch.cuda.empty_cache()

        deliberate_oom(bench)

    bench.finish(notes=f"cand={tag}")
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["inpaint", "controlnet"], required=True)
    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()
    run(a.variant, a.slice)
