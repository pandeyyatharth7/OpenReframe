"""03_inpaint_sd15_cn.py — Candidate B: SD1.5-inpaint + depth-ControlNet.

Same protocol as 02 (identical pairs, seeds, steps) at SD1.5's sensible
operating resolution (cap 768), per the pre-registered comparison rules in
docs/decisions.md D0.1.
"""
from __future__ import annotations

import sys
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
BASE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
CN_SD15 = "lllyasviel/control_v11f1p_sd15_depth"


def run() -> None:
    from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    tag = "B_sd15_cn"
    out_dir = OUTPUTS_DIR / "inpaint" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = Bench(script="03_inpaint_sd15_cn", model=BASE + "+" + CN_SD15,
                  precision="fp16", steps=str(STEPS),
                  scheduler="DPMSolverMultistep+Karras")

    with bench.stage("load"):
        cn = ControlNetModel.from_pretrained(local_or_repo(CN_SD15),
                                             torch_dtype=torch.float16,
                                             variant="fp16")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            local_or_repo(BASE), controlnet=cn, torch_dtype=torch.float16,
            variant="fp16", safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True,
            algorithm_type="dpmsolver++")
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    pairs = get_pairs()
    times = []
    for name, pair in pairs.items():
        warped_pil = Image.open(pair["warped"])
        nw, nh = fit_resolution(*warped_pil.size, cap=768)
        img_pil, mask_pil, cond_pil = load_triplet(pair, nw, nh)
        gen = torch.Generator("cpu").manual_seed(SEED)
        with bench.stage(f"gen_{name}", resolution=f"{nw}x{nh}") as st:
            try:
                out = pipe(prompt=NEUTRAL_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                           image=img_pil, mask_image=mask_pil,
                           control_image=cond_pil,
                           controlnet_conditioning_scale=0.9,
                           width=nw, height=nh,
                           num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                           generator=gen).images[0]
                st.update(hole_pct=round(100 * (np.asarray(mask_pil) > 127).mean(), 1))
                out.save(out_dir / f"{Path(name).stem}.png")
            except torch.cuda.OutOfMemoryError:
                st.update(oom=True)
                torch.cuda.empty_cache()
                print(f"[cand B] OOM on {name}")
                continue
        times.append(bench.rows[-1]["wall_s"])

    t = np.array(times)
    print(f"[cand {tag}] p50={np.percentile(t, 50):.1f}s p95={np.percentile(t, 95):.1f}s "
          f"over {len(t)} imgs @<=768px")
    bench.finish(notes=f"cand={tag}")

    deliberate_oom(bench)
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
