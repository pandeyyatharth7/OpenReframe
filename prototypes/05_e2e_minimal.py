"""05_e2e_minimal.py — minimal end-to-end proof on ONE real image.

depth (ViT-L relative) -> warp (validated hybrid) -> diffusion inpaint.
Console-driven, no API/UI. --pipeline A|B|C selects the inpainting backend;
artifacts land in outputs/e2e/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench  # noqa: F401
from _bench import OUTPUTS_DIR, Bench
from _prepwarp import (GUIDANCE, NEGATIVE_PROMPT, NEUTRAL_PROMPT, SEED,
                       STEPS, local_or_repo)
from _reproj import (build_intrinsics, depth_rel_to_pseudo_z,
                     forward_warp_full, render_preview_with_gaps,
                     slider_to_center)

IMG = Path(__file__).resolve().parents[1] / "assets" / "test_images" / "mix05.jpg"
OUT = OUTPUTS_DIR / "e2e"
SHIFT = (0.18, -0.06, 0.12)

REPOS = {
    "A": ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", None),
    "B": ("stable-diffusion-v1-5/stable-diffusion-inpainting",
          "lllyasviel/control_v11f1p_sd15_depth"),
    "C": ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
          "diffusers/controlnet-depth-sdxl-1.0"),
}
CAPS = {"A": 1024, "B": 768, "C": 1024}


def fit(w, h, cap, mult=64):
    s = min(1.0, cap / max(w, h))
    return max(mult, int(round(w * s / mult)) * mult), max(mult, int(round(h * s / mult)) * mult)


def build(pipeline_id: str):
    from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetInpaintPipeline,
                           StableDiffusionXLControlNetInpaintPipeline,
                           StableDiffusionXLInpaintPipeline)
    base, cn_id = REPOS[pipeline_id]
    base = local_or_repo(base)
    if pipeline_id == "A":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16")
    elif pipeline_id == "C":
        cn = ControlNetModel.from_pretrained(local_or_repo(cn_id),
                                             torch_dtype=torch.float16,
                                             variant="fp16")
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base, controlnet=cn, torch_dtype=torch.float16, variant="fp16")
    else:
        cn = ControlNetModel.from_pretrained(local_or_repo(cn_id),
                                             torch_dtype=torch.float16,
                                             variant="fp16")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base, controlnet=cn, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=list(REPOS), default="B")
    ap.add_argument("--image", default=str(IMG))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = Bench(script="05_e2e_minimal", model=str(REPOS[a.pipeline]),
                  precision="fp16", steps=str(STEPS),
                  scheduler="DPMSolverMultistep+Karras")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    da_id = "depth-anything/Depth-Anything-V2-Large-hf"

    img = Image.open(a.image).convert("RGB")
    w, h = img.size

    with bench.stage("depth"):
        proc = AutoImageProcessor.from_pretrained(da_id)
        dmodel = AutoModelForDepthEstimation.from_pretrained(
            da_id, dtype=torch.float16).to("cuda").eval()
        inp = proc(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            pred = dmodel(**inp).predicted_depth.squeeze().float().cpu()
        del dmodel
        torch.cuda.empty_cache()
        z = depth_rel_to_pseudo_z(
            torch.nn.functional.interpolate(pred[None, None], size=(h, w),
                                            mode="bicubic",
                                            align_corners=False).squeeze())

    with bench.stage("warp") as st:
        res = forward_warp_full(torch.from_numpy(
            np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1),
            z, build_intrinsics(w, h), slider_to_center(SHIFT, float(z.median())))
        mask = res["mask"]
        st.update(hole_pct=round(100 * (mask > 0).mean(), 1))
        Image.fromarray(render_preview_with_gaps(res["warped_filled"], mask)
                        ).save(OUT / "warped_gaps.png")

    nw, nh = fit(w, h, CAPS[a.pipeline])
    to_pil = lambda t8: Image.fromarray(t8)
    warped_pil = to_pil((res["warped_filled"].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    mask_pil = Image.fromarray(mask).resize((nw, nh), Image.NEAREST)
    cond_np = res["dense_z"].numpy()
    lo, hi = np.percentile(cond_np, 2), np.percentile(cond_np, 98)
    cond_pil = to_pil((np.clip((cond_np - lo) / max(hi - lo, 1e-6), 0, 1) * 255
                       ).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)

    with bench.stage("load_inpaint"):
        pipe = build(a.pipeline)

    gen = torch.Generator("cpu").manual_seed(SEED)
    kw = dict(prompt=NEUTRAL_PROMPT, negative_prompt=NEGATIVE_PROMPT,
              image=warped_pil, mask_image=mask_pil, width=nw, height=nh,
              num_inference_steps=STEPS, guidance_scale=GUIDANCE,
              generator=gen)
    if a.pipeline != "A":
        kw["control_image"] = cond_pil
        kw["controlnet_conditioning_scale"] = 0.9
    with bench.stage("generate", resolution=f"{nw}x{nh}") as st:
        filled = pipe(**kw).images[0]
        st.update(hole_pct=round(100 * (np.asarray(mask_pil) > 127).mean(), 1))

    orig_r = img.resize((nw, nh), Image.LANCZOS)
    warped_r = warped_pil.copy()
    warr = np.array(warped_r)
    m = np.array(mask_pil) > 127
    warr[m] = (255, 0, 255)
    panel = Image.new("RGB", (nw * 2 + 8, nh * 2 + 8), (20, 20, 20))
    panel.paste(orig_r, (0, 0))
    panel.paste(Image.fromarray(warr), (nw + 8, 0))
    panel.paste(filled, (0, nh + 8))
    panel.paste(Image.open(OUT / "warped_gaps.png").resize((nw // 2, nh // 2)),
                (nw + 8, nh + 8))
    panel.save(OUT / f"panel_{a.pipeline}.png")
    filled.save(OUT / f"filled_{a.pipeline}.png")
    print(f"[e2e] saved outputs/e2e/panel_{a.pipeline}.png "
          f"(orig | warped+gaps | filled | preview)")
    bench.finish(notes=f"pipeline={a.pipeline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
