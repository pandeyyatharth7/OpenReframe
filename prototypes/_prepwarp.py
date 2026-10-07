"""Shared prep for inpainting-candidate profiling: produces identical
(warped image, binary disocclusion mask, conditioning depth) triples for every
candidate by running the validated Sprint-0 chain (ViT-L relative depth ->
pseudo-z -> forward-splat warp -> mask) over the eval set. Cached on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench  # noqa: F401  (sets HF_HUB_CACHE into <repo>/models before any HF call)
from _reproj import (build_intrinsics, depth_rel_to_pseudo_z, forward_warp_full,
                     slider_to_center)

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "assets" / "test_images"
PAIRS_DIR = REPO / "outputs" / "warppairs"

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
CANON_SHIFT = (0.18, -0.06, 0.12)
NEUTRAL_PROMPT = "a high quality, detailed photograph"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text"
SEED = 0
STEPS = 25
GUIDANCE = 5.5

EVAL_IMAGES = ["ind01.jpg", "out02.jpg", "mix01.jpg", "mix05.jpg",
               "out04.jpg", "mix06.jpg"]


def local_or_repo(repo_id: str) -> str:
    """Prefer locally downloaded weights (models/direct/<org>__<name>) over HF id."""
    p = _bench.MODELS_DIR / "direct" / repo_id.replace("/", "__")
    ok = (p / "model_index.json").exists() or (p / "config.json").exists()
    return str(p) if ok else repo_id


def get_pairs() -> dict[str, dict[str, Path]]:
    """Returns {image_name: {warped, mask, cond, orig}} paths."""
    if all((PAIRS_DIR / n / "mask.png").exists() for n in EVAL_IMAGES):
        return {n: _paths_for(n) for n in EVAL_IMAGES}

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL, dtype=torch.float16).to("cuda").eval()

    for name in EVAL_IMAGES:
        img = Image.open(IMG_DIR / name).convert("RGB")
        w, h = img.size
        inp = proc(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            pred = model(**inp).predicted_depth.squeeze().float().cpu()
        d_full = torch.nn.functional.interpolate(
            pred[None, None], size=(h, w), mode="bicubic",
            align_corners=False).squeeze()
        z = depth_rel_to_pseudo_z(d_full)
        rgb = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)
        Cen = slider_to_center(CANON_SHIFT, float(z.median()))
        res = forward_warp_full(rgb, z, build_intrinsics(w, h), Cen)
        mask = res["mask"]
        cond = res["dense_z"]

        out = PAIRS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        warped_u8 = (res["warped_filled"].clamp(0, 1).permute(1, 2, 0).numpy() * 255
                     ).astype(np.uint8)
        Image.fromarray(warped_u8).save(out / "warped.png")
        Image.fromarray(mask).save(out / "mask.png")
        cn = cond.numpy()
        lo, hi = np.percentile(cn, 2), np.percentile(cn, 98)
        cn8 = (np.clip((cn - lo) / max(hi - lo, 1e-6), 0, 1) * 255
               ).astype(np.uint8)
        Image.fromarray(cn8).save(out / "cond_depth.png")
        img.save(out / "orig.png")
        hf = 100.0 * (mask > 0).mean()
        print(f"[prep] {name}: hole+ring {hf:.1f}% of pixels")
    del model, proc
    torch.cuda.empty_cache()
    return {n: _paths_for(n) for n in EVAL_IMAGES}


def _paths_for(name: str) -> dict[str, Path]:
    d = PAIRS_DIR / name
    return {"warped": d / "warped.png", "mask": d / "mask.png",
            "cond": d / "cond_depth.png", "orig": d / "orig.png"}


def fit_resolution(w: int, h: int, cap: int, mult: int = 64) -> tuple[int, int]:
    s = min(1.0, cap / max(w, h))
    nw = max(mult, int(round(w * s / mult)) * mult)
    nh = max(mult, int(round(h * s / mult)) * mult)
    return nw, nh


def load_triplet(pair: dict[str, Path], nw: int, nh: int):
    warped = Image.open(pair["warped"]).convert("RGB").resize((nw, nh), Image.LANCZOS)
    mask = Image.open(pair["mask"]).convert("L").resize((nw, nh), Image.NEAREST)
    cond = Image.open(pair["cond"]).convert("L").resize((nw, nh), Image.BILINEAR)
    return warped, mask, cond


if __name__ == "__main__":
    get_pairs()
