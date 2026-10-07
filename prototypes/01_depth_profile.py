"""01_depth_profile.py — profile Depth Anything V2 ViT-S/B/L (relative depth).

Measures load time, per-image inference latency (warm), peak VRAM per the bench
harness, saves normalized 8-bit depth previews, and prints cross-variant
agreement so the ViT-L-vs-smaller preview-budget call is evidence-based.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import OUTPUTS_DIR, Bench

VARIANTS = [
    ("vit-s", "depth-anything/Depth-Anything-V2-Small-hf"),
    ("vit-b", "depth-anything/Depth-Anything-V2-Base-hf"),
    ("vit-l", "depth-anything/Depth-Anything-V2-Large-hf"),
]
IMG_DIR = Path(__file__).resolve().parents[1] / "assets" / "test_images"
DEPTH_OUT = OUTPUTS_DIR / "depth"


def run_variant(tag: str, model_id: str, image_paths: list[Path]) -> dict[str, np.ndarray]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    bench = Bench(script="01_depth_profile", model=model_id, precision="fp16")
    depths: dict[str, np.ndarray] = {}
    with bench.stage("load") as st:
        t0 = time.perf_counter()
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(
            model_id, dtype=torch.float16).to("cuda").eval()
        st.update(seconds=time.perf_counter() - t0)

    # warmup (not recorded as a row)
    im0 = Image.open(image_paths[0]).convert("RGB")
    inp = proc(images=im0, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model(**inp)
    torch.cuda.synchronize()

    lat = []
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        inp = proc(images=img, return_tensors="pt").to("cuda")
        with bench.stage(f"infer_{i:02d}", resolution=f"{w}x{h}") as st:
            with torch.inference_mode():
                out = model(**inp)
            pred = out.predicted_depth.squeeze().float().cpu()
        lat.append(bench.rows[-1]["wall_s"])
        d = torch.nn.functional.interpolate(
            pred[None, None], size=(h, w), mode="bicubic",
            align_corners=False).squeeze()
        dn = (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)
        arr = (dn.numpy() * 255).astype(np.uint8)
        depths[p.name] = dn.numpy()
        out_dir = DEPTH_OUT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_dir / f"{p.stem}.png")

    lat_arr = np.array(lat[1:]) if len(lat) > 1 else np.array(lat)
    print(f"[depth] {tag}: warm p50={np.percentile(lat_arr, 50)*1000:.0f}ms "
          f"p95={np.percentile(lat_arr, 95)*1000:.0f}ms over {len(lat_arr)} imgs")
    bench.finish(notes=f"warm_p50_ms={np.percentile(lat_arr,50)*1000:.0f} "
                       f"warm_p95_ms={np.percentile(lat_arr,95)*1000:.0f}")

    del model, proc
    torch.cuda.empty_cache()
    return depths


def agreement(ref: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> float:
    """Mean Pearson correlation between variant depth maps (downsampled)."""
    cs = []
    for name, a in ref.items():
        b = other[name]
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        a8 = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w // 4, h // 4)))
        b8 = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((w // 4, h // 4)))
        cs.append(np.corrcoef(a8.ravel(), b8.ravel())[0, 1])
    return float(np.mean(cs))


if __name__ == "__main__":
    paths = sorted(IMG_DIR.glob("*.jpg"))
    assert paths, "no test images found"
    results = {}
    for tag, mid in VARIANTS:
        results[tag] = run_variant(tag, mid, paths)

    print("\n=== cross-variant agreement (mean Pearson r vs ViT-L) ===")
    r_s = agreement(results["vit-l"], results["vit-s"])
    r_b = agreement(results["vit-l"], results["vit-b"])
    print(f"S vs L: {r_s:.4f}")
    print(f"B vs L: {r_b:.4f}")

    # contact sheet of one indoor + one outdoor for visual QA
    names = ["ind01.jpg", "out03.jpg"]
    tiles = []
    for n in names:
        row = []
        for tag, _ in VARIANTS:
            p = DEPTH_OUT / tag / (Path(n).stem + ".png")
            if p.exists():
                row.append(np.asarray(Image.open(p).convert("L").resize((320, 240))))
        tiles.append(np.hstack(row))
    sheet = np.vstack(tiles)
    Image.fromarray(sheet).save(DEPTH_OUT / "contact_sheet.png")
    print(f"contact sheet -> {DEPTH_OUT / 'contact_sheet.png'}")
