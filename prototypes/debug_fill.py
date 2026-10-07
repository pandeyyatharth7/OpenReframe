import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproj import build_intrinsics, depth_rel_to_pseudo_z, forward_warp_full

img = Image.open("assets/test_images/out04.jpg").convert("RGB")
w, h = img.size
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
mid = "depth-anything/Depth-Anything-V2-Large-hf"
proc = AutoImageProcessor.from_pretrained(mid)
dm = AutoModelForDepthEstimation.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
inp = proc(images=img, return_tensors="pt").to("cuda")
with torch.inference_mode():
    pred = dm(**inp).predicted_depth.squeeze().float().cpu()
del dm
torch.cuda.empty_cache()
z = depth_rel_to_pseudo_z(torch.nn.functional.interpolate(
    pred[None, None], size=(h, w), mode="bicubic", align_corners=False).squeeze())
rgb = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)
res = forward_warp_full(rgb, z, build_intrinsics(w, h), (0.18, -0.06, 0.12))
raw = (res["warped"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
fil = (res["warped_filled"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
mk = res["mask"]
sky = np.s_[30:150, 480:700]
panel = np.hstack([raw[sky], fil[sky]])
Image.fromarray(panel).resize((1320, 360), Image.NEAREST).save("outputs/dbg_fill_check.png")
print("mask %:", round(100 * (mk > 0).mean(), 1), "| saved crop (raw | filled)")
