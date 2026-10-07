import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench  # noqa: F401
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

mid = "depth-anything/Depth-Anything-V2-Large-hf"
img = Image.open("assets/test_images/out04.jpg").convert("RGB")
proc = AutoImageProcessor.from_pretrained(mid)
dm = AutoModelForDepthEstimation.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
inp = proc(images=img, return_tensors="pt").to("cuda")
with torch.inference_mode():
    pred = dm(**inp).predicted_depth.squeeze().float().cpu()
print("pred shape:", tuple(pred.shape))
p = pred.numpy()
print("nans:", int(np.isnan(p).sum()), "zeros:", int((p == 0).sum()),
      "min/max:", float(np.nanmin(p)), float(np.nanmax(p)))
m = (p <= 0).astype(np.uint8)
if m.sum() > 0:
    ys, xs = np.nonzero(m)
    print("<=0 uv sample:", list(zip(xs[:8].tolist(), ys[:8].tolist())),
          "count", int(m.sum()))
Image.fromarray(((np.nan_to_num(p) / max(np.nanmax(p), 1e-6)) * 255).astype(np.uint8)).save(
    "outputs/dbg_rawpred.png")
print("saved")
