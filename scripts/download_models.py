"""Download model weights into <repo>/models via huggingface_hub (resume-safe).

Usage: python scripts/download_models.py depth|inpaint_a|inpaint_b|inpaint_c
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "prototypes"))

import _bench  # noqa: E402,F401  (import side effect: sets HF_HUB_CACHE env)

from huggingface_hub import snapshot_download  # noqa: E402

GROUPS = {
    "depth": [
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything/Depth-Anything-V2-Base-hf",
        "depth-anything/Depth-Anything-V2-Large-hf",
    ],
    "inpaint_a": ["diffusers/stable-diffusion-xl-1.0-inpainting-0.1"],
    "inpaint_b": [
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "lllyasviel/control_v11f1p_sd15_depth",
    ],
    "inpaint_c": [
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "diffusers/controlnet-depth-sdxl-1.0",
    ],
}

IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*.onnx_data", "*openvino*", "*flax*", "*tf_*"]

if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "depth"
    for repo_id in GROUPS[group]:
        print(f"downloading {repo_id} ...", flush=True)
        p = snapshot_download(repo_id, ignore_patterns=IGNORE)
        print(f"done -> {p}", flush=True)
