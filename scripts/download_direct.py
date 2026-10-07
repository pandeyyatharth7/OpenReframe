"""Direct-streaming model downloader (bypasses hf-xet stalls).

Downloads each repo into <repo>/models/direct/<org>__<name>/ with normal file
structure so diffusers/transformers can load it via a plain local path.
Chunked GET with Range-based resume + size verification.
"""
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "prototypes"))
from _bench import MODELS_DIR  # noqa: E402

ROOT = MODELS_DIR / "direct"
SKIP = (".msgpack", ".h5", ".onnx", ".onnx_data", ".bin", ".ckpt")
SKIP_DIRS = ("onnx/", "openvino/", "flax/", "tf_", "vae/diffusion_pytorch_model.bin")

GROUPS = {
    "inpaint_b": ["stable-diffusion-v1-5/stable-diffusion-inpainting",
                  "lllyasviel/control_v11f1p_sd15_depth"],
    "inpaint_ac": ["diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                   "diffusers/controlnet-depth-sdxl-1.0"],
}

CHUNK = 1 << 20


def fetch(repo_id: str, fp16_only: bool = False) -> None:
    api = HfApi()
    files = api.list_repo_files(repo_id)

    def want(f: str) -> bool:
        if f.endswith(SKIP):
            return False
        if any(f.startswith(d) or f"/{d}" in f + "/" for d in SKIP_DIRS):
            return False
        if "safety_checker" in f:
            return False
        # Every OpenReframe load path passes variant="fp16" and runs fp16 on
        # CUDA, so the fp32 copies are never read -- they only double the
        # transfer. Config files are always kept.
        if fp16_only and f.endswith(".safetensors") \
                and not f.endswith(".fp16.safetensors"):
            return False
        return True

    keep = [f for f in files if want(f)]
    short = repo_id.replace("/", "__")
    print(f"[{repo_id}] {len(keep)} files", flush=True)
    for f in keep:
        url = hf_hub_url(repo_id, f)
        md = get_hf_file_metadata(url)
        total = md.size
        dest = ROOT / short / f
        if dest.exists() and dest.stat().st_size == total:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        pos = part.stat().st_size if part.exists() else 0
        if pos >= total and total > 0:
            part.rename(dest)
            continue
        t0 = time.time()
        for attempt in range(6):
            try:
                headers = {"Range": f"bytes={pos}-"} if pos else {}
                with requests.get(md.location, headers=headers, stream=True,
                                  timeout=(15, 60)) as r:
                    r.raise_for_status()
                    mode = "ab" if pos else "wb"
                    with open(part, mode) as fh:
                        for chunk in r.iter_content(CHUNK):
                            fh.write(chunk)
                            pos += len(chunk)
                if pos >= total:
                    part.rename(dest)
                    dt = time.time() - t0
                    print(f"  ok {f} [{total/2**20:.0f}MB "
                          f"{total/2**20/max(dt,0.1):.1f}MB/s]", flush=True)
                    break
            except Exception as e:
                print(f"  retry{attempt} {f} @ {pos/2**20:.0f}MB: {type(e).__name__}",
                      flush=True)
                time.sleep(2 * attempt + 1)
        else:
            raise RuntimeError(f"failed {repo_id}/{f}")
    print(f"[{repo_id}] complete", flush=True)


if __name__ == "__main__":
    group = sys.argv[1]
    fp16_only = "--fp16-only" in sys.argv[2:]
    for rid in GROUPS[group]:
        fetch(rid, fp16_only=fp16_only)
