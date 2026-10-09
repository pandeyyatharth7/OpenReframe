# Windows Setup Guide

This guide describes the validated Windows environment for the local OpenReframe
app. The reference machine is an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM and
32 GB RAM. Other systems may work, but performance and available VRAM will vary.

For an overview and the shortest install path, start with the
[README](../README.md).

## Requirements

| Component | Validated setup |
|---|---|
| OS | Windows 10/11 x64 |
| Python | 3.11.9, 64-bit |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM |
| System RAM | 32 GB (16 GB minimum recommended for CPU offload) |
| Free disk | At least 25 GB for model weights |

Conda and the CUDA Toolkit are not required. The PyTorch CUDA wheel supplies its
runtime. A GPU is strongly recommended; CPU supports geometry checks, but CPU
diffusion is not a supported workflow.

## Create the environment

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The validated package versions are pinned in `requirements.txt`. The PyTorch
installation must use the CUDA wheel index shown above; install it before the
remaining requirements.

Check that Python, PyTorch, and the GPU are detected:

```powershell
& .\.venv\Scripts\python.exe prototypes\00_env_check.py
```

The script should finish with `ENVIRONMENT OK`. It reports individual checks and
warnings if the CUDA device or expected precision support is unavailable.

## Download model weights

Weights are not tracked in Git. The default downloader stores them under the
ignored `models/` directory. Download the active depth model family and selected
SDXL inpainting model:

```powershell
& .\.venv\Scripts\python.exe scripts\download_models.py depth
& .\.venv\Scripts\python.exe scripts\download_models.py inpaint_a
```

The depth group downloads ViT-S, ViT-B, and ViT-L; the app uses ViT-L. The active
inpainting backend is candidate A, SDXL inpainting without ControlNet. The
`inpaint_b` and `inpaint_c` groups are optional research/comparison candidates,
not required to run the app.

If Hugging Face's large-file transport stalls, `scripts\download_direct.py`
supports resumable direct downloads for the inpainting comparison set:

```powershell
& .\.venv\Scripts\python.exe scripts\download_direct.py inpaint_ac --fp16-only
```

That group includes the optional SDXL depth-ControlNet weights in addition to
the active inpainting model. Weights can be stored in `models/hub/` or
`models/direct/`; the application checks the direct-download location before
falling back to the Hugging Face repository ID.

## Run the application

Start in preview-first mode:

```powershell
& .\.venv\Scripts\python.exe -m openreframe.server --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Upload a photograph, adjust the X, Y, and forward
camera shifts, and inspect the geometry preview. Magenta regions mark areas to
be filled. Select **Render final** to run diffusion. Add `--inpaint` to load the
diffusion pipeline on a background thread at startup:

```powershell
& .\.venv\Scripts\python.exe -m openreframe.server --host 127.0.0.1 --port 8000 --inpaint
```

Models load lazily by default, so the health endpoint is available while model
weights are loading. The first depth preview can take 10–60 seconds to load the
depth model; later previews are faster. The app uses a single GPU worker and
prioritizes queued previews over queued renders, but does not interrupt a render
already in progress.

The server binds to localhost by default. It has no authentication and keeps
images and job results in memory only; do not expose it to an untrusted network.
Uploads are limited to a 4096-pixel long side and the in-memory image store keeps
up to 12 images.

## Checks and profiling

Run the reprojection check suite after setup:

```powershell
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py
```

It should report **34/34 PASS**. The suite can run without a GPU. Depth profiling,
diffusion probes, and `/api/reframe` require CUDA and the corresponding model
weights. See `AGENTS.md` for the project validation rules. Benchmark records in
`docs/benchmarks.md` are machine-generated and append-only; use the relevant
probe rather than editing measured rows by hand.

## Hardware notes

- On the reference laptop, desktop applications use a variable amount of
  dedicated VRAM. The benchmark ceiling is 6.8 GiB; other GPU workloads can
  reduce available headroom.
- Windows WDDM may fall back to system memory instead of raising a normal CUDA
  out-of-memory exception. The time-based watchdog handles the observed
  slowdown behavior; consult the latest status in `docs/decisions.md`.
- Plug in a laptop before long renders to avoid power and thermal throttling.
- Model downloads can take time and require several gigabytes per checkpoint.
