# OpenReframe

OpenReframe turns a single photograph into a gently reframed, depth-aware view.
It estimates relative depth, moves the virtual camera, reprojects visible
surfaces, and fills regions revealed behind objects with an inpainting model.
The local web app lets you preview the geometry before generating a final image.

This is an experimental, local-first project—not a hosted service. The shipped
inpainting backend is Stable Diffusion XL inpainting (without ControlNet),
selected for the validated RTX 4060 Laptop 8 GB target. Model weights are large,
are downloaded separately, and are not included in this repository.

## What it does

- Upload a photograph and adjust horizontal, vertical, and forward camera shifts.
- Preview depth-aware reprojection and see disocclusion regions before inpainting.
- Generate a final fill for the newly exposed regions; generation can be cancelled.
- Keep image processing and model execution on the machine running the server.

The preview runs depth estimation and geometry only; a final render also needs
the diffusion model. Models load lazily, so the first preview or render can take
longer while weights are loaded. The app keeps uploads and job results in memory;
they are not durable and are lost when the server stops.

## Requirements

- Windows 10/11 x64 and 64-bit Python 3.11 are the validated setup.
- An NVIDIA GPU is recommended for practical depth previews and is required for
  the intended diffusion workflow. The validated target is an RTX 4060 Laptop
  GPU with 8 GB VRAM and at least 16 GB system RAM.
- Allow at least 25 GB of free disk space for model weights.
- Internet access is needed for the one-time dependency and model downloads.

CPU mode supports the geometry code, but CPU diffusion is not a supported target.
See [the Windows setup guide](docs/setup-windows.md) for the validated environment,
model downloads, GPU notes, and troubleshooting details.

## Quick start (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the depth checkpoints and the selected SDXL inpainting weights:

```powershell
& .\.venv\Scripts\python.exe scripts\download_models.py depth
& .\.venv\Scripts\python.exe scripts\download_models.py inpaint_a
```

These downloads can be several gigabytes. The downloader uses the Hugging Face
cache under the repository's ignored `models/` directory. If Hugging Face's
large-file transport stalls, see the direct-download option in
[the setup guide](docs/setup-windows.md).

Start the server:

```powershell
& .\.venv\Scripts\python.exe -m openreframe.server --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, upload an image, and drag the three camera-shift
sliders. The magenta preview indicates regions that will be inpainted. Choose
**Render final** to generate the filled result. To start model loading in the
background at server startup instead of waiting for first use, add `--inpaint`.

The app binds to localhost by default. Only expose it to a trusted network; this
prototype does not provide authentication or persistent storage.

## Verify the environment

Run the environment gate and the CPU/GPU-independent reprojection checks:

```powershell
& .\.venv\Scripts\python.exe prototypes\00_env_check.py
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py
```

The environment check should end with `ENVIRONMENT OK`; the reprojection suite
should report **34/34 PASS**. The inpainting probes and `/api/reframe` require a
working CUDA GPU and downloaded weights. See `AGENTS.md` for the repository's
developer commands and validation rules.

## Project layout

```text
openreframe/          Product pipeline, depth estimation, reprojection, server
static/               Browser UI served by the FastAPI app
scripts/              Model-weight download tools
prototypes/           Research probes and the reprojection check suite
assets/test_images/   Frozen evaluation images
docs/                 Setup, benchmark history, and decision record
models/               Local model weights (ignored by Git)
outputs/              Generated artifacts (ignored by Git)
```

`prototypes/` is intentionally retained as the research and validation record;
`prototypes/_reproj.py` re-exports the shipped implementation, and
`prototypes/04_reprojection_math.py` checks that implementation. See
[`AGENTS.md`](AGENTS.md) before making code or benchmark changes.

## Technical overview

The shipped path uses Depth Anything V2 ViT-L relative depth, edge-preserving
depth smoothing, forward-splat geometry with a backward color resample, and
SDXL inpainting. A single GPU worker prioritizes previews over queued final
renders; a running render is not preempted by a later preview. The server
provides health, upload, preview, render, progress, cancellation, and result
endpoints under `/api/`.

Current measurements and known limitations are recorded in
[`docs/benchmarks.md`](docs/benchmarks.md) and
[`docs/decisions.md`](docs/decisions.md). The benchmark values are specific to
the documented hardware and should not be treated as performance guarantees on
other systems. The current known visual weakness is residual texture in the
densest foliage of `out04`.

## Model licenses

OpenReframe code and model weights are separate works. Check the upstream model
cards and license terms for Depth Anything V2 and Stable Diffusion XL before
downloading or using their weights; see the decision record for the project's
documented license assessment.
