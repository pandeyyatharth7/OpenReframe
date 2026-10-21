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
- **Smart Reframe** proposes several camera movements for the image, scores them
  geometrically, and lets you compare them side by side.
- **Multi-variant generation** renders the proposed movements one at a time and
  reports the quality of each result.
- **Quality evaluation** scores every result on original-region preservation,
  boundary consistency, expansion coverage and composition.
- **Recovery** detects problems such as excessive disocclusion and applies a
  bounded, targeted correction instead of failing silently.
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

## Smart Reframe and variants

Switch the **Mode** selector in the sidebar to *Smart*. The app then proposes
camera movements for the uploaded image and scores them geometrically:

| Control | Meaning |
|---|---|
| Variants | How many candidates to render (1–6, default 3). |
| Strategy | `preserve` favours a small, safe move; `balanced` is the default; `explore` rewards a larger one. |
| Propose camera moves | Warps and scores every candidate. Costs one geometry pass each, no diffusion. |
| Generate variants | Renders the checked candidates one after another. |

Each variant card shows the image, the X/Y/Z camera shift, the quality report,
and buttons to **Select**, **Regenerate** or **Download** it. Regenerating one
variant leaves the other results alone, and a variant that fails does not stop
the ones that follow.

Planning reuses the depth map that is already cached for the image, so it costs
no extra model inference. Variants are rendered sequentially to keep peak VRAM
within the 8 GB budget.

## Quality metrics

Every number below is measured from the data the pipeline already produces. A
metric that cannot be measured is reported as *unavailable* rather than given
a made-up value.

| Metric | What it measures |
|---|---|
| Original-Region Preservation | SSIM between the reprojected frame and the output, restricted to the region the generator was not allowed to touch (a 5 px band around the mask is excluded). |
| Boundary Consistency | Colour and gradient discontinuity between the generated region and the adjacent original pixels across the mask edge. A low value means a visible seam — it does not prove the content is realistic. |
| Expansion Coverage | Share of the missing area the generator actually replaced, measured against the pre-inpaint placeholder. Replacing a region is not the same as filling it correctly. |
| Composition Score | Geometric heuristic: original-content retention, invented area, cropping, how concentrated the invented area is, and how far the camera moved. |
| Subject Preservation | Not evaluated — the project ships no object detector. |

A result is accepted when each measured metric clears its own floor
(preservation ≥ 75, coverage ≥ 90, boundary ≥ 55). Acceptance is deliberately
not one aggregate score, so full expansion coverage cannot hide severe damage
to the original image.

## Recovery

After a render, the app looks for problems it can identify reliably and applies
at most one correction per variant by default:

| Problem | Correction |
|---|---|
| Excessive disocclusion (> 20 % of the frame) | Pull the camera move back by 40 %. |
| Unfilled regions | Widen the generation mask by one pixel and render again. |
| Poor boundary consistency | Feather the generated content into the original across the seam. |
| Damage to the preserved region | Restore the untouched region from the reprojected frame. |
| A stalled or failed generation | Run it again once. |

The correction, the problem and the metrics of every attempt are recorded and
shown under the result. If a retry produces a worse image, the better earlier
result is kept. Deterministic problems (a rejected configuration, an
undecodable image) are never retried.

## Verify the environment

Run the environment gate and the CPU/GPU-independent check suites:

```powershell
& .\.venv\Scripts\python.exe prototypes\00_env_check.py
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py
& .\.venv\Scripts\python.exe prototypes\06_v2_features.py
```

The environment check should end with `ENVIRONMENT OK`. `04_reprojection_math.py`
checks the reprojection math and `06_v2_features.py` checks the smart-refaming,
quality-evaluation and recovery logic; both print one `[PASS]`/`[FAIL]` line per
check and exit non-zero if anything failed. Neither needs model weights or a
GPU. The inpainting probes and `/api/reframe` require a working CUDA GPU and
downloaded weights. See `AGENTS.md` for the repository's developer commands and
validation rules.


## Project layout

```text
openreframe/          Product pipeline, depth estimation, reprojection, server
  _quality.py           Quality metrics for a generated image
  _smart.py             Camera-movement planner and geometric scoring
  _variants.py          Variant orchestration and bounded recovery
static/               Browser UI served by the FastAPI app
scripts/              Model-weight download tools
prototypes/           Research probes and the check suites
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
provides health, upload, preview, plan, variant, render, progress, cancellation,
and result endpoints under `/api/`.

## Known limitations

- The planner is geometric. It cannot find a subject, because the project ships
  no object detector, so "composition" means retention, cropping and invented
  area — not "is the subject well placed".
- Output aspect ratio is not configurable. The render keeps the source aspect
  ratio, scaled to fit `config.CAP_PX`.
- The metrics describe measurable properties of the pixels. They cannot say
  whether the generated content is *correct* — a confidently wrong fill scores
  well on every metric.
- Subject preservation is reported as unavailable rather than estimated.
- Recovery can only correct problems it can detect. Semantic failures (a
  duplicated object, an implausible fill) are not detected at all.
- Variants are rendered one at a time, so N variants cost roughly N renders.

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
