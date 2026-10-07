# OpenReframe — Windows Setup Guide (target: RTX 4060 Laptop, 8GB)

Validated configuration — this exact stack is confirmed working on the Sprint-0
machine. Follow top to bottom for an identical environment.

## 1. Prerequisites

| Component | Requirement | Validated on this machine |
|---|---|---|
| OS | Windows 10/11 x64 | Windows 11 (build 26xxx) |
| GPU | NVIDIA RTX 4060 Laptop 8GB (Ada, sm_89) | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver | any recent Game Ready/Studio driver | **610.62** (CUDA UMD 13.3), WDDM mode |
| Python | 3.10 – 3.12 (64-bit) | **3.11.9** (`py -3.11`) |
| System RAM | ≥ 16 GB hard requirement (cpu-offload holds SDXL in RAM) | 32 GB soldered |
| Disk | ≥ 25 GB free for model weights | ~350 GB free |

Notes:
- Conda is NOT required; plain `venv` + pip works and is what we test.
- No CUDA Toolkit install needed — PyTorch wheels bundle the CUDA runtime.
- Hybrid-graphics note (Optimus laptops): the Windows desktop composites on the
  dGPU here (~0.4–1.6 GB VRAM baseline varies with open apps). All VRAM rules
  therefore use fixed ceilings measured via nvidia-smi, never "free memory".

## 2. Environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install transformers diffusers accelerate safetensors opencv-python huggingface_hub scipy pillow numpy
.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" python-multipart
```

Pinned versions validated together (also in `requirements.txt`):

```
torch==2.6.0+cu124  torchvision==0.21.0+cu124
transformers==5.15.1  diffusers==0.40.0  accelerate==1.14.0
huggingface_hub==1.28.0  safetensors==0.8.0  opencv-python==5.0.0.93
numpy==2.4.6  scipy==1.17.1  Pillow==12.3.0
fastapi==0.142.2  uvicorn[standard]==0.54.0  python-multipart==0.0.32
```

## 3. Validate

```powershell
.venv\Scripts\python.exe prototypes\00_env_check.py
```

Must end with `ENVIRONMENT OK`. Expected key lines:
- `torch 2.6.0+cu124`, `CUDA available at runtime`
- `GPU detected — NVIDIA GeForce RTX 4060 Laptop GPU`, capability `(8, 9)`
- fp16 + bf16 matmul PASS

## 4. Model weights (one-time, needs internet)

```powershell
.venv\Scripts\python.exe scripts\download_models.py depth      # DA-V2 S/B/L ~1.7 GB
.venv\Scripts\python.exe scripts\download_models.py inpaint_a   # SDXL-inpaint
.venv\Scripts\python.exe scripts\download_models.py inpaint_b   # SD1.5-inpaint + CN-depth
.venv\Scripts\python.exe scripts\download_models.py inpaint_c   # + SDXL CN-depth
```

Weights land under `models/hub/` (gitignored). After this step the machine is
fully offline-capable.

## 5. Running the app

```powershell
# geometry preview only — diffusion is not loaded, so this needs no diffusion VRAM
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000

# preload the diffusion pipeline on a background thread at startup
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000 --inpaint
```

Then open <http://127.0.0.1:8000>. Drop an image, drag the three shift sliders
for a geometry-only preview (magenta marks the regions that would be
inpainted), then press **Render final** for the diffusion pass.

The first preview pays a one-off depth-model load (10-60 s). After that a warm
preview is ~100 ms against the 2 s SLA. Models load lazily by design so the
server never blocks on startup — `/api/health` is reachable immediately and
reports `depth_loaded` / `inpaint_loaded`.

Scheduling: one GPU, so one worker thread and a priority queue. Previews
outrank finals, but a preview submitted *during* a running diffusion waits for
it — the queue orders new work, it does not preempt a generation the user is
waiting on.

## 6. GPU runbook

The machine is fully provisioned for GPU work; nothing needs downloading.
Because laptop GPUs are thermally/power limited, plug in the charger before
starting.

```powershell
# 0. gate the environment — must print ENVIRONMENT OK
& .\.venv\Scripts\python.exe prototypes\00_env_check.py

# 1. the test suite — must exit 0 with 28/28 PASS
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py

# 2. regenerate outputs/warppairs/ (invalidated in Phase 0) with repaired masks,
#    then re-profile candidate B — writes rows to docs/benchmarks.md
& .\.venv\Scripts\python.exe prototypes\03_inpaint_sd15_cn.py

# 3. candidates A and C — now that the SDXL weights are present
& .\.venv\Scripts\python.exe prototypes\02_inpaint_sdxl.py --variant inpaint
& .\.venv\Scripts\python.exe prototypes\02_inpaint_sdxl.py --variant controlnet

# 4. first real end-to-end image (has never been run in this project's history)
& .\.venv\Scripts\python.exe prototypes\05_e2e_minimal.py --pipeline C
```

Step 2 must precede step 3: `get_pairs()` regenerates the shared
(warped, mask, cond) cache on first call, so whichever candidate runs first
fixes the inputs for all of them. Run them in that order, or the benchmark rows
will not be comparable.

Steps 2-4 all append to `docs/benchmarks.md`. Do not hand-edit that file.

See open blockers O1-O4 in `docs/decisions.md` for what these runs are meant to
close.

## 7. Known platform facts discovered during validation

- Desktop baseline VRAM usage fluctuates (~420 MiB → ~1.6 GiB observed) with
  open apps; treat 6.8 GiB as the hard ceiling for any single pipeline stage.
- WDDM "CUDA sysmem fallback" status is probed by `_bench.deliberate_oom()`;
  if it reports ACTIVE, real OOMs surface as extreme slowdowns instead of
  exceptions. **This machine reported ACTIVE.** Recommended fix: disable
  "CUDA System Memory Fallback" in NVIDIA Control Panel (Manage 3D settings).
  Until that is done, the `except OutOfMemoryError` handlers cannot fire —
  `openreframe/_watchdog.py` detects stalls by wall-clock instead and interrupts
  the run. The exception handlers stay in place and take over once the toggle
  is flipped. **Outstanding: the toggle has not been flipped.**
- Driver 610.62 ships CUDA UMD 13.3 → any cu12x wheel works; cu124 chosen for
  stability, not necessity.
- `models/direct/` exists because hf-xet stalled during downloads.
  `scripts/download_direct.py` bypasses xet with chunked Range-resume GETs.
  Weights may live in either store; resolve with `config.local_or_repo()`.
- All five weight sets required by every candidate are present and verified
  loadable: Depth Anything V2 S/B/L (in `models/hub/`), SD1.5-inpaint +
  SD1.5 depth-ControlNet (candidate B), SDXL-inpaint (A), SD1.5-inpaint +
  SDXL depth-ControlNet (C). Use `--fp16-only` when downloading — every load
  path passes `variant="fp16"`, so the fp32 copies are dead weight.
