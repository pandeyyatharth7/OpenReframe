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

## 6. Known platform facts discovered during validation

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
- **SDXL weights are incomplete.** `text_encoder_2/model.fp16.safetensors` is a
  truncated `.part` file and `diffusers/controlnet-depth-sdxl-1.0` was never
  fetched, so candidates A and C cannot be profiled. See `docs/decisions.md`
  open blocker O1.
