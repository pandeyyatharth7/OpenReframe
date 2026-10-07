# AGENTS.md — OpenReframe

Guidance for coding agents and humans working in this repository. Read this
before editing. The authoritative project record is `docs/decisions.md`;
nothing here overrides it.

## What this is

Single-image **3D photo reframing**: estimate depth, reproject the image as if
the camera had translated, then inpaint the regions that were hidden behind
objects in the original shot. Local-only, targeting an **RTX 4060 Laptop with
8 GB VRAM**. That VRAM ceiling drives nearly every design decision.

## Read these first

| File | Why |
|---|---|
| `docs/decisions.md` | Pre-registered rules, resolved decisions, **and the open blockers**. Source of truth for *why* any constant is what it is. |
| `docs/benchmarks.md` | Append-only measurement log, machine-written. |
| `docs/setup-windows.md` | The validated environment recipe. |

## Layout

```
openreframe/          <- shipped implementation
  _paths.py             HF cache redirection (must be imported first)
  config.py             frozen constants; each traces to a decision
  _warp.py              forward-splat reprojection  <- the core math
  _depth.py             Depth Anything V2 ViT-L + per-image depth cache
  _inpaint.py           candidate B diffusion, cancellable
  _watchdog.py          time-based OOM stall detection
  pipeline.py           Reframer: preview / reframe
  server.py             FastAPI, single GPU worker + priority queue
prototypes/           <- Sprint-0 audit trail, NOT the product
  00..05_*.py           numbered probe scripts
  _bench.py             VRAM/time harness; appends to docs/benchmarks.md
  _reproj.py            RE-EXPORT SHIM onto openreframe/_warp.py
  04_reprojection_math.py   THE TEST SUITE
scripts/              model downloaders
static/index.html     the web UI
assets/test_images/   frozen 15-image eval set
outputs/              artifacts (gitignored)
models/               weights (gitignored)
```

## Hard rules

1. **Never add implementation to `prototypes/_reproj.py`.** It is a shim that
   re-exports `openreframe/_warp.py`. Edit `_warp.py`. This exists so the test
   suite validates the code that actually ships.
2. **`prototypes/04_reprojection_math.py` is the test suite** (there is no
   pytest). It exits non-zero on any failure. Run it after touching warp code
   and require **34/34 PASS**.
3. **The shipped backend is candidate A** — `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`,
   no ControlNet, `CAP_PX = 1024`. Selected under D0.1 (see `docs/decisions.md`
   R5). Do not reintroduce the ControlNet conditioning path as the default:
   candidate C exceeds both the VRAM and latency gates on this hardware.
4. **`config.CANON_SHIFT`, `config.SEED`, `config.STEPS`, `config.GUIDANCE` are
   frozen.** They define the comparison protocol. Changing one invalidates every
   benchmark row recorded under it.
5. **`docs/decisions.md` is append-only for decisions.** Do not rewrite settled
   decisions. Add new entries; correct mistakes with an erratum entry (see the
   R2 erratum for the pattern).
6. **`docs/benchmarks.md` is machine-generated and append-only.** Add rows by
   running a probe through `_bench.py`, never by hand-editing.
7. **Never import `huggingface_hub` / `transformers` before `openreframe._paths`.**
   The HF cache location is read into a module constant at import time.
   Importing the `openreframe` package does this for you.
8. **Never raise `CANON_SHIFT` above ~0.06** to get a more dramatic result.
   Mask coverage scales with it — at 0.18 the masks covered 18–53% of the frame
   and inpainting quality collapsed. `MAX_SHIFT` is 0.25.

## Conventions

- Python 3.11, Windows/PowerShell, plain `venv` at `.venv` (conda is not used).
  Run scripts with `& .\.venv\Scripts\python.exe <path>`.
- Type hints on public functions, `from __future__ import annotations` at module
  top. Leading-underscore module names mark private helpers.
- Comments explain *why*, especially where behaviour is empirically derived.
  Match the density of the surrounding code rather than adding narration.
- **fp16 and `enable_model_cpu_offload()` are CUDA-only.** Both `_depth.py` and
  `_inpaint.py` branch on device; CPU is supported for depth and geometry so the
  math stays runnable without a GPU, but diffusion on CPU is not a target.
- Weights may live in `models/hub/` (HF cache) or `models/direct/`. Resolve with
  `config.local_or_repo(repo_id)`, never by hardcoding a path. `models/direct/`
  exists because hf-xet stalled; `scripts/download_direct.py` works around it.
- Reproducibility is a feature: the splat's tie-breaking is deterministic
  (`scatter_reduce(amin)` twice, lowest source index wins). Preserve that.

## Running things

```powershell
# environment gate — must print ENVIRONMENT OK
& .\.venv\Scripts\python.exe prototypes\00_env_check.py

# THE test suite — must exit 0 with 0 failures
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py

# server (geometry preview works without the diffusion pipeline)
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000 --inpaint
```

GPU is required for: `prototypes/01_depth_profile.py` timing, `02`, `03`, `05`,
and anything calling `/api/reframe`. Geometry and preview run on CPU.

## Current state — read before claiming anything works

`docs/decisions.md` is the authoritative record. **D0.1 is closed** (R5):
candidate A is the shipped backend. Depth selection (R1) and the reprojection
math (R2) are resolved. Three reprojection defects found during R4 were fixed in
R5 and every candidate re-profiled — see that entry for the numbers.

Remaining known issues:

- **`out04` residual foliage texture** — the weakest result in the set. Dense
  fine foliage against flat fog still shows a scribbled texture after inpainting.
- **WDDM CUDA System Memory Fallback is still ACTIVE**, so `except
  OutOfMemoryError` is dead code and real OOMs appear as slowdowns. Candidate C
  is the standing demonstration (it exceeded dedicated VRAM and ran 18x over
  budget instead of raising). `openreframe/_watchdog.py` catches this by
  wall-clock instead. Disable the toggle in the NVIDIA Control Panel to make the
  exception handlers live again.
- `outputs/warppairs/` and `outputs/inpaint/*` were regenerated under R5, but
  `05_e2e_minimal.py` has only ever been run for candidate A.
- SD1.5 and SDXL weights are both present; `models/direct/` is ~13 GB.

## Verification expectations

- Any change to warp math: `04_reprojection_math.py` green, plus a new check if
  behaviour changed.
- Any change to depth/pseudo-depth: add a check that covers the real-photo case
  the synthetic suite misses, or the suite will happily pass a broken change.
  T11 exists for exactly this reason — its two cases were both real bugs that
  the synthetic suite could not see.
- Any change to generation settings: re-profile the affected candidate and let
  `_bench.py` write the rows. Do not hand-write numbers.
- If GPU access is unavailable, say so and stop. Do not infer benchmark numbers
  and do not resolve a gate that was not measured.