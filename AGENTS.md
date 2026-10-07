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
   and require 28/28 PASS.
3. **The test suite has a known blind spot** — it uses a synthetic linear depth
   ramp, so median normalization is always well-behaved. It cannot detect real-photo
   depth failures. See open blocker O3 in `docs/decisions.md`.
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

`docs/decisions.md` ends with five open blockers. Two are severe and neither is
resolved:

- **O1 — candidate B fails the D0.1 artifact gate.** Existing outputs in
  `outputs/inpaint/B_sd15_cn/` show a grid artifact across the whole frame.
  The warp output is coherent, so the defect is in the diffusion stage, most
  likely a degenerate ControlNet conditioning map. **D0.1 is NOT closed** — the
  "ship B and document the gap" escape clause does not apply when the candidate
  intended to ship fails the quality gate itself.
- **O2 — relative-depth normalization is defective on far-field-heavy images.**
  `z = 1/(dn + 1e-3)` is unbounded and median-normalization collapses when ~50%
  of an image sits at `dn ≈ 0` (`mix01.jpg`: 49.9%). Do not treat the pipeline
  as shippable until this is fixed, re-validated and re-profiled.

Also outstanding: `outputs/warppairs/` was invalidated and not regenerated; no
benchmark rows were added in this pass; `05_e2e_minimal.py` has never been run;
and the NVIDIA "CUDA System Memory Fallback" toggle is still off.

## Verification expectations

- Any change to warp math: `04_reprojection_math.py` green, plus a new check if
  behaviour changed.
- Any change to depth/pseudo-depth: add a check that covers the real-photo case
  the synthetic suite misses, or the suite will happily pass a broken change.
- Any change to generation settings: re-profile the affected candidate and let
  `_bench.py` write the rows. Do not hand-write numbers.
- If GPU access is unavailable, say so and stop. Do not infer benchmark numbers
  and do not resolve a gate that was not measured.