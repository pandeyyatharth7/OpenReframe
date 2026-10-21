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
| `README.md` | User-facing overview and quick start. |
| `docs/decisions.md` | Pre-registered rules, resolved decisions, historical findings, and current status; source of truth for *why* constants are what they are. |
| `docs/benchmarks.md` | Append-only measurement log, machine-written. |
| `docs/setup-windows.md` | Windows environment, model downloads, and run instructions. |

## Layout

```
openreframe/          <- shipped implementation
  _paths.py             HF cache redirection (must be imported first)
  config.py             frozen constants; each traces to a decision
  _warp.py              forward-splat reprojection  <- the core math
  _depth.py             Depth Anything V2 ViT-L + per-image depth cache
  _inpaint.py           candidate A SDXL diffusion, cancellable
  _watchdog.py          time-based OOM stall detection
  _quality.py           quality metrics for a generated image (V2)
  _smart.py             camera-movement planner + geometric scoring (V2)
  _variants.py          variant orchestration + bounded recovery (V2)
  pipeline.py           Reframer: preview / plan / reframe
  server.py             FastAPI, single GPU worker + priority queue
prototypes/           <- Sprint-0 audit trail, NOT the product
  00..05_*.py           numbered probe scripts
  _bench.py             VRAM/time harness; appends to docs/benchmarks.md
  _reproj.py            RE-EXPORT SHIM onto openreframe/_warp.py
  04_reprojection_math.py   THE REPROJECTION TEST SUITE
  06_v2_features.py         THE V2 TEST SUITE (planner / metrics / recovery)
scripts/              model downloaders
static/index.html     the web UI
assets/test_images/   frozen 15-image eval set; don't rename or prune it. The
                      6-image benchmark subset lives in config.EVAL_IMAGES and
                      again in prototypes/_prepwarp.py — update both.
outputs/              artifacts (gitignored)
models/               weights (gitignored; absent in a fresh checkout)
```

## Hard rules

1. **Never add implementation to `prototypes/_reproj.py`.** It is a shim that
   re-exports `openreframe/_warp.py`. Edit `_warp.py`. This exists so the test
   suite validates the code that actually ships.
2. **The check suites are `prototypes/04_reprojection_math.py` and
   `prototypes/06_v2_features.py`** — there is no pytest, and these two scripts
   are the only harness (don't add a framework unless asked). Both use the same
   shape: one `[PASS]`/`[FAIL]` line per check, a trailing `N failure(s)`,
   non-zero exit on any failure, and **no arguments**, so every run is the full
   suite. `04` expects `33 [PASS]` / `0 failure(s)`; `06` expects
   `51 [PASS]` / `0 failure(s)`. The scripts' own output is authoritative.
   Neither needs weights or a GPU.

3. **The shipped backend is candidate A** — `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`,
   no ControlNet, `CAP_PX = 1024`. Selected under D0.1 (see `docs/decisions.md`
   R5). Do not reintroduce the ControlNet conditioning path as the default:
   candidate C exceeds both the VRAM and latency gates on this hardware.
   `_inpaint.py` still carries that branch, keyed off `config.BACKEND_CONTROLNET`
   (currently `None`). Keep it `None`; the branch stays for re-profiling only.
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
- **No linter, formatter, type checker, or CI workflow exists.** Verification is
  the reprojection suite plus `compileall`; don't assume a pipeline will catch a
  mistake, and don't introduce `ruff`/`mypy`/`make` steps as part of a change.
- `_depth.py` caches depth per image by content hash. A slider drag must cost one
  splat, not a ViT-L forward pass — don't bypass that cache in the preview path.
- PowerShell renders git/python stderr as a red `NativeCommandError` even on
  success (`git switch`, pip progress bars). Judge commands by `$LASTEXITCODE`,
  not by the colour.
- Work lands through branches and PRs; `git log origin/main` shows the merged
  pull requests. Local `main` can trail `origin/main`, so confirm a file is
  current before trusting its contents.

## Running things

```powershell
# cheap gate after any edit: syntax/import breakage, no GPU or weights needed
& .\.venv\Scripts\python.exe -m compileall -q openreframe prototypes scripts

# environment gate — must print ENVIRONMENT OK
& .\.venv\Scripts\python.exe prototypes\00_env_check.py

# THE reprojection suite — must exit 0 with 0 failures (no weights needed)
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py

# THE V2 suite — must exit 0 with 0 failures (no weights needed; mocks the model)
& .\.venv\Scripts\python.exe prototypes\06_v2_features.py

# one-image end-to-end proof, console only: depth -> warp -> inpaint
# NOTE: defaults to --pipeline B; pass A to exercise the shipped backend
& .\.venv\Scripts\python.exe prototypes\05_e2e_minimal.py --pipeline A

# server (geometry preview works without the diffusion pipeline)
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000 --inpaint
```

GPU is required for: `prototypes/01_depth_profile.py` timing, `02`, `03`, `05`,
and anything calling `/api/reframe`. Geometry and preview run on CPU.

## Current state — read before claiming anything works

`docs/decisions.md` is the authoritative record. **D0.1 is closed** (R5) — the
clarification at the end of that file supersedes the stale appended O1–O5
blocker section. Candidate A is the shipped backend; depth selection (R1) and
the reprojection math (R2) are resolved. Three reprojection defects found during
R4 were fixed in R5 and every candidate re-profiled — see that entry for the
numbers.

Remaining known issues:

- **`out04` residual foliage texture** — the weakest result in the set. Dense
  fine foliage against flat fog still shows a scribbled texture after inpainting.
- WDDM CUDA System Memory Fallback was observed on the reference machine. Check
  the current machine before relying on `OutOfMemoryError`; the watchdog also
  detects stalls by wall-clock time.
- R5 end-to-end verification used candidate A, while `05_e2e_minimal.py`
  defaults to `--pipeline B` — a bare run exercises a backend that is no longer
  shipped.
- Model weights and generated outputs are local-only, gitignored artifacts.
  Follow `docs/setup-windows.md` to download weights; do not assume they exist
  in a fresh checkout.

## Verification expectations

- Any change to warp math: `04_reprojection_math.py` green, plus a new check if
  behaviour changed.
- Any change to depth/pseudo-depth: add a check that covers the real-photo case
  the synthetic suite misses, or the suite will happily pass a broken change.
  T11 exists for exactly this reason — its two cases were both real bugs that
  the synthetic suite could not see.
- Any change to generation settings: re-profile the affected candidate and let
  `_bench.py` write the rows.
- Any change to the planner, the quality metrics or the recovery logic:
  `06_v2_features.py` green, plus a new check if behaviour changed. The V2
  suite mocks the model, so keep it that way — a check that needs weights or a
  GPU does not belong there.
- Anything in `server.py` is outside the suite's coverage: with the server
  running (CPU is enough), check `/api/health` and one `/api/preview` round trip.
  The V2 endpoints (`/api/plan`, `/api/variants`) are also outside it.
- If GPU access is unavailable, say so and stop. Do not infer benchmark numbers
  and do not resolve a gate that was not measured.
