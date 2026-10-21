# CLEANUP_REPORT.md

Behavior-preserving cleanup audit of OpenReframe.

| | |
|---|---|
| Branch | `chore/cleanup-audit` (cut from `bf89d28 Final Commit`) |
| Commits | `6d92e2f`, `62cdec7`, `88d0068`, `7a21122` |
| Code diff vs base | 10 files, **+8 / −82** lines (54 of the deletions are the two removed scripts) |
| External behavior | unchanged — same tests, same routes, same outputs |
| GPU benchmarks run | **none** (no benchmark rows added; `docs/benchmarks.md` untouched) |

Nothing in the application's public surface changed: no function signature, CLI flag,
HTTP route, config key, output artifact, or dependency was altered.

---

## 1. Project overview

OpenReframe turns a single photograph into a short "camera moved sideways" view. It
estimates relative depth, converts that to a unitless pseudo-depth field, forward-splats
the image as if the camera had translated, and then inpaints the regions that were hidden
behind objects in the original shot (disocclusions).

The whole design is shaped by one constraint: it must run locally on an **RTX 4060 Laptop
with 8 GB of VRAM**. `docs/decisions.md` records the pre-registered acceptance gates (D0) and
every measured candidate (R1–R5); the shipped configuration is **candidate A** — SDXL
inpainting, no ControlNet, `CAP_PX = 1024`.

**Tech stack**

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.11, `venv` at `.venv` |
| Math + tensor ops | PyTorch 2.6.0+cu124, NumPy 2.4.6, OpenCV 5.0 |
| Depth | `transformers` 5.15 + Depth Anything V2 ViT-L (relative depth, fp16 on CUDA) |
| Inpainting | `diffusers` 0.40, `stable-diffusion-xl-1.0-inpainting-0.1` (fp16, 25 steps, CFG 5.5, seed 0) |
| Serving | FastAPI 0.142 + uvicorn 0.54, one GPU worker thread, priority queue |
| UI | Single static `static/index.html` (vanilla JS/canvas, no build step) |

**Build / test / run**

There is **no build step, no package manifest, no CI workflow and no test framework** — no
`pyproject.toml`, `Makefile`, `package.json`, or `.github/`. The repository's own contract
(recorded in `AGENTS.md`) is:

```powershell
# environment gate -> must print ENVIRONMENT OK
& .\.venv\Scripts\python.exe prototypes\00_env_check.py

# THE test suite -> must exit 0 with 0 failures (no pytest)
& .\.venv\Scripts\python.exe prototypes\04_reprojection_math.py

# server: geometry preview works without the diffusion pipeline
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000
& .\.venv\Scripts\python.exe -m openreframe.server --port 8000 --inpaint
```

Those four commands are what this audit used as its baseline and final verification.

---

## 2. Directory structure

```
openreframe/                 shipped package (the product)
  __init__.py                  version + package marker; importing it applies the cache redirect
  _paths.py                    points HF_HUB_CACHE into <repo>/models; MUST import before transformers
  config.py                    frozen constants (shifts, steps, seed, caps); each traces to a decision
  _warp.py                     reprojection core: forward splat, hole classification, depth->pseudo-z
  _depth.py                    Depth Anything V2 ViT-L loader + content-hash depth cache
  _inpaint.py                  candidate-A diffusion backend, cancellable via pipe._interrupt
  _watchdog.py                 wall-clock stall detection (WDDM fallback makes OOM exceptions dormant)
  pipeline.py                  Reframer.preview (geometry only) / Reframer.reframe (full chain)
  server.py                    FastAPI app, single GPU worker + priority queue, result image routes

prototypes/                  Sprint-0 audit trail (NOT the product, but still authoritative)
  00_env_check.py              machine/environment gate
  01_depth_profile.py          depth-model latency + VRAM profiling
  02_inpaint_sdxl.py           candidate A/C profiling (SDXL inpaint, optional ControlNet)
  03_inpaint_sd15_cn.py        candidate B profiling (SD1.5 inpaint + depth ControlNet)
  04_reprojection_math.py      THE test suite: 33 numeric checks vs an independent ray-caster
  05_e2e_minimal.py            one-image end-to-end proof, console only
  _bench.py                    VRAM/wall-clock harness; appends rows to docs/benchmarks.md
  _prepwarp.py                 shared prep so every candidate inpaints identical pairs
  _reproj.py                   RE-EXPORT SHIM onto openreframe/_warp.py (no implementation allowed)

scripts/                     weight acquisition (hf-xet workarounds)
  download_models.py           hub downloader via snapshot_download, grouped depth/inpaint_a/b/c
  download_direct.py           direct-streaming downloader for stalled hf-xet hosts
  run_downloads.ps1            one-off wrapper that runs download_direct twice and logs

static/index.html            the whole browser UI (sliders, preview canvas, job polling)

docs/
  decisions.md                 pre-registered rules, resolved decisions, open blockers — source of truth
  benchmarks.md                append-only, machine-written measurement log
  setup-windows.md             validated environment recipe

assets/test_images/          frozen 15-image evaluation set (ind01-04, mix01-06, out01-05)
assets/val_images/           4 untracked photos of unknown provenance (left untouched)

models/                      weights, gitignored (~13 GB)
outputs/                     artifacts, gitignored (depth/, warppairs/, inpaint/, e2e/)
.venv/                       the environment, gitignored
.freebuff/                   stray tool metadata (project-id); see §7
```

---

## 3. Workflow

### Control flow of one preview

1. **Browser** (`static/index.html`) picks a file and POSTs it to `/api/upload`.
2. `server.upload` decodes it with PIL, rejects > `MAX_UPLOAD_PX` (4096) long side, stores it
   in `Store` (in-memory, LRU over `MAX_IMAGES = 12`) and returns a 12-char `image_id`.
3. A slider move POSTs `/api/preview` with `{image_id, shift[3]}`. The route validates that
   each component is within `MAX_SHIFT = 0.25` and enqueues a `Job` at `PRIO_PREVIEW = 0`.
   Reframes use `PRIO_REFRAME = 1`, so interactive work is never stuck behind a diffusion run.
4. One daemon thread (`Service._run`) pops the heap and executes the job, holding
   `Reframer._gpu_lock` for the duration.
5. `Reframer.preview` → `DepthEstimator.pseudo_z(img)`: **depth is cached by SHA-256 of the
   pixels** (4 images), so a slider drag costs one splat rather than one ViT-L forward pass.
   First call loads the model lazily.
6. `slider_to_center(shift, median(z))` converts the normalized sliders into a camera-center
   offset in pseudo-depth units; `forward_warp_full` then splats (z-buffer, deterministic tie
   break) , classifies holes by *thickness* into true disocclusions vs magnification speckle,
   and fills channels.
7. `validate_warp` rejects geometrically nonsensical results instead of rendering them.
8. The job returns PNGs (`warped`, `mask`, `overlay`) plus stats; the browser polls
   `/api/jobs/{id}` and fetches images from `/api/result/{id}/{kind}`.

### What `reframe` adds

Same front half, then `config.fit_resolution` downscales to `CAP_PX`, `Reframer._triplet`
resamples warped/mask/conditioning to the generation size, and `Inpainter.generate` runs
SDXL. Generation is wrapped in `StageWatchdog`, which trips on wall-clock (not on
`OutOfMemoryError`, which cannot fire while the WDDM sysmem fallback is enabled) and sets
`pipe._interrupt` so the loop stops at the next step boundary. Cancellation arrives the same
way via the job's `threading.Event`.

### Where state lives

`Store` is process memory only: restart or LRU eviction loses image ids, and `/api/jobs/{id}`
returns 404 for anything it no longer knows. Nothing is persisted by the server; `outputs/`
is written only by the prototype scripts.

---

## 4. Files removed

**Tracked files (committed as `6d92e2f`)**

| Path | Why |
|---|---|
| `prototypes/debug_pred.py` | Ad-hoc depth-sanity dump on a hardcoded image (`assets/test_images/out04.jpg`). No docstring, no references from any source, script, doc, or workflow. |
| `prototypes/debug_fill.py` | Ad-hoc warp dump on the same hardcoded image, using the **pre-R5 shift `(0.18, -0.06, 0.12)`** — a value `AGENTS.md` explicitly forbids. No references. |

Both were verified to be unreferenced with a repo-wide search across `*.py`, `*.md`,
`*.txt`, `*.ps1` and `static/`.

**Untracked local artifacts (deleted from disk only; gitignored, so no commit)**

15 files left over from weight downloads: `cand_b.log`, `cand_b.pid`, `dl_b.log`,
`dl_depth.log`, `dl_depth.pid`, `dl_direct.log`, `dl_direct.pid`, `dl_inpaint.pid`,
`dl_sdxl.err`, `dl_sdxl.log`, `pip_ml.log`, `pip_ml.pid`, `pip_torch.log`, `pip_torch.pid`,
`pip_torch_err.log`.

They match `.gitignore` (`*.log`, `*.err`, `*.pid*`), no code or script reads them, and
`docs/benchmarks.md` — not these logs — is the canonical measurement record.

---

## 5. Code changes

### `openreframe/server.py` *(shipped code)*
- Removed `from pathlib import Path` and `import numpy as np` — both unused (confirmed by
  attribute scan; `Response`/`Image`/`io` cover every actual use).
- Removed `Store.depth()`. It counted cached images and had **no callers**; `/api/health`
  reports load from `len(service._heap)` (queued jobs) and `Reframer.health()` (model state)
  instead.

### `openreframe/_warp.py` *(shipped code)*
- `dev = "cuda" if torch.cuda.is_available() else "cpu"` → `dev = device()`.
  `config.device()` is that exact expression, and the module already imports `FOV_H_DEG`
  from `.config`, so this removes a duplicated device decision without adding a dependency
  or an import edge that did not already exist.

### `openreframe/_depth.py` *(shipped code)*
- One continuation line of the `DepthEstimator.warm` docstring sat at 4-space indent inside
  an 8-space docstring. Text-only fix; no code touched.

### `prototypes/03_inpaint_sd15_cn.py`
- Removed unused `import time` (no `time.` reference in the module).

### `prototypes/04_reprojection_math.py`
- Removed a duplicated `sys.path.insert(0, str(Path(__file__).resolve().parent))` that
  appeared twice with identical arguments immediately before the `_reproj` import. **The test
  suite itself is untouched** — no assertion, threshold, or case was modified.

### `prototypes/05_e2e_minimal.py`
- Removed a redundant `import _bench  # noqa: F401`: the very next line,
  `from _bench import OUTPUTS_DIR, Bench`, already executes the module, so the env-var side
  effect it guarded is preserved.
- Replaced the `to_pil = lambda t8: Image.fromarray(t8)` wrapper with direct
  `Image.fromarray(...)` calls at its two call sites.

### `prototypes/_bench.py`
- Collapsed a dead branch in `Bench.finish`:
  ```python
  header_needed = BENCHMARKS_MD.exists()
  if header_needed:
      pass
  else:
      <write header>
  ```
  became `if not BENCHMARKS_MD.exists(): <write header>`. Equivalent behavior, verified by
  writing to a temporary path (header emitted once when absent, row-only append when
  present). `docs/benchmarks.md` was **not** touched during verification.

### `scripts/download_models.py`
- `from _bench import MODELS_DIR` bound a name the script never used; the import exists only
  for its side effect (`HF_HUB_CACHE`/`HF_HOME` redirection). Now `import _bench  # noqa:
  E402,F401` with a comment saying so, so a future reader does not "clean up" the import and
  silently relocate the weight cache.

### Error handling reviewed and deliberately **kept**

| Location | Why it stays |
|---|---|
| `server.upload` decode `try/except` → `HTTPException(400)` | User-provided bytes: a real boundary, and the message is what the UI shows. |
| `_inpaint.generate` `except torch.cuda.OutOfMemoryError` → `empty_cache()` + re-raise | Genuine failure mode with required cleanup. Dormant only while the WDDM sysmem fallback is enabled; `_watchdog.py` documents that. |
| `_watchdog._run` `try/except Exception: pass` around `on_exceed` | Caller-supplied callback arriving on a daemon thread; letting it raise would kill the watchdog without tripping, which is strictly worse. |
| `_bench._nvidia_smi` broad `except` → `None` | External process boundary; absence of `nvidia-smi` is normal on CPU hosts. |
| `00_env_check` handlers | The script's entire purpose is to report FAIL instead of crashing. |
| `_warp.make_disocclusion_mask` assert | Asserts the binary-mask contract that checks T7/T10 rely on. Cheap, and it documents intent. |

### Comments
Every Python module already opens with a role-and-provenance header, and the non-obvious
logic (thickness-based hole classification, the near-half depth anchor, the two-pass
deterministic tie-break, the watchdog rationale) is already explained at the point of use.
`AGENTS.md` asks for that density to be *matched*, not exceeded, so no new narration was
added; the only comment-adjacent fix is the `_depth.py` docstring indentation above.

---

## 6. Verification

Baseline was measured on the pre-cleanup tree by checking out `bf89d28` into a detached
worktree, then re-running the identical commands on the cleaned branch. Machine: RTX 4060
Laptop, driver 616.92, torch 2.6.0+cu124 (CUDA 12.4), 31.3 GB RAM.

| Check | Command | Baseline (`bf89d28`) | Final (`chore/cleanup-audit`) |
|---|---|---|---|
| Syntax/compile | `python -m compileall -q openreframe prototypes scripts` | exit 0 | exit 0 |
| Import surface | `import openreframe; from openreframe import server, pipeline, _warp, _depth, _inpaint, _watchdog` | OK, `len(app.routes) == 13` | OK, `len(app.routes) == 13` |
| Environment gate | `python prototypes/00_env_check.py` | 15 pass, 0 warn, 0 fail | 15 pass, 0 warn, 0 fail |
| **Test suite** | `python prototypes/04_reprojection_math.py` | **33 `[PASS]`, 0 `[FAIL]`, exit 0** | **33 `[PASS]`, 0 `[FAIL]`, exit 0** |
| Working tree | `git status --porcelain` | `?? assets/val_images/` | `?? assets/val_images/` |

Representative suite values were identical on both trees (e.g. T3 IoU, T9 shifted-view
`frac=0.9828 meanAD=0.00171`, T11 `max/median=11.1`), which is the expected outcome: the
warp math and its thresholds were not edited.

Note on the count: `AGENTS.md` states the suite must be "34/34". The script prints **33**
`[PASS]` lines and never prints a total — it only prints `N failure(s)`. The pristine tree
also reports 33, so this is a pre-existing documentation discrepancy, not a regression, and
`AGENTS.md` was left unmodified.

Checks deliberately **not** run: the GPU profiling scripts (`01`/`02`/`03`/`05`). They append
rows to the machine-written `docs/benchmarks.md`, and reproducing them is not part of a
behavior-preserving cleanup. `git status docs/benchmarks.md` is clean.

---

## 7. Needs your decision

Left untouched on purpose — each is either ambiguous, referenced somewhere, or owned by you.

1. **`.freebuff/project-id`** (37 B, untracked): metadata from some external tool. Removing
   it is probably harmless (it looks regenerable), but it is not this project's file and I
   could not prove nothing depends on it.
2. **`scripts/run_downloads.ps1`**: tracked, and its group names (`inpaint_b`, `inpaint_ac`)
   do exist in `download_direct.py`, so it is a working helper — but it is referenced by no
   code, doc, or workflow, hardcodes one machine's invocation, and writes `dl_direct.log`
   (the kind of stray log this cleanup removed). Kept as a runbook script; delete it if you
   never invoke it by name.
3. **`_warp.backward_resample`**: has no callers anywhere in the repo, yet it is exported
   through `prototypes/_reproj.py`'s `__all__`. Deleting it would be a genuine dead-code
   removal but also a change to the shim's surface, which `AGENTS.md` rule 1 puts off-limits
   to casual edits. Your call.
4. **`AGENTS.md` "34/34"**: the suite prints 33 passing checks. Either the count drifted
   when a check was folded into another, or it counted the summary line. Worth correcting in
   the docs (I did not edit `AGENTS.md`).
5. **`prototypes/_prepwarp.py` duplicates `config.fit_resolution`**: intentional — the
   probes are meant to stay self-contained and able to run without the package. Flagging it
   so it is not mistaken for copy-paste damage later.
6. **No `README.md`**: the only run instructions live in `AGENTS.md` and
   `docs/setup-windows.md`. A README for human visitors is a content decision, so out of
   scope here — but it is the single biggest onboarding gap.
7. **`assets/val_images/`** (4 photos, untracked): unknown provenance, unchanged. If they
   are meant to be part of the evaluation set, they should be committed and named to match
   `EVAL_IMAGES`; otherwise they are safe to delete.
8. **`models/` (~13 GB) and `outputs/`**: gitignored working data. Neither was modified,
   regenerated, or deleted.
