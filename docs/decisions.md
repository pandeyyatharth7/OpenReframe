# Decisions Log — OpenReframe

This file is append-only for decisions with evidence. Pre-registered rules are
written **before** the experiments that resolve them; results reference
`docs/benchmarks.md` rows and files under `outputs/`.

---

## D0 — Sprint-0 pre-registered evaluation protocol

Written: 2026-08-22 (before any candidate profiling run).

### D0.1 Inpainting backend selection (resolves SRS TBD-1)

Candidates, ranked by fidelity prior (highest first):

| Rank | Candidate | Pipeline | Operating point |
|---|---|---|---|
| C | SDXL inpaint + SDXL depth-ControlNet | `StableDiffusionXLControlNetInpaintPipeline`, UNet `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, CN `diffusers/controlnet-depth-sdxl-1.0` | fp16, cpu-offload, vae tiling, ≤1024 long side, 25 steps |
| A | SDXL inpaint, no ControlNet | `StableDiffusionXLInpaintPipeline`, same UNet as C | fp16, cpu-offload, vae tiling, ≤1024 long side, 25 steps |
| B | SD1.5 inpaint + depth-ControlNet | `StableDiffusionControlNetInpaintPipeline`, UNet `stable-diffusion-v1-5/stable-diffusion-inpainting`, CN `lllyasviel/control_v11f1p_sd15_depth` | fp16, cpu-offload, vae tiling, ≤768 long side, 25 steps |

**Pre-committed selection rule:** pick the **highest-fidelity candidate** (C > A > B)
whose measured results satisfy ALL of:

1. Peak VRAM ≤ **6.8 GiB (6959 MiB)**, where peak = max(torch `max_memory_allocated`,
   nvidia-smi driver-level `memory.used` observed during generation) on the RTX 4060
   Laptop (8188 MiB) with the standard desktop baseline running.
2. p95 end-to-end generation latency ≤ **30 s** across ≥5 eval images.
3. No severe artifacts on the fixed eval set in `assets/test_images/`, scored by
   rubric (structure plausibility inside hole / texture coherence / boundary seam /
   global consistency, each 1–5) against Candidate B as baseline on identical
   inputs (same image, mask, seed).

Tie/edge handling: if no candidate satisfies all three, ship B and document the
gap; if several satisfy all three, take the highest fidelity rank. **This rule will
not be modified after seeing results.**

Notes fixed at registration time: candidates are compared at each model's own
sensible operating resolution (SDXL ≤1024, SD1.5 ≤768), not identical pixel counts;
resolution, steps and scheduler are recorded per row in benchmarks.md.

### D0.2 Depth model pre-commitment

- Primary candidate: **Depth Anything V2 relative-depth checkpoints**
  (`depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf`), ViT-L preferred.
- Relative depth is treated as disparity-like (`higher = closer`); pseudo-depth for
  reprojection is derived via `z ∝ 1 / (d_norm + eps)`. Slider normalization by
  median scene depth makes absolute scale unnecessary (locked decision).
- **Fallback ladder:** if ViT-L depth inference p95 > ~1.2 s (leaving headroom in
  the 2 s preview SLA), drop to ViT-B, then ViT-S — documenting quality loss per step.
- **Metric fallback trigger:** only if relative depth produces *visibly broken*
  geometry (inverted parallax, collapsed layers) across the eval set do we fall back
  to metric checkpoints + scene classifier, with written justification here.

### D0.3 Locked scope decisions (carried from plan review)

- Sliders normalized by median scene depth (not meters).
- Fill prompt: fully automatic neutral prompt; no user-facing prompt in v1.
- Generation must be cancellable between denoising steps; Sprint 0 proves the
  interrupt mechanism.
- Licenses accepted: SDXL OpenRAIL++ (CreativeML), Depth Anything V2 Apache-2.0 —
  local-only use compliant.

---

## Results (appended after experiments — see evidence pointers)

### R1 — Depth model choice (resolves D0.2) — CONFIRMED

Evidence: `docs/benchmarks.md` rows from `01_depth_profile.py` (15 images,
fp16, warm inference on target GPU).

| Variant | warm p50 | warm p95 | torch peak reserved | Pearson r vs ViT-L |
|---|---|---|---|---|
| ViT-S | 35 ms | 50 ms | ~366 MB | 0.9929 |
| ViT-B | 40 ms | 55 ms | ~710 MB | 0.9972 |
| **ViT-L** | **95 ms** | **119 ms** | **~1548 MB** | 1.000 |

- **ViT-L relative-depth checkpoint adopted.** Warm p95 = 119 ms consumes ~6%
  of the 2 s preview SLA (budget allowed ≤1.2 s); no fallback to ViT-B needed.
- Relative (disparity-like) depth CONFIRMED sufficient: contact-sheet QA
  (`outputs/depth/contact_sheet.png`) shows correct figure-ground layering;
  cross-variant r ≥ 0.993 means S/B/L agree so tightly that relative-vs-metric
  ambiguity is immaterial under median-normalized slider units. The metric+
  classifier fallback trigger was NOT hit.
- Pseudo-depth convention locked: `z = 1/(d_norm + eps)`, median-normalized;
  sliders operate in fractions of median scene depth (D0.3).

### R2 — Reprojection stack validated numerically (supports §4 of plan)

`prototypes/04_reprojection_math.py`: 22/22 checks pass vs independent
numpy ray-cast ground truth (identity exactness, analytic lateral-flow match,
co-visible coverage IoU 0.991, depth transport <0.001% median err, behind-camera
rejection, degenerate-shift rejection, portrait/landscape/square robustness,
strict-binary masks, hybrid resampler agreement).

Deviations discovered and fixed during validation (documented for Sprint 2):
1. Nearest-splat color alone produces speckle/stripes under magnification
   (real-photo flows reach ~550 px). Fix: forward-splat geometry + mask,
   backward bilinear resample for color ("forward-warp-full" hybrid).
2. Hole masks must be classified by THICKNESS (morphological closing +
   distance transform), not component area — resampling stripes percolate
   across frames and defeat area thresholds.
3. Known artifact to revisit in Sprint 5: high-frequency depth noise leaves a
   faint "netting" texture on unmasked surfaces after resampling; mitigation
   options are mask-density dilation or guided filtering.

### Erratum to R2 — check count

The R2 text above records "22/22 checks". The suite as it now stands contains
**28** checks. R2 was measured against the pre-promotion suite; T10 (4 checks,
hole classification) and the non-vacuity guard were added later. The suite is
the authority, not the prose count. Re-verified 28/28 PASS after the Phase-0
repair (see F3).

---

## Phase 0 — reprojection repair and app promotion (2026-08-23)

### F1 — `classify_holes` was non-functional (FIXED)

`prototypes/_reproj.py` referenced two undefined names (`holes`, `raw_hole`)
and could not execute. `forward_warp_full` — the entry point used by the
eval-set prep, the e2e script and check T9 — raised `NameError` on every call.
The last successful run predated the edit that introduced them, so the failure
was masked by a warm disk cache of `outputs/warppairs/`.

Three defects were repaired:

| Defect | Repair |
|---|---|
| `holes` / `raw_hole` undefined | bound to `holes_for_mask` / `raw_holes` |
| thickness applied as a global per-pixel `dist >= 3.0` | applied **per connected component** (max inscribed radius), matching the stated intent |
| redundant `max_speckle_area` area gate retained | removed — R2 deviation 2 states area thresholds are defeated by percolation |

The per-pixel thickness test was the more serious of the two logic defects: a
hole narrower than ~7 px has no pixel with distance >= 3, so **every hole up to
that width was excluded from the inpainting mask entirely** and was instead
nearest-winner smeared. The repair keeps whole qualifying components, so the
boundary shell — the seam-prone ring inpainting must own — is no longer lost.
Check T10 (`04_reprojection_math.py`) locks this behaviour in.

The implementation was promoted to `openreframe/_warp.py`; `prototypes/_reproj.py`
is now a re-export shim so `04_reprojection_math.py` validates the code that
actually ships rather than a stale copy.

### F2 — `static/` was gitignored

`.gitignore` listed `static/`, which would have silently excluded the entire
web UI from version control. Removed.

### F3 — Suite re-verified

`04_reprojection_math.py`: **28/28 PASS**, exit 0, after F1. Runs on CPU when
no GPU is present (the geometry math is device-agnostic); T3 coverage IoU
0.9908, T9 hybrid identity pixel-exact.

---

## OPEN — blockers that must be cleared before D0.1 can be closed

These are recorded as findings, **not** decisions. Nothing below has been
resolved, and no gate has been passed on the strength of this work.

### O1 — Candidate B fails D0.1 gate 3 on artifacts (SEVERE)

D0.1 gate 3 requires "no severe artifacts on the fixed eval set". Inspecting
the existing `outputs/inpaint/B_sd15_cn/` results shows gate 3 is **not met**:

- `mix01.jpg` — sky replaced by a regular grid pattern across the whole frame;
  the foreground dam smeared into radial streaks.
- `ind01.jpg` — the same grid covers the entire frame, including unmasked
  regions; terrain detail is destroyed.

Isolation run at 512 px (see `outputs/diag_ind01_warped.png`) shows the
**pre-inpainting warp output is coherent** — recognisable terrain, clean sky,
only local blockiness in the ridge. The artifact is therefore introduced by the
diffusion stage, not by the reprojection. Suspected cause: the ControlNet
conditioning map is degenerate. `dense_z` carries extreme outliers
(`z_max` up to 584 against a median of 1.0), so the 2/98-percentile stretch in
`_cond_to_u8` compresses the entire usable range into a narrow band and hands
ControlNet an almost-flat depth map.

Consequence: **the D0.1 escape clause cannot be used to ship B.** That clause
covers "no candidate satisfies all three"; here the candidate intended to ship
fails gate 3 outright. Shipping B on the current settings would ship unusable
output. D0.1 stays OPEN.

### O2 — Relative-depth normalization degenerates on far-field-heavy images

`depth_rel_to_pseudo_z` applies `z = 1/(dn + 1e-3)` then divides by the median.
With `eps = 1e-3` the transform is unbounded, and median normalization is not
robust to a bimodal depth histogram. Measured on the eval set:

| image | share of pixels with `dn < 0.001` | `z_max` | median-normalized near:far |
|---|---|---|---|
| mix05.jpg | 0.000% | 544 | — |
| out04.jpg | 0.000% | 379 | — |
| ind01.jpg | 0.000% | 181 | — |
| mix01.jpg | **49.9%** | 3.0 | **~500:1** |

`mix01.jpg` is half sky at `dn ≈ 0`, so the sky dominates the median and the
dam is compressed to a near-zero depth that collapses to the principal point
under translation — the radial streaking seen in the output. The few-pixel
outliers on the other images produce `z_max` in the hundreds, which drives
`flow_max` to 540 px on a 512 px frame.

**This is a real algorithmic defect, not tuning.** The convention
`z ∝ 1/(d_norm + eps)` (D0.3 / R1) does not pin the value of `eps` nor the
normalization statistics, so a fix is available — percentile-clipped `dn`, a
data-scaled `eps`, and percentile rather than median normalization. It has
**not** been applied, because it changes a locked decision and must be
re-validated (T11 does not exist) and re-profiled before adoption.

### O3 — The validation suite cannot detect O2

`04_reprojection_math.py` validates against a synthetic scene whose depth is a
controlled linear ramp. Median normalization is well-behaved there, so all 28
checks pass while a real-photo failure mode remains entirely uncovered. The
suite has a structural blind spot: it proves the *reprojection* is correct but
says nothing about the *pseudo-depth field fed into it*. A T11 that asserts
robustness on a bimodal / near-zero-clamped relative-depth map is required
before the O2 fix can be trusted.

### O4 — Hardware access unavailable during Phase 0

The agent session used for this work had no GPU (`nvidia-smi` denied,
`torch.cuda.device_count() == 0`). Consequently the following were **not**
performed and no rows were added to `docs/benchmarks.md`:

- re-profiling candidate B against the repaired, regenerated masks;
- the `deliberate_oom` re-probe;
- `05_e2e_minimal.py` (never run at any point in this project's history);
- the D0.1 gate-3 rubric on fresh outputs.

`outputs/warppairs/` was invalidated and has **not** been regenerated; the
pre-repair masks it held are gone.

**Update — weights now complete.** All five required weight sets are on disk
and verified to construct. `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
(unet 4897 MB fp16, text_encoder_2 1325 MB fp16, vae 160 MB fp16) and
`diffusers/controlnet-depth-sdxl-1.0` (2386 MB fp16) were fetched after this
entry was written; the previously truncated `text_encoder_2` `.part` completed.
`prototypes/02_inpaint_sdxl.py` can now profile candidates A and C, which
reopens the only legitimate route to closing D0.1 (see O1). Only the GPU run
is outstanding.

### O5 — OOM handling is now duplicated but only partly effective

`openreframe/_watchdog.py` adds time-based stall detection, because WDDM CUDA
sysmem fallback is ACTIVE (benchmarks `oom_probe`) and therefore a genuine OOM
never raises `OutOfMemoryError`. The exception handlers are retained and become
effective once "CUDA System Memory Fallback" is disabled in the NVIDIA Control
Panel — that toggle is still outstanding and was not performed here.

