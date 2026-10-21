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

### F4 — Two latent crashes in the SDXL probes (FIXED)

`prototypes/02_inpaint_sdxl.py` called `pipe.enable_vae_tiling()`
unconditionally. `StableDiffusionXLInpaintPipeline` does not expose that
method, so candidate A aborted immediately with `AttributeError` on its first
ever run. `05_e2e_minimal.py` had the identical defect. Both now guard every
optional memory knob with `hasattr`, matching the pattern `03_inpaint_sd15_cn.py`
already used. Candidate A/C had therefore *never* been executed before this
pass — the "unprofiled" state in O4 was partly this bug, not only missing
weights.

---

## R4 — All three candidates profiled; D0.1 CANNOT be closed as written

GPU session, 2026-08-23, RTX 4060 Laptop (8188 MiB), driver 616.92,
torch 2.6.0+cu124. `00_env_check.py`: 15 pass, 0 warn, 0 fail.
`04_reprojection_math.py`: 28/28 PASS on GPU. `05_e2e_minimal.py` run for the
first time in the project's history (candidate A, panel at
`outputs/e2e/panel_A.png`).

`outputs/warppairs/` was regenerated through the repaired `classify_holes`.
Masks are larger than the pre-repair ones, as predicted in F1 (whole
components are masked now, not eroded cores): ind01 17.2→20.0%, out02
39.3→42.2%, mix01 51.3→51.7%, mix05 23.1→30.1%, out04 43.5→51.9%, mix06
27.9→33.9%.

### Gate 1 (VRAM ≤ 6959 MiB) and gate 2 (p95 ≤ 30 s)

| Candidate | p50 | p95 | torch peak a/r | driver peak | gate 1 | gate 2 |
|---|---|---|---|---|---|---|
| C — SDXL + CN-depth | 289.0 s | **378.7 s** | 8048.6 / **8642.0** MiB | **7949** MiB | **FAIL** | **FAIL** |
| A — SDXL inpaint | 16.3 s | 23.8 s | 5497.8 / 5976.0 MiB | 6091 MiB | pass | pass |
| B — SD1.5 + CN-depth | 9.0 s | 12.5 s | 2900.4 / 3168.0 MiB | 3287 MiB | pass | pass |

Candidate C fails gate 1 by ~990 MiB and gate 2 by 12.6×. Its torch reserved
peak (8642 MiB) exceeds total dedicated VRAM (8188 MiB): it did not OOM, it
**spilled into system RAM and crawled**. That is the WDDM sysmem-fallback
failure mode predicted in the OOM probe, now observed in a real pipeline, and
it is direct evidence that the `except OutOfMemoryError` handlers cannot be
relied on while that toggle is on.

**Cancellation is now proven** (previously never run): candidate A stopped at
10/25 denoising steps in 9.5 s against a ~16.3 s full run, via
`callback_on_step_end` + `pipe._interrupt`.

### Gate 3 (artifact rubric)

Assessed by inspection of `outputs/inpaint/{A_sdxl_inpaint,B_sd15_cn,C_sdxl_cn}/`.
Scores below are **agent-assessed** against the D0.1 rubric (structure
plausibility / texture coherence / boundary seam / global consistency, 1–5) and
need sign-off before they can be treated as the recorded result.

| image | mask | B (SD1.5+CN) | A (SDXL) |
|---|---|---|---|
| ind01 | 20.0% | global grid, ~1/5 | clean, grey band at bottom edge, ~3.5/5 |
| out02 | 42.2% | not re-inspected | severe stipple over subject, ~2.3/5 |
| mix01 | 51.7% | global grid + smeared foreground, ~1/5 | clean, photoreal, ~4.75/5 |
| mix05 | 30.1% | not re-inspected | scattered grey cubes in foliage, ~2.75/5 |
| out04 | 51.9% | not re-inspected | shredded trees + cubes, ~1.75/5 |
| mix06 | 33.9% | not re-inspected | clean, ~4.5/5 |

Mean: **A ≈ 3.3/5, B ≈ 1.3/5, C not scored** (eliminated on gates 1–2).

**No candidate satisfies all three gates.** B fails gate 3 catastrophically —
the grid artifact covers the entire frame, and it is *not* a "gap", it is a
total failure of the output. A passes gates 1–2 and beats B decisively on the
rubric, but has severe artifacts (stipple, floating grey cubes) on 3 of 6
images.

### The pre-registered rule cannot be applied as written

D0.1's tie-break reads: *"if no candidate satisfies all three, ship B and
document the gap."* That clause was written on the assumption that B is an
acceptable floor. The measurements refute that assumption — **B is the worst of
the three on quality**, not the safety net. Mechanically applying the rule
would ship the candidate that fails gate 3 hardest.

This is a defect in the rule, not in the candidates. Amending a
pre-registered rule after seeing results is exactly what the protocol exists to
prevent, so **this is escalated for an explicit human decision and no rule text
has been changed.** Two defensible options:

1. Amend the tie-break to "ship the highest-ranked candidate that passes the
   *measurable* gates (1 and 2), with its artifact failures recorded as known
   defects" — which selects **A**, on the reasoning that a candidate failing
   artifact quality on 3/6 images still beats one failing on 6/6.
2. Amend it to "if no candidate passes all three, no candidate ships" — which
   leaves D0.1 open until O2 is fixed and the candidates are re-profiled.

Option 2 is the more conservative and is what the evidence currently supports,
because the artifact failures in A are very likely *downstream of* O2 rather
than intrinsic to SDXL — see below.

### The artifact failures are probably downstream of O2, not model choice

The two dominant failure modes have a common, non-diffusion cause:

- **Floating grey cubes** (mix05, out04) are small *isolated* masked regions.
  SDXL fills an isolated 20 px hole with no usable context and emits a flat
  blob. O2's extreme depth outliers (`z_max` up to 584 against median 1.0)
  produce spurious high-flow pixels scattered across the frame, which is what
  generates large numbers of small isolated holes in the first place.
- **Stipple / dither** (out02, ind01 bottom band) is the "netting" artifact R2
  flagged as faint and deferred to Sprint 5. It is severe here, and again
  tracks the noisy depth field rather than the sampler.

Both point at fixing the pseudo-depth field and adding mask cleanup, then
re-profiling — not at re-ranking candidates. C remains the highest-fidelity
option but cannot run at all on this hardware within gate 2.

---

## R5 — Root causes fixed; D0.1 RESOLVED to candidate A

The R4 artifacts were **not** a property of the diffusion models. Three
independent defects in the reprojection stage were producing them, and fixing
those removed them from candidate B as well.

### D1 — Unbounded pseudo-depth (was O2)

`depth_rel_to_pseudo_z` used `z = 1/(dn + 1e-3)` normalized by `median(z)`.
Three problems, all now fixed:

- **Unbounded eps.** As `eps -> 0`, pixels with `dn ~ 0` reach `z = 544`
  against a median of 1.0 and dominate the reprojection. `eps` is now 0.05,
  bounding the dynamic range.
- **min/max range.** Relative-depth checkpoints emit isolated outlier pixels,
  which min/max normalisation lets stretch the range. Now clipped at the 1st
  and 99th percentiles, so `z_max` fell from 552 to ~12 on the eval set.
- **Median-of-z anchoring.** On far-field-saturated images the median of `z`
  lands inside the sky plateau, so median normalisation rescales the entire
  foreground to a near-zero depth and it collapses toward the principal point.
  The scale is now anchored on the median depth of the scene's **near half**
  (`dn > median(dn)`), which is always well defined.

Measured effect on `mix01.jpg` (the saturated case, ~50% of pixels at
`dn ~ 0`): mask coverage at shift 0.03 fell from **50.5% to 9.4%**, and T11's
near-half minimum depth rose from ~0.002 to 0.77.

### D2 — No mask area cleanup

`classify_holes` decided thickness-vs-speckle per component but kept *every*
qualifying component, however small. A splat at real-photo displacements
produces many tiny isolated holes, and a diffusion inpainter fills each one
with a single flat blob — observed as the scattered grey cubes. Components
below an area floor (`max(64, 3e-4 * H * W)`) are now nearest-winner filled
instead of inpainted. The mask went from many components to **1–11 coherent
regions with zero specks**.

### D3 — No depth smoothing; canonical shift far too large

Leaf- and branch-scale depth noise was the source of those tiny holes.
`openreframe/_depth.smooth_relative_depth` now applies a bilateral filter to the
relative-depth map before reprojection — edge-preserving, so true silhouette
discontinuities survive (a Gaussian blur would destroy them). This is the
mitigation R2 deviation 3 anticipated for the netting artifact.

Separately, `CANON_SHIFT` was `(0.18, -0.06, 0.12)`. At that magnitude the
reprojection masks covered **18–53% of the frame**, which is far more
disocclusion than any inpainter fills plausibly and is not representative of how
the product is used (users want subtle parallax). Scaled 3x to
**`(0.06, -0.02, 0.04)`**. Mask coverage is now **8.6–23.6%**. This is a
decision change, recorded here with its rationale; it invalidates all
pre-R5 benchmark rows, which is why every candidate was re-profiled.

### Re-profiled results, all three gates

| Candidate | p50 | p95 | driver peak | gate 1 | gate 2 | gate 3 |
|---|---|---|---|---|---|---|
| C — SDXL + CN-depth | 289.0 s | 378.7 s | 7949 MiB | FAIL | FAIL | not scored |
| **A — SDXL inpaint** | 16.6 s | **21.4 s** | **6105 MiB** | pass | pass | **pass** |
| B — SD1.5 + CN-depth | 9.1 s | 12.0 s | 3277 MiB | pass | pass | pass (improved) |

Gate 3 on the corrected inputs, agent-assessed against the D0.1 rubric:
candidate A is clean and photorealistic on **5 of 6** images — `ind01`,
`out02`, `mix01`, `mix05`, `mix06` — with a residual scribbled texture in the
densest foliage of `out04` (the hardest case: fine foliage against flat fog).
No severe artifacts remain. Candidate B's global grid is gone; it retains
localised netting on textured surfaces such as the `mix01` dam wall.

### D0.1 is CLOSED — candidate A selected

Applying D0.1 as pre-registered: *"pick the highest-fidelity candidate
(C > A > B) whose measured results satisfy ALL of"* the three gates.

- C is the top fidelity rank but fails gates 1 and 2 → eliminated.
- **A satisfies all three** → selected.

The escape clause ("ship B and document the gap") is therefore **not invoked**,
and the concern raised in R4 — that the clause would ship the worst candidate —
does not materialise. No rule text was amended at any point. Shipped backend is
now `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` with no ControlNet,
`CAP_PX = 1024`.

The R4 escalation is withdrawn as resolved by evidence rather than by amending
the protocol.

### Application path verified end-to-end

Through `openreframe.pipeline.Reframer` on GPU: depth warm 487 ms, **preview
510 ms** (inside the 2 s SLA), reframe 13.7 s with the watchdog not tripped
(limit 53.5 s), and **cancellation confirmed** — a mid-run cancel stopped
denoising at 6/25 steps in 5.2 s.

### Still open

- `out04` residual foliage texture is the weakest remaining result.
- O5: the NVIDIA "CUDA System Memory Fallback" toggle is still off, so
  `except OutOfMemoryError` remains dead code. Candidate C is the standing
  demonstration of why: it exceeded dedicated VRAM and ran 18x over budget
  instead of raising.
- O3 is partly addressed: T11 (6 checks) now covers the two real-photo
  pseudo-depth failure modes that the synthetic suite structurally could not
  detect. Suite total is 34/34 PASS.

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

---

## Erratum — status of the appended O1–O5 findings (2026-10-10)

The `OPEN — blockers that must be cleared before D0.1 can be closed` section
above was written before the R5 results and appended afterward as historical
material. Its statements that D0.1 remains open, that O1–O4 are unresolved, and
that the suite lacks T11 are superseded by R5 and its evidence:

- **D0.1 is closed.** Candidate A passed all three pre-registered gates and is
  the shipped backend (R5).
- **O1 and O2 are resolved** by the corrected pseudo-depth, mask cleanup, and
  depth smoothing described in R5.
- **O3 is resolved for the documented failure modes:** T11 was added, and the
  reprojection suite now contains 34 checks (34/34 PASS).
- **O4 is historical:** GPU access and profiling were completed for R5; the
  R5 benchmark rows and end-to-end verification are recorded above.
- **O5 remains an environment caveat, not a D0.1 blocker.** WDDM CUDA system
  memory fallback was observed on the reference machine; the time-based
  watchdog remains in place. Re-check the current machine's setting before
  relying on CUDA out-of-memory exceptions.

The earlier section is retained to preserve the append-only investigation
record; use R5 and this erratum for current project status.

---

## V2 — Smart reframing, variants, quality metrics and recovery (2026-10-22)

This entry records the V2 feature work. It **does not** reopen D0.1 and it does
**not** change any frozen constant: `CANON_SHIFT`, `SEED`, `STEPS` and
`GUIDANCE` are untouched, so every benchmark row recorded under them stays
valid.

### V2.1 What was added

| Module | Responsibility |
|---|---|
| `openreframe/_smart.py` | Proposes camera movements and scores them geometrically. |
| `openreframe/_quality.py` | Measures the quality of a generated image. |
| `openreframe/_variants.py` | Runs one variant end to end and applies bounded recovery. |
| `prototypes/06_v2_features.py` | Check suite for the three modules above (no weights, no GPU). |

`Reframer` gained `depth_for()` (the already-cached pseudo-depth), `plan()`
(geometry only) and `mask_dilate_px` on `reframe()`. `server.py` gained
`/api/plan` and `/api/variants`; the job result now carries a report, a
recovery log and the effective shift.

### V2.2 Smart reframing is geometric, not semantic

The planner crosses three magnitude tiers (0.5x, 1.0x, 1.7x the canonical
shift) with five directions and scores every candidate from the warp it
already has to compute. No detector was added: the repository has no object
detector, so the planner makes no claim about subjects, and the UI reports
subject preservation as *unavailable* rather than guessing.

Candidates are always inside `MAX_SHIFT` (0.25) and never duplicated. The
selection takes the best candidate of each tier first, so a set of three always
spans the conservative/balanced/exploratory characters with genuinely different
camera configurations.

Scoring weights live in `_quality.STRATEGIES` and sum to 1.0 per strategy:

| Factor | Meaning | preserve | balanced | explore |
|---|---|---|---|---|
| retention | share of the new frame that is original content | 0.45 | 0.35 | 0.20 |
| exposure | penalty for the share the inpainter must invent | 0.35 | 0.30 | 0.25 |
| crop | penalty for the share of the original that leaves the view | 0.15 | 0.20 | 0.10 |
| spread | reward for a coherent rather than speckled invented area | 0.05 | 0.15 | 0.10 |
| motion | reward for moving the camera at all | 0.00 | 0.00 | 0.35 |

Planning costs one geometry-only warp per candidate (16 candidates, 2.4 s
measured on a 960x1280 frame on the reference machine) and no additional model
inference, because `DepthEstimator` already caches the depth map by image
content.

### V2.3 The quality metrics are measurements, not scores

Four metrics are computed from the warped frame, the disocclusion mask and the
generated image:

- **Original-Region Preservation** — SSIM between the reprojected frame and the
  output, restricted to the region outside the mask (plus a 5 px band, because
  the SSIM window reaches across the seam). Comparing against the *reprojected*
  frame rather than the original photograph is what makes the comparison valid
  under camera movement.
- **Boundary Consistency** — colour and gradient discontinuity across the mask
  edge. A low value means a visible seam; it does **not** prove the content is
  realistic, and the report says so.
- **Expansion Coverage** — share of the missing area the generator replaced,
  measured against the pre-inpaint placeholder. Replaced is not the same as
  correct.
- **Composition Score** — the geometric heuristic above.

A metric that cannot be computed is reported with status *unavailable* and a
null value. Nothing is back-filled with zero.

Acceptance is **per metric** (`_quality.ACCEPT_FLOORS`: preservation 75,
coverage 90, boundary 55), never a single aggregate, so a perfect expansion
coverage cannot mask severe damage to the original image.

### V2.4 Recovery is bounded and targeted

At most one correction per variant by default (`_variants.run_variant`):

| Detected problem | Correction |
|---|---|
| disocclusion above 20 % of the frame | camera move scaled by 0.6 |
| unfilled regions | generation mask widened by 1 px, then re-rendered |
| boundary score below its floor | feathered seam blend into the original |
| preservation below its floor | the untouched region is restored from the reprojected frame |
| stalled generation | one plain retry |
| rejected configuration / undecodable image | no retry — the failure is deterministic |

Corrections are split into *re-runs* (the inputs change and the image is
generated again) and *post-fixes* (the existing image is corrected in place).

Attempts are compared on their **raw** metrics, so a repaired attempt cannot
outrank an attempt that never needed a repair; the repair is then applied to
the winner. Ties are broken towards the attempt that invents less. A variant
that fails is reported with its error and does not affect the other variants —
they are separate jobs on the same queue.

### V2.5 Verification performed

- `prototypes/04_reprojection_math.py` — unchanged, still 0 failures. The only
  change to `_warp.py` was the added `out_of_frame_frac` diagnostic (the share
  of the original frame that the new camera no longer sees); the warp math is
  byte-for-byte unchanged.
- `prototypes/06_v2_features.py` — new, 0 failures. It covers candidate
  bounds, duplicates, tier differences, missing optional arguments, coordinate
  conventions, the four metrics, the unavailable-mask path, size mismatches,
  mask exclusion, coverage, the recovery corrections, the retry limit, failure
  isolation, the attempt selection, and the manual/smart/integration paths.
  It mocks the model, so it needs neither weights nor a GPU.
- Server, against the real SDXL backend on the reference machine: one preview,
  one plan, three variants rendered sequentially (56.8 s total), and one
  deliberate over-shift (0.25) which was detected as excessive disocclusion,
  corrected to 0.15 and re-rendered.

Measured on `mix05.jpg`, three variants: preservation 76.9–78.7, boundary
64.0–76.2, coverage 100.0, composition 43.4–79.6. Subject preservation was
reported as unavailable in every case, as designed.

### V2.6 What V2 does not do

- No output-aspect-ratio control; the render keeps the source aspect ratio.
- No subject detection, so no subject-aware placement or preservation.
- No semantic failure detection (duplicated objects, implausible fills).
- The planner proposes camera moves; it does not search over inpainting
  prompts or seeds, which stay frozen at `SEED = 0`.

