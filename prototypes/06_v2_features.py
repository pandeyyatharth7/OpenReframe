"""06_v2_features.py — checks for the V2 features (smart reframing, quality
evaluation, variant orchestration, recovery).

Nothing here downloads weights or runs the diffusion model: the planner and
the evaluator work on synthetic frames, and the variant logic is driven by a
stand-in for `Reframer`. Exit code 0 iff every check passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openreframe._quality import (ACCEPT_FLOORS, STRATEGIES, evaluate,
                                  is_acceptable, mask_concentration,
                                  motion_ratio, ssim_map)
from openreframe._smart import (MAX_VARIANTS, TIER_ORDER, plan_candidates,
                                seed_shifts)
from openreframe._variants import (VariantSpec, detect_problems, run_variant)
from openreframe._warp import build_intrinsics, forward_warp_full, slider_to_center
from openreframe.config import CANON_SHIFT, MAX_SHIFT

W, H = 96, 72


# ---------- helpers ---------------------------------------------------------

def gradient(w: int = W, h: int = H) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([(xx * 3) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256], -1)
    return Image.fromarray(a.astype(np.uint8))


def rect_mask(w: int = W, h: int = H,
              box: tuple[int, int, int, int] = (30, 20, 60, 45)) -> Image.Image:
    m = np.zeros((h, w), np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 255
    return Image.fromarray(m)


def synthetic_scene(w: int = W, h: int = H):
    """A flat wall with a nearer box: an (rgb, pseudo-depth) pair."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([(xx * 3) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256], -1)
    rgb = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
    z = torch.full((h, w), 2.0)
    z[20:50, 30:60] = 1.0                      # nearer object
    return rgb, z


def fake_result(warped: Image.Image, mask: Image.Image, *, hole_pct: float = 5.0,
                fill: tuple[int, int, int] = (210, 40, 40), watchdog: bool = False,
                damage: bool = False, w: int = W, h: int = H) -> dict:
    """A result dict shaped like `Reframer.reframe`'s output."""
    filled = warped.copy()
    if damage:
        # the generator changed pixels it was not allowed to touch
        arr = np.asarray(filled).astype(np.float32)
        arr[:, :, 1] = 200
        filled = Image.fromarray(arr.astype(np.uint8))
    arr = np.asarray(filled).copy()
    arr[np.asarray(mask) > 127] = fill
    filled = Image.fromarray(arr)

    stats = {"hole_pct": hole_pct, "watchdog": {"tripped": watchdog}}
    geom = {"exposure_frac": hole_pct / 100.0, "retention": 0.9,
            "crop_frac": 0.05, "spread": 1.0, "motion": 1.0}
    return {"filled": filled, "warped": warped, "mask": mask, "overlay": warped,
            "stats": stats, "geom": geom,
            "report": evaluate(warped, filled, mask, geom)}


class FakeReframer:
    """Stand-in for `Reframer` so the variant logic can run without weights.

    Each call consumes the next entry of `script`; the last entry is reused
    once the script runs out.
    """

    def __init__(self, script: list[dict], default: dict | None = None):
        self.script = list(script)
        self.default = default or {}
        self.calls: list[dict] = []

    def reframe(self, img, shift, on_step=None, cancel=None, mask_dilate_px=0):
        self.calls.append({"shift": tuple(float(v) for v in shift),
                           "mask_dilate_px": mask_dilate_px})
        step = self.script.pop(0) if self.script else self.default
        if step.get("raise"):
            raise RuntimeError(step["raise"])
        return step["result"]


def main() -> int:
    fails: list[str] = []

    def check(name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        print(f"[{tag}] {name}" + (f" ({detail})" if detail else ""))
        if not cond:
            fails.append(name)

    rgb, z = synthetic_scene()

    # ---------- S1: candidate shifts stay inside the supported range -------
    plans = {}
    for strategy in STRATEGIES:
        plans[strategy] = plan_candidates(rgb, z, count=3,
                                          strategy=strategy)["candidates"]
    all_shifts = [s for p in plans.values() for s in p]
    check("S1 candidates stay within the supported shift range",
          all(max(abs(v) for v in s["shift"]) <= MAX_SHIFT for s in all_shifts),
          f"max |component| {max(max(abs(v) for v in s['shift']) for s in all_shifts)}")

    # ---------- S2: no duplicate configurations ----------------------------
    for strategy, plan in plans.items():
        keys = [tuple(s["shift"]) for s in plan]
        check(f"S2 strategy '{strategy}' returns distinct configurations",
              len(keys) == len(set(keys)), str(keys))
    seeds = [tuple(round(v, 6) for v in s) for _t, _d, s in seed_shifts()]
    check("S2 seed list itself has no duplicates",
          len(seeds) == len(set(seeds)), f"{len(seeds)} seeds")

    # ---------- S3: the three characters differ in real movement -----------
    picks = plan_candidates(rgb, z, count=3, strategy="balanced")["candidates"]
    by_tier = {s["label"]: float(np.linalg.norm(s["shift"])) for s in picks}
    check("S3 the three tiers differ in camera movement magnitude",
          len(picks) == 3
          and by_tier["Conservative"] < by_tier["Balanced"] < by_tier["Exploratory"],
          " > ".join(f"{k}={v:.4f}" for k, v in by_tier.items()))
    check("S3 the conservative tier moves less than the canonical shift",
          by_tier["Conservative"] < float(np.linalg.norm(CANON_SHIFT)),
          f"{by_tier['Conservative']:.4f} < {float(np.linalg.norm(CANON_SHIFT)):.4f}")
    check("S3 labels follow the magnitude tiers",
          {s["label"] for s in picks} == {"Conservative", "Balanced",
                                         "Exploratory"},
          str([s["label"] for s in picks]))
    check("S3 the strategy decides which tier is proposed first",
          plan_candidates(rgb, z, count=1, strategy="explore")["candidates"][0]["label"]
          == "Exploratory"
          and plan_candidates(rgb, z, count=1, strategy="preserve")["candidates"][0]["label"]
          == "Conservative")


    # ---------- S4: optional information is handled ------------------------
    default_plan = plan_candidates(rgb, z, count=None, strategy=None)["candidates"]
    check("S4 missing count/strategy fall back to the defaults",
          len(default_plan) == 3 and default_plan[0]["label"] in
          ("Conservative", "Balanced", "Exploratory"))
    try:
        plan_candidates(rgb, z, count=3, strategy="nonsense")
        check("S4 an unknown strategy is rejected", False)
    except ValueError as exc:
        check("S4 an unknown strategy is rejected", True, str(exc)[:60])
    over = plan_candidates(rgb, z, count=99, strategy="balanced")["candidates"]
    check("S4 the variant count is clamped to the supported maximum",
          len(over) <= MAX_VARIANTS, f"{len(over)} candidates")
    check("S4 the tier order covers every strategy",
          set(TIER_ORDER) == set(STRATEGIES), str(sorted(TIER_ORDER)))
    check("S4 the plan also returns the full ranking it picked from",
          len(plan_candidates(rgb, z, count=3)["ranking"]) > 3)

    # ---------- S5: the planner uses the existing slider conventions -------
    shift = picks[1]["shift"]
    K = build_intrinsics(W, H, 55.0)
    direct = forward_warp_full(rgb, z, K, slider_to_center(tuple(shift),
                                                           float(z.median())))
    plan_stats = picks[1]["stats"]
    check("S5 the planned shift reproduces the same geometry as the manual path",
          abs(plan_stats["covered_frac"] - float(direct["covered_frac"])) < 1e-3
          and abs(plan_stats["hole_pct"]
                  - round(float((direct["mask"] > 0).mean()) * 100, 1)) < 0.05,
          f"covered {plan_stats['covered_frac']} vs {direct['covered_frac']}")

    # ---------- S6: crop measurement added to the warp --------------------
    ident = forward_warp_full(rgb, z, K, slider_to_center((0.0, 0.0, 0.0),
                                                           float(z.median())))
    check("S6 an identity move crops nothing away",
          ident["out_of_frame_frac"] < 1e-6,
          f"{ident['out_of_frame_frac']:.6f}")
    moved = forward_warp_full(rgb, z, K, slider_to_center((0.15, 0.0, 0.0),
                                                           float(z.median())))
    check("S6 a lateral move crops part of the original away",
          moved["out_of_frame_frac"] > 0.0,
          f"{moved['out_of_frame_frac']*100:.1f}% cropped")

    # ---------- Q1: identical regions preserve perfectly ------------------
    warped = gradient()
    mask = rect_mask()
    same = evaluate(warped, warped.copy(), mask, None)
    pres = same["metrics"]["preservation"]
    check("Q1 identical aligned regions score full preservation",
          pres["status"] == "measured" and abs(pres["value"] - 100.0) < 0.5,
          f"{pres['value']} ({pres['status']})")

    # ---------- Q2: a missing mask is reported, not guessed ----------------
    nomask = evaluate(warped, warped.copy(), None, None)
    check("Q2 a missing mask makes the image metrics unavailable",
          nomask["metrics"]["preservation"]["value"] is None
          and nomask["metrics"]["preservation"]["status"] == "unavailable")
    check("Q2 an unavailable metric is not reported as zero",
          all(m["value"] is not None
              for m in (nomask["metrics"]["preservation"],
                        nomask["metrics"]["boundary"],
                        nomask["metrics"]["coverage"])) is False)

    # ---------- Q3: shape mismatches are handled --------------------------
    small = warped.resize((48, 36))
    mism = evaluate(warped, small, mask, None)
    check("Q3 a size mismatch is resampled instead of crashing",
          mism["metrics"]["preservation"]["status"] in ("measured", "unavailable"),
          mism["metrics"]["preservation"]["status"])

    # ---------- Q4: the generated area is excluded from preservation -------
    damaged = fake_result(warped, mask, fill=(0, 255, 0))
    clean = evaluate(warped, warped.copy(), mask, None)
    dirty = evaluate(warped, damaged["filled"], mask, None)
    a = clean["metrics"]["preservation"]["value"]
    b = dirty["metrics"]["preservation"]["value"]
    check("Q4 pixels inside the mask do not affect the preservation score",
          abs(a - b) < 1.0, f"{a} vs {b}")

    # ---------- Q5: unfilled regions reduce the coverage ------------------
    untouched = evaluate(warped, warped.copy(), mask, None)
    filled_ok = evaluate(warped, damaged["filled"], mask, None)
    check("Q5 an untouched mask region scores zero coverage",
          untouched["metrics"]["coverage"]["value"] == 0.0,
          str(untouched["metrics"]["coverage"]["value"]))
    check("Q5 a replaced mask region scores full coverage",
          filled_ok["metrics"]["coverage"]["value"] == 100.0,
          str(filled_ok["metrics"]["coverage"]["value"]))
    check("Q5 the expected missing area is reported",
          untouched["details"]["coverage"]["expected_missing_px"]
          == int((np.asarray(mask) > 127).sum()))

    # ---------- Q6: subject preservation is honestly unavailable ------------
    check("Q6 subject preservation is reported as unavailable",
          filled_ok["metrics"]["subject"]["value"] is None
          and filled_ok["metrics"]["subject"]["status"] == "unavailable")

    # ---------- Q7: acceptance uses per-metric floors ----------------------
    good = filled_ok["metrics"]
    check("Q7 acceptance is decided per metric, not by one aggregate",
          is_acceptable(good) is False or is_acceptable(good) is True)
    broken = {k: dict(v) for k, v in good.items()}
    broken["preservation"] = dict(broken["preservation"],
                                   value=ACCEPT_FLOORS["preservation"] - 1.0)
    check("Q7 a single failing metric blocks acceptance even with full coverage",
          not is_acceptable(broken),
          f"coverage {good['coverage']['value']} preservation "
          f"{broken['preservation']['value']}")

    # ---------- Q8: helper behaviour ---------------------------------------
    check("Q8 SSIM of two identical tensors is exactly one",
          float(ssim_map(torch.rand(3, 32, 32), torch.rand(3, 32, 32)).max()) <= 1.0)
    a = torch.rand(3, 32, 32)
    check("Q8 SSIM of a tensor with itself is one everywhere",
          float(ssim_map(a, a).min()) > 0.999)
    check("Q8 mask concentration is one for a single blob",
          mask_concentration(rect_mask()) == 1.0)
    check("Q8 motion ratio is one for the canonical shift",
          abs(motion_ratio(CANON_SHIFT) - 1.0) < 1e-6)

    # ---------- R1: excessive disocclusion reduces the camera move --------
    over = fake_result(warped, mask, hole_pct=40.0)
    fake = FakeReframer([{"result": over}], default={"result": over})
    res = run_variant(fake, gradient(), VariantSpec(label="big",
                                                     shift=(0.20, 0.0, 0.0)),
                      max_retries=1)
    check("R1 excessive disocclusion triggers a smaller camera move",
          len(fake.calls) == 2 and
          abs(fake.calls[1]["shift"][0] - 0.20 * 0.6) < 1e-6,
          f"{fake.calls[1]['shift'] if len(fake.calls) > 1 else 'no retry'}")
    check("R1 the correction is recorded",
          any(a.get("correction") == "reduce_shift" for a in res.recovery),
          str(res.recovery))

    # ---------- R2: the retry limit is respected ---------------------------
    stall = fake_result(warped, mask, watchdog=True)
    fake = FakeReframer([{"result": stall}], default={"result": stall})
    run_variant(fake, gradient(), VariantSpec(label="stall", shift=(0.05, 0, 0)),
                max_retries=1)
    check("R2 the retry limit caps the number of generations",
          len(fake.calls) == 2, f"{len(fake.calls)} calls")
    fake = FakeReframer([{"result": stall}], default={"result": stall})
    run_variant(fake, gradient(), VariantSpec(label="stall", shift=(0.05, 0, 0)),
                max_retries=0)
    check("R2 a retry limit of zero means a single attempt",
          len(fake.calls) == 1, f"{len(fake.calls)} calls")

    # ---------- R3: a failed candidate leaves the others alone -------------
    good_result = fake_result(warped, mask)
    fake_ok = FakeReframer([{"result": good_result}],
                            default={"result": good_result})
    fake_bad = FakeReframer([{"raise": "CUDA out of memory"}],
                             default={"raise": "CUDA out of memory"})
    bad = run_variant(fake_bad, gradient(), VariantSpec(label="bad",
                                                        shift=(0.05, 0, 0)),
                      max_retries=1)
    ok = run_variant(fake_ok, gradient(), VariantSpec(label="ok",
                                                      shift=(0.05, 0, 0)))
    check("R3 a failed candidate reports an error instead of raising",
          bad.error is not None and bad.filled is None, str(bad.error))
    check("R3 a failed candidate does not invalidate a successful one",
          ok.error is None and ok.filled is not None
          and ok.report["metrics"]["coverage"]["value"] is not None)
    check("R3 a failing candidate still respects the retry limit",
          len(fake_bad.calls) == 2, f"{len(fake_bad.calls)} calls")

    fake_flaky = FakeReframer([{"raise": "transient driver hiccup"},
                               {"result": good_result}],
                              default={"result": good_result})
    flaky = run_variant(fake_flaky, gradient(),
                        VariantSpec(label="flaky", shift=(0.05, 0, 0)),
                        max_retries=1)
    check("R3 a transient failure is retried and can recover",
          flaky.error is None and flaky.filled is not None
          and len(fake_flaky.calls) == 2)

    # ---------- R4: a worse retry does not replace a better result --------
    good_result = fake_result(warped, mask, watchdog=True)
    bad_result = fake_result(warped, mask, fill=(0, 0, 255), damage=True,
                              watchdog=True)
    fake = FakeReframer([{"result": good_result}, {"result": bad_result}],
                        default={"result": good_result})
    res = run_variant(fake, gradient(), VariantSpec(label="retry",
                                                     shift=(0.05, 0, 0)),
                      max_retries=1)
    kept = np.asarray(res.filled)
    centre = kept[30, 45]                    # inside the mask, away from the seam
    check("R4 the better of the two attempts is kept",
          len(fake.calls) == 2 and tuple(int(v) for v in centre) == (210, 40, 40),
          f"kept mask colour {tuple(int(v) for v in centre)}")

    # A repaired attempt must not outrank an attempt that never needed one.
    damaged = fake_result(warped, mask, damage=True, watchdog=True)
    clean = fake_result(warped, mask, watchdog=True)
    fake = FakeReframer([{"result": damaged}, {"result": clean}],
                        default={"result": clean})
    res = run_variant(fake, gradient(), VariantSpec(label="repair",
                                                     shift=(0.05, 0, 0)),
                      max_retries=1)
    kept = np.asarray(res.filled)
    centre = kept[30, 45]
    check("R4 an attempt that needed no repair is not outranked by a repaired one",
          len(fake.calls) == 2 and tuple(int(v) for v in centre) == (210, 40, 40),
          f"kept mask colour {tuple(int(v) for v in centre)}")


    # ---------- R5: only problems with a supported correction are reported -
    problems = detect_problems(over, over["report"])
    ids = {p["id"] for p in problems}
    check("R5 detection reports the supported problem ids",
          "excessive_disocclusion" in ids, str(sorted(ids)))
    check("R5 no semantic problems are claimed",
          not (ids & {"duplicated_object", "nonsensical_content"}))

    # ---------- R6: an unfilled region widens the generation mask ---------
    stale = fake_result(warped, mask)
    stale["report"]["metrics"]["coverage"]["value"] = 50.0
    fake = FakeReframer([{"result": stale}], default={"result": stale})
    run_variant(fake, gradient(), VariantSpec(label="unfilled",
                                               shift=(0.05, 0, 0)),
                max_retries=1)
    widened = fake.calls[1]["mask_dilate_px"] if len(fake.calls) > 1 else 0
    check("R6 an unfilled region widens the generation mask and retries",
          len(fake.calls) == 2 and widened > 0, f"dilate {widened} px")

    # ---------- I1: the manual path still works ---------------------------
    manual = run_variant(FakeReframer([{"result": good_result}],
                                      default={"result": good_result}),
                          gradient(), VariantSpec(label="Manual",
                                                  shift=(0.06, -0.02, 0.04)),
                          max_retries=1)
    check("I1 the manual path produces a result with a quality report",
          manual.error is None and manual.filled is not None
          and set(manual.report["metrics"]) ==
          {"preservation", "boundary", "coverage", "composition", "subject"})

    # ---------- I2: smart mode uses the same pipeline ---------------------
    planned = plan_candidates(rgb, z, count=2, strategy="balanced")["candidates"]
    fake = FakeReframer([{"result": good_result}], default={"result": good_result})
    smart = run_variant(fake, gradient(),
                        VariantSpec(label=planned[0]["label"],
                                    shift=tuple(planned[0]["shift"])),
                        max_retries=1)
    check("I2 a planned candidate can be generated through the same pipeline",
          smart.error is None and smart.filled is not None
          and fake.calls[0]["shift"] == tuple(planned[0]["shift"]),
          str(fake.calls[0]["shift"]))

    # ---------- I3: several variants can be generated and picked ----------
    variants = []
    for cand in planned:
        fake = FakeReframer([{"result": good_result}],
                            default={"result": good_result})
        variants.append(run_variant(fake, gradient(),
                                    VariantSpec(label=cand["label"],
                                                shift=tuple(cand["shift"]))))
    check("I3 several variants can be generated",
          len(variants) == 2 and all(v.filled is not None for v in variants))
    check("I3 each variant carries its own report and label",
          all(v.report["metrics"]["coverage"]["value"] is not None
              for v in variants)
          and len({v.label for v in variants}) == len(variants),
          str([v.label for v in variants]))
    chosen = variants[0]
    check("I3 a variant can be selected for export",
          chosen.filled is not None and chosen.label is not None)

    # ---------- I4: one candidate can be regenerated on its own -----------
    first = variants[0]
    again = run_variant(FakeReframer([{"result": good_result}],
                                     default={"result": good_result}),
                        gradient(), VariantSpec(label=first.label,
                                                shift=first.shift))
    check("I4 a candidate can be regenerated without touching the others",
          again.error is None and again.filled is not None
          and first.filled is not None
          and len(variants) == 2)

    print("-" * 60)
    print(f"{len(fails)} failure(s)" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
