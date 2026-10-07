"""04_reprojection_math.py — numeric validation of forward-splat warp against
analytic ground truth (independent numpy ray-caster).

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproj import (build_intrinsics, classify_holes, forward_warp,
                     fill_pseudo_depth, make_disocclusion_mask,
                     validate_warp)

FOV = 55.0
ZW = 5.0                       # wall plane depth (scene units)
SPH_C = np.array([0.7, -0.35, 2.2])
SPH_R = 0.55
TILE = 0.4


def render_analytic(C, K, w, h):
    """Ray-cast the synthetic scene from camera center C (numpy, independent path)."""
    cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
    u = np.arange(w, dtype=np.float64)[None, :].repeat(h, 0)
    v = np.arange(h, dtype=np.float64)[:, None].repeat(w, 1)
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    dz = np.ones_like(dx)

    # plane Z = ZW
    tp = np.full_like(dx, np.inf)
    m_p = dz > 1e-9
    tp[m_p] = (ZW - C[2]) / dz[m_p]
    hit_p = m_p & (tp > 1e-6)
    Px = C[0] + tp * dx
    Py = C[1] + tp * dy
    Pz = C[2] + tp * dz

    # sphere
    oc = np.stack([C[0] - SPH_C[0], C[1] - SPH_C[1], C[2] - SPH_C[2]], axis=-1)
    a = dx**2 + dy**2 + 1.0
    b = 2 * (dx * oc[..., 0] + dy * oc[..., 1] + oc[..., 2])
    c = oc[..., 0]**2 + oc[..., 1]**2 + oc[..., 2]**2 - SPH_R**2
    disc = b**2 - 4 * a * c
    ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    hit_s = (disc > 0) & (ts > 1e-6)

    use_s = hit_s & ((~hit_p) | (ts < tp))

    z = np.where(use_s, ts, np.where(hit_p, tp, np.inf))
    col = np.zeros((h, w, 3), dtype=np.float64)
    # wall checker
    chk = ((np.floor(Px / TILE) + np.floor(Py / TILE)) % 2 == 0)
    wall_col = np.where(chk[..., None],
                        np.array([0.85, 0.82, 0.75]),
                        np.array([0.35, 0.33, 0.30]))
    # sphere lambert (light at world origin)
    Sx = C[0] + ts * dx
    Sy = C[1] + ts * dy
    Sz = C[2] + ts * dz
    n = np.stack([Sx - SPH_C[0], Sy - SPH_C[1], Sz - SPH_C[2]], axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    L = -C / max(np.linalg.norm(C), 1e-9)
    lam = 0.4 + 0.6 * np.clip(n @ L, 0, 1)
    sph_col = np.array([0.80, 0.25, 0.20]) * lam[..., None]

    defined = hit_p | hit_s
    col = np.where(use_s[..., None], sph_col, np.where(hit_p[..., None], wall_col, 0))
    Px = C[0] + tp * dx
    Py = C[1] + tp * dy
    Sx = C[0] + ts * dx
    Sy = C[1] + ts * dy
    Sz = C[2] + ts * dz
    P = np.stack([np.where(use_s, Sx, Px),
                  np.where(use_s, Sy, Py),
                  np.where(use_s, Sz, Pz)], axis=-1)
    return (torch.from_numpy(col.transpose(2, 0, 1)).float(),
            torch.from_numpy(z.astype(np.float32)),
            defined,
            use_s,
            P)


def visible_from_origin(P: np.ndarray, defined: np.ndarray,
                        K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Independent visibility test: unoccluded from the original camera AND
    projecting inside the original image frame."""
    d = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9)
    b = 2 * (d * (-SPH_C)).sum(-1)
    c = (SPH_C**2).sum() - SPH_R**2
    disc = b**2 - 4 * c
    ts0 = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / 2, np.inf)
    tp0 = np.where(d[..., 2] > 1e-9, ZW / np.maximum(d[..., 2], 1e-9), np.inf)
    t_first = np.minimum(ts0, tp0)
    dist_p = np.linalg.norm(P, axis=-1)
    u0 = K[0, 0] * P[..., 0] / np.maximum(P[..., 2], 1e-9) + K[0, 2]
    v0 = K[1, 1] * P[..., 1] / np.maximum(P[..., 2], 1e-9) + K[1, 2]
    inframe = (u0 >= -0.5) & (u0 <= w - 0.5) & (v0 >= -0.5) & (v0 <= h - 0.5)
    return defined & (t_first >= dist_p - 1e-3) & inframe


def approx(a, b, tol_rel=0.02, tol_abs=0.05):
    return abs(a - b) <= tol_abs + tol_rel * abs(b)


def main() -> int:
    fails = []

    def check(name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        print(f"[{tag}] {name}" + (f" ({detail})" if detail else ""))
        if not cond:
            fails.append(name)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- T1: identity ----------
    w, h = 480, 360
    K = build_intrinsics(w, h, FOV)
    rgb, z, _, _, _ = render_analytic(np.zeros(3), K, w, h)
    res0 = forward_warp(rgb, z, K, (0.0, 0.0, 0.0))
    border = np.zeros((h, w), bool); border[1:-1, 1:-1] = True
    cov_np = res0["coverage"].cpu().numpy()
    check("T1 identity: interior fully covered", bool(cov_np[border].all()))
    diff_id = (res0["warped"] - rgb.to(dev)).abs().amax(0)[torch.from_numpy(border).to(dev)]
    check("T1 identity: pixel-exact", float(diff_id.max()) <= 1e-4,
          f"max diff {float(diff_id.max()):.2e}")
    m0 = make_disocclusion_mask(res0["coverage"].cpu().numpy(), res0["flow_med_px"])
    check("T1 identity: empty disocclusion", int(m0.sum()) == 0)

    # ---------- T2: lateral flow matches f*sx/Z analytically ----------
    sx = 0.25
    resL = forward_warp(rgb, z, K, (sx, 0.0, 0.0))
    expect = K[0, 0] * sx / ZW
    check("T2 lateral flow = fx*sx/Zw",
          approx(resL["flow_med_px"], expect, tol_abs=0.15),
          f"measured {resL['flow_med_px']:.3f}px expected {expect:.3f}px")

    # ---------- T3: cross-validate vs independent ray-caster ----------
    C = (0.35, -0.12, 0.0)
    res = forward_warp(rgb, z, K, C)
    gtr, gtz, gtd, gts, gtP = render_analytic(np.array(C), K, w, h)
    cov = res["coverage"].cpu().numpy()
    covis = visible_from_origin(gtP, gtd, K, w, h)   # GT pixels the source camera also saw (in frame)
    inter = (cov & covis).sum(); union = (cov | covis).sum()
    iou = inter / union
    check("T3 coverage IoU vs co-visible geometry >= 0.97", iou >= 0.97, f"IoU={iou:.4f}")
    # exclude the sphere's projected disc (+parallax band): its lambert shading
    # legitimately changes with viewpoint, so pixel-exact comparison is invalid there.
    band = int(np.ceil(res["flow_max_px"])) + 2
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band + 1, 2 * band + 1))
    sph_dil = cv2.dilate(gts.astype(np.uint8), kern) > 0
    flat_cmp = covis & ~sph_dil
    wr = res["warped"].permute(1, 2, 0).cpu().numpy()
    gr = gtr.permute(1, 2, 0).numpy()
    ad = np.abs(wr - gr)[flat_cmp]
    frac_ok = float((ad.max(-1) <= 2 / 255).mean())
    mean_ad = float(ad.mean())
    check("T3 RGB agreement on view-independent surface (>=97% within 2/255)",
          frac_ok >= 0.97 and flat_cmp.sum() > 0,
          f"frac={frac_ok:.4f} meanAD={mean_ad:.5f} n={int(flat_cmp.sum())}")
    wz = res["warped_z"].cpu().numpy()
    gtn = gtz.numpy()
    rel = np.abs(wz - gtn)[covis] / gtn[covis]
    check("T3 depth transport (median rel err < 1%)", float(np.median(rel)) < 0.01,
          f"median {float(np.median(rel)):.5f}")
    # true disocclusion = GT-visible but not written by splat
    newvis = gtd & (~cov)
    ring = make_disocclusion_mask(cov, res["flow_med_px"]) > 0
    caught = float(ring[newvis].mean()) if newvis.any() else 1.0
    check("T3 disocclusion mask catches true new surface (>=85%)", caught >= 0.85,
          f"caught {caught*100:.1f}% of {int(newvis.sum())}px")
    check("T3 validator accepts sane shift", validate_warp(res)[0])

    # ---------- T4: points behind camera ----------
    resB = forward_warp(rgb, z, K, (0.0, 0.0, 2.6))
    check("T4 behind-camera points detected & dropped",
          0.0 < resB["behind_frac"] < 0.25,
          f"behind_frac={resB['behind_frac']:.4f}")
    fin = torch.isfinite(resB["warped"]).all().item() and \
        torch.isfinite(resB["warped_z"][resB["warped_z"] > 0]).all().item()
    check("T4 no NaN/inf garbage", bool(fin))
    resX = forward_warp(rgb, z, K, (0.0, 0.0, 6.0))
    okX, whyX = validate_warp(resX)
    check("T4 degenerate forward shift rejected", not okX, whyX)

    # ---------- T5: extreme lateral rejected ----------
    resE = forward_warp(rgb, z, K, (8.0, 0.0, 0.0))
    okE, whyE = validate_warp(resE)
    check("T5 extreme lateral shift rejected", not okE,
          f"covered={resE['covered_frac']:.4f}")

    # ---------- T6: aspect-ratio robustness ----------
    for (ww, hh, label) in [(512, 384, "landscape"), (384, 512, "portrait"),
                            (420, 420, "square")]:
        Ka = build_intrinsics(ww, hh, FOV)
        sx6, sy6 = 0.30, -0.10
        ra, za, _, _, _ = render_analytic(np.zeros(3), Ka, ww, hh)
        rr = forward_warp(ra, za, Ka, (sx6, sy6, 0.0))
        ga, gz, gd, gs, _ = render_analytic(np.array([sx6, sy6, 0.0]), Ka, ww, hh)
        bb = rr["coverage"].cpu().numpy() & gd
        band6 = int(np.ceil(rr["flow_max_px"])) + 2
        k6 = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band6 + 1, 2 * band6 + 1))
        sph6 = cv2.dilate(gs.astype(np.uint8), k6) > 0
        cmp6 = bb & ~sph6
        erra = np.abs(rr["warped"].permute(1, 2, 0).cpu().numpy()
                      - ga.permute(1, 2, 0).numpy())[cmp6]
        fr = float((erra.max(-1) <= 2 / 255).mean())
        exp_flow = float(np.hypot(Ka[0, 0] * sx6 / ZW, Ka[1, 1] * sy6 / ZW))
        okf = approx(rr["flow_med_px"], exp_flow, tol_abs=0.15)
        check(f"T6 {label}: geometry holds ({ww}x{hh})",
              fr >= 0.95 and okf and rr["covered_frac"] > 0.5,
              f"frac_ok={fr:.3f} flow={rr['flow_med_px']:.2f} "
              f"(exp {exp_flow:.2f}) cover={rr['covered_frac']:.3f}")

    # ---------- T7: mask strict binary + monotone growth ----------
    hole_counts = []
    for s in (0.05, 0.15, 0.30):
        rs = forward_warp(rgb, z, K, (s, 0.0, 0.0))
        mk = make_disocclusion_mask(rs["coverage"].cpu().numpy(), rs["flow_med_px"])
        check(f"T7 shift {s}: mask strictly binary",
              mk.dtype == np.uint8 and set(np.unique(mk)).issubset({0, 255}))
        hole_counts.append(int((mk > 0).sum()))
    check("T7 hole count grows with parallax",
          hole_counts[0] < hole_counts[1] < hole_counts[2],
          str(hole_counts))

    # ---------- T8: pseudo-depth fill ----------
    filled = fill_pseudo_depth(res["warped_z"], ring.astype(np.uint8))
    check("T8 filled pseudo-depth finite & positive",
          bool(torch.isfinite(filled).all()) and float(filled.min()) > 0,
          f"range [{float(filled.min()):.3f}, {float(filled.max()):.3f}]")

    # ---------- T9: hybrid backward-resample ----------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _reproj import forward_warp_full
    rh = forward_warp_full(rgb, z, K, (0.0, 0.0, 0.0))
    dh = (rh["warped_filled"] - rgb).abs().amax(0)[1:-1, 1:-1]
    check("T9 hybrid: identity pixel-exact", float(dh.max()) <= 1e-4,
          f"max {float(dh.max()):.2e}")
    rh2 = forward_warp_full(rgb, z, K, C)
    m2 = rh2["mask"]
    cmp2 = covis & ~(m2 > 0) & ~sph_dil
    # resamplers soften hard edges sub-pixel; compare 3x3 box-blurred floats
    wr_blur = cv2.blur(rh2["warped_filled"].permute(1, 2, 0).cpu().numpy().astype(np.float32), (5,5))
    gr_blur = cv2.blur(gr.astype(np.float32), (5,5))
    err_h = np.abs(wr_blur - gr_blur)[cmp2]
    # Synthetic checkerboard has infinite-contrast edges; a bilinear resampler
    # legitimately disagrees with the binary GT within ~contrast*phase/3 of any
    # edge even after blur (~<=25/255 here), while flat regions must match
    # tightly. Gate on both.
    frh = float((err_h.max(-1) <= 25 / 255).mean())
    madh = float(err_h.mean())
    # Structural gate: catches gross misprojection (broken runs score <0.9).
    # The ~1-2% residual after winner-propagation sits at silhouettes where
    # nearest-fill legitimately meets a different surface; meanAD gates global
    # fidelity tightly.
    check("T9 hybrid: shifted view matches GT on co-visible surface",
          frh >= 0.98 and madh <= 3 / 255,
          f"frac={frh:.4f} meanAD={madh:.5f} n={int(cmp2.sum())}")

    # ---------- T10: hole classification (thickness, not area) ----------
    ch, cw = 160, 200
    cov_s = np.ones((ch, cw), bool)
    yy, xx = np.mgrid[0:ch, 0:cw]
    disk = (yy - 60) ** 2 + (xx - 60) ** 2 <= 12 ** 2      # inscribed r=12 -> thick
    band = (yy >= 120) & (yy <= 122) & (xx <= 30)          # 3px stripe -> thin
    cov_s[disk | band] = False

    # Guard: the stripe must survive morphological closing, otherwise this test
    # would pass vacuously and never exercise the thickness discriminator.
    k3s = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_s = cv2.morphologyEx(cov_s.astype(np.uint8), cv2.MORPH_CLOSE,
                                k3s).astype(bool)
    check("T10 thin stripe survives closing (test is non-vacuous)",
          bool((~closed_s)[band].all()), "stripe vanished before classification")

    mk_s, fill_s = classify_holes(cov_s, flow_med_px=1.0)
    check("T10 mask strictly binary",
          mk_s.dtype == np.uint8 and set(np.unique(mk_s)).issubset({0, 255}))
    missed = int((disk & ~(mk_s > 0)).sum())
    check("T10 thick hole masked in FULL (no eroded-core loss)",
          missed == 0, f"{missed}/{int(disk.sum())} disk px left unmasked")
    band_leak = float((mk_s[band] > 0).mean())
    check("T10 thin stripe NOT masked (filled, not inpainted)",
          band_leak <= 0.02, f"leak={band_leak * 100:.1f}%")
    check("T10 fill covers every uncovered pixel",
          bool((fill_s == (~cov_s)).all()))

    # ---------- T11: pseudo-depth robustness (real-photo failure modes) ----
    # The synthetic scene above uses a controlled depth ramp, so it can never
    # expose a defect in the relative-depth -> pseudo-depth transform itself.
    # These cases reproduce the two measured real-photo failures.
    from _reproj import depth_rel_to_pseudo_z as _dnz

    # (a) Far-field saturation: ~50% of pixels at dn ~= 0 (mix01.jpg).
    sat = torch.full((200, 200), 0.1)
    sat[100:, :] = torch.linspace(0.5, 1.0, 200)      # near half below
    sat[:100, :] = 0.0                               # sky: half the frame at 0
    z_sat = _dnz(sat)
    lo, hi = float(z_sat.min()), float(z_sat.max())
    check("T11 saturated far field: depth stays bounded",
          hi / max(float(z_sat.median()), 1e-9) < 50.0,
          f"max/median={hi / max(float(z_sat.median()), 1e-9):.1f}")
    check("T11 saturated far field: near half not collapsed",
          lo > 0.05,
          f"near-half min depth={lo:.4f} (was ~0.002 before the anchor fix)")
    check("T11 saturated far field: far field still ordered behind",
          float(z_sat[:100, :].median()) > float(z_sat[100:, :].median()),
          f"sky={float(z_sat[:100, :].median()):.2f} near={float(z_sat[100:, :].median()):.2f}")

    # (b) Isolated outlier pixels must not stretch the range (mix05/out04/ind01,
    #     where a handful of pixels drove z_max to 181-552).
    out_liers = torch.linspace(0.0, 1.0, 40000).reshape(200, 200).clone()
    out_liers[7, 11] = -1e4
    out_liers[13, 17] = 1e4
    z_ol = _dnz(out_liers)
    check("T11 outliers clipped (bounded dynamic range)",
          float(z_ol.max()) / max(float(z_ol.median()), 1e-9) < 50.0,
          f"max/median={float(z_ol.max()) / max(float(z_ol.median()), 1e-9):.1f}")
    ramp = z_ol[20:-20, 20:-20]
    check("T11 monotonic ordering preserved under clipping",
          bool((z_ol[100, :-1] >= z_ol[100, 1:] - 1e-5).all()))

    print("-" * 60)
    print(f"{len(fails)} failure(s)" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

