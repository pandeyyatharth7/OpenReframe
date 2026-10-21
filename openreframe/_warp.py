"""Forward-splat reprojection core.

Provenance: promoted from `prototypes/_reproj.py` (Sprint 0). That module is now
a re-export shim onto this file so the validation suite in
`prototypes/04_reprojection_math.py` exercises the code that actually ships.

Conventions:
- Pixel coordinates: u right, v down, origin top-left.
- Camera rays: X = (u - cx) * z / fx, Y = (v - cy) * z / fy, Z = z.
- Relative depth (Depth Anything V2) is disparity-like (higher = closer);
  pseudo-depth is derived as z = 1 / (d_norm + eps), unitless.
- Slider translation t_norm = [x, y, fwd] is in units of median pseudo-depth;
  new camera center C = [x*zmed, y*zmed, fwd*zmed]; points map p' = p - C.

The z-buffer splat is deterministic: pass 1 finds the minimum depth per target
bin via scatter_reduce(amin); pass 2 selects, among sources tied with their
bin's winning depth, the lowest source index as the visible writer.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .config import FOV_H_DEG, device

EPS_Z = 1e-4


def build_intrinsics(w: int, h: int, fov_h_deg: float = FOV_H_DEG) -> np.ndarray:
    fx = fy = (w / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def depth_rel_to_pseudo_z(d_rel: np.ndarray | torch.Tensor,
                          lo_pct: float = 1.0, hi_pct: float = 99.0,
                          eps: float = 0.05) -> torch.Tensor:
    """Relative depth (higher=closer, DA-V2) -> unitless pseudo-depth.

    Three robustness measures, all of which the original formula lacked:

    - Range clipping from percentiles rather than min/max. Relative-depth
      checkpoints emit isolated outlier pixels; min/max normalisation lets a
      handful of them stretch the range and squash all real structure.
    - A *bounded* `eps` (0.05, was 1e-3). `1/(dn + eps)` is unbounded as
      eps -> 0, so a pixel with dn ~= 0 reached z = 544 against a median of
      1.0 and dominated the reprojection.
    - The scale is anchored on the median depth of the scene's **near half**
      (`dn > median(dn)`), not on the median of z itself. On images where far
      field saturates -- e.g. mix01.jpg, where ~50% of pixels sit at dn ~= 0 --
      the median of z lands inside the sky plateau, so median normalisation
      rescales the entire foreground to a near-zero depth and it collapses to
      the principal point. Anchoring on the near half is always well defined
      and keeps a bounded ~1/eps dynamic range.
    """
    if isinstance(d_rel, torch.Tensor):
        d = d_rel.detach().float().cpu()
    else:
        d = torch.from_numpy(np.asarray(d_rel, dtype=np.float32))
    d = torch.nan_to_num(d, nan=0.0, posinf=float(d.max() if d.numel() else 1.0),
                         neginf=0.0)
    flat = d.flatten().float()
    lo = torch.quantile(flat, lo_pct / 100.0)
    hi = torch.quantile(flat, hi_pct / 100.0)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or hi - lo < EPS_Z:
        return torch.ones_like(d)
    dn = ((d - lo) / (hi - lo)).clamp(0.0, 1.0)
    z = 1.0 / (dn + eps)

    near = dn > dn.median()
    anchor = z[near].median() if bool(near.any()) else z.median()
    return z / anchor.clamp_min(EPS_Z)


def slider_to_center(t_norm: tuple[float, float, float],
                     z_med: float) -> tuple[float, float, float]:
    x, y, f = t_norm
    return (float(x) * z_med, float(y) * z_med, float(f) * z_med)


def forward_warp(rgb: torch.Tensor, z: torch.Tensor, K: np.ndarray,
                 C: tuple[float, float, float]) -> dict:
    """Z-buffered forward warp of a pinhole view to a translated camera.

    rgb: (3,H,W) float32; z: (H,W) float32 (same units for all axes);
    K: (3,3); C: camera-center offset in z-units. Returns dict with:
      warped (3,H,W), coverage (H,W) bool, warped_z (H,W),
      behind_frac, covered_frac, out_of_frame_frac,
      flow_med_px, flow_max_px
    """
    assert rgb.shape[0] == 3 and z.shape == rgb.shape[1:], "expect CHW"
    dev = device()
    H, W = z.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    rgb_d = rgb.to(dev)
    z_d = z.to(dev).clamp_min(EPS_Z)

    vv, uu = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    X = (uu - cx) * z_d / fx
    Y = (vv - cy) * z_d / fy

    Zp = z_d - C[2]
    Xp = X - C[0]
    Yp = Y - C[1]

    bad = ~torch.isfinite(Zp) | (Zp <= EPS_Z)
    Zsafe = torch.where(bad, torch.ones_like(Zp), Zp)
    u2 = torch.where(bad, uu, fx * Xp / Zsafe + cx)
    v2 = torch.where(bad, vv, fy * Yp / Zsafe + cy)

    ui = torch.round(u2)
    vi = torch.round(v2)
    inb = (~bad) & (ui >= 0) & (ui <= W - 1) & (vi >= 0) & (vi <= H - 1)

    flat_keep = inb.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    ti = (vi.reshape(-1)[flat_keep] * W + ui.reshape(-1)[flat_keep]).long()
    zp = Zp.reshape(-1)[flat_keep]

    # pass 1: winning depth per target bin
    zbuf = torch.full((H * W,), float("inf"), device=dev)
    zbuf.scatter_reduce_(0, ti, zp, reduce="amin", include_self=True)

    cand = zp <= zbuf.gather(0, ti) * (1 + 1e-6) + 1e-6
    cand_idx = flat_keep[cand]                      # source pixel ids (flattened)
    cand_bins = ti[cand]

    # pass 2: deterministic tie-break by lowest source id per bin
    winbuf = torch.full((H * W,), float("inf"), device=dev)
    winbuf.scatter_reduce_(0, cand_bins, cand_idx.float(), reduce="amin",
                           include_self=True)
    hit = torch.isfinite(winbuf)
    winners = winbuf[hit].long()   # flattened source pixel ids

    rgb_pix = rgb_d.permute(1, 2, 0).reshape(-1, 3)   # (H*W, 3) pixel-major rows
    out_rgb = torch.zeros((H * W, 3), device=dev)
    out_z = torch.zeros(H * W, device=dev)
    out_rgb[hit] = rgb_pix[winners]
    out_z[hit] = z_d.reshape(-1)[winners]

    win_field = torch.where(hit, winbuf.clamp(max=H * W - 1),
                            torch.full_like(winbuf, -1.0))
    flow_mag = torch.sqrt((u2 - uu).pow(2) + (v2 - vv).pow(2))
    fv = flow_mag[inb]
    flow_med = fv.median().item() if fv.numel() else 0.0
    flow_max = fv.max().item() if fv.numel() else 0.0

    n_behind = bad.sum().item()
    # Original-frame pixels that are in front of the new camera but land
    # outside its frame. They are the content the move crops away, which is
    # what the composition score needs to judge a shift.
    n_oob = ((~bad) & (~inb)).sum().item()
    return dict(
        warped=out_rgb.view(H, W, 3).permute(2, 0, 1),
        coverage=hit.view(H, W),
        warped_z=out_z.view(H, W),
        winners=win_field.view(H, W).cpu(),
        behind_frac=n_behind / (H * W),
        covered_frac=hit.float().mean().item(),
        out_of_frame_frac=n_oob / (H * W),
        flow_med_px=flow_med,
        flow_max_px=flow_max,
    )


def classify_holes(coverage: np.ndarray, flow_med_px: float,
                   min_inscribed_radius: float = 3.0,
                   min_component_area: int | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split splat holes into (a) magnification speckle/thin stripes ->
    filled but unmasked, and (b) thick true-disocclusion regions -> binary
    {0,255} mask plus smear ring.

    Thickness, not raw area, separates them: undersampling leaves 1-2 px wide
    stripes whose connected components percolate across the frame, while real
    disocclusion crescents are many pixels thick. An area threshold alone is
    useless here because percolation makes stripe components arbitrarily large.

    The thickness test is per connected component: a component survives when its
    largest inscribed radius reaches `min_inscribed_radius`, and then the WHOLE
    component is kept. Keeping only the eroded core would discard the boundary
    shell -- exactly the seam-prone ring that inpainting has to own -- so the
    core is used only to decide membership, never as the mask extent.

    `min_component_area` is a *second*, independent cleanup, applied after
    thickness has made the primary decision. A splat at real-photo
    displacements punches out hundreds to thousands of small isolated holes
    wherever depth is locally inconsistent; inpainting each one separately
    makes the diffusion model emit a flat blob per component (observed as
    scattered grey cubes). Specks below the floor are nearest-winner filled
    instead of inpainted.
    """
    from scipy import ndimage

    cov_u8 = (coverage.astype(bool)).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(cov_u8, cv2.MORPH_CLOSE, k3)
    holes_for_mask = (~closed.astype(bool)).astype(np.uint8)
    raw_holes = (~coverage.astype(bool)).astype(np.uint8)

    h, w = raw_holes.shape
    if min_component_area is None:
        min_component_area = int(max(64, 3.0e-4 * h * w))

    n, labels = cv2.connectedComponents(holes_for_mask, connectivity=8)
    thick = np.zeros_like(holes_for_mask)
    if n > 1:
        dist = cv2.distanceTransform(holes_for_mask, cv2.DIST_L2, 5)
        radii = ndimage.maximum(dist, labels, index=np.arange(1, n))
        areas = ndimage.sum_labels(np.ones_like(holes_for_mask), labels,
                                   index=np.arange(1, n))
        keep = (radii >= min_inscribed_radius) & (areas >= min_component_area)
        keep = np.flatnonzero(keep) + 1
        if keep.size:
            thick[np.isin(labels, keep)] = 1

    k = int(np.clip(round(max(1.0, flow_med_px * 0.25)), 1, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    grown = cv2.dilate(thick, kernel, iterations=1)
    mask = np.where(grown > 0, 255, 0).astype(np.uint8)
    fill = raw_holes   # every uncovered pixel gets a nearest-winner id
    return mask, fill


def fill_channels(x: torch.Tensor, m: np.ndarray) -> torch.Tensor:
    """Nearest-valid fill of masked pixels for a (H,W) or (H,W,K) tensor."""
    from scipy.ndimage import distance_transform_edt
    arr = x.detach().cpu().numpy()
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[..., None]
    hole = m > 0
    if not hole.any():
        t = torch.from_numpy(arr.astype(np.float32))
        return t[..., 0] if squeeze else t
    ind = distance_transform_edt(hole, return_distances=False, return_indices=True)
    out = arr[ind[0], ind[1]]
    t = torch.from_numpy(out.astype(np.float32))
    return t[..., 0] if squeeze else t


def backward_resample(rgb: torch.Tensor, dense_z: torch.Tensor,
                      K: np.ndarray, C: tuple[float, float, float]) -> torch.Tensor:
    """Reconstruct each target pixel's source ray from dense pseudo-depth and
    bilinear-sample the original image."""
    dev = dense_z.device
    H, W = dense_z.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    vv, uu = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    zt = dense_z.clamp_min(EPS_Z)
    Pwx = (uu - cx) * zt / fx + C[0]
    Pwy = (vv - cy) * zt / fy + C[1]
    Pwz = zt + C[2]
    safe = Pwz.clamp_min(EPS_Z)
    us = fx * Pwx / safe + cx
    vs = fy * Pwy / safe + cy
    gx = (us / max(W - 1, 1)) * 2 - 1
    gy = (vs / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = torch.nn.functional.grid_sample(
        rgb[None].to(dev), grid.to(dev), mode="bilinear",
        padding_mode="border", align_corners=True)
    return out[0]


def forward_warp_full(rgb: torch.Tensor, z: torch.Tensor, K: np.ndarray,
                      C: tuple[float, float, float]) -> dict:
    """Hybrid pipeline: validated z-buffered forward splat for geometry;
    thin splat gaps are closed by nearest-winner propagation (surface-true
    colors), thick components remain the true disocclusion mask."""
    base = forward_warp(rgb, z, K, C)
    cov_np = base["coverage"].cpu().numpy()
    mask, fill = classify_holes(cov_np, base["flow_med_px"])
    dev = base["warped"].device
    H, W = base["coverage"].shape
    wf = base.pop("winners").to(dev)
    wf_filled = fill_channels(wf, fill).to(dev)            # nearest-valid winner ids
    ids = wf_filled.reshape(-1).long().clamp_(0, H * W - 1)
    rgb_pix = rgb.to(dev).permute(1, 2, 0).reshape(-1, 3)
    z_flat = z.to(dev).reshape(-1)
    out_rgb = rgb_pix[ids].view(H, W, 3).permute(2, 0, 1)
    dense_z = z_flat[ids].view(H, W)
    return dict(**base, warped_filled=out_rgb.cpu(), mask=mask,
                dense_z=dense_z.cpu())


def make_disocclusion_mask(coverage: np.ndarray,
                           flow_med_px: float,
                           max_dilate: int = 6) -> np.ndarray:
    """Binary {0,255} uint8 mask: true holes plus a smear ring scaled by parallax."""
    holes = (~coverage.astype(bool)).astype(np.uint8)
    k = int(np.clip(round(max(1.0, flow_med_px * 0.5)), 1, max_dilate))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    grown = cv2.dilate(holes, kernel, iterations=1)
    mask = np.where(grown > 0, 255, 0).astype(np.uint8)
    assert set(np.unique(mask)).issubset({0, 255})
    return mask


def validate_warp(res: dict, min_covered: float = 0.05,
                  max_holes: float = 0.90) -> tuple[bool, str]:
    cf, hf = res["covered_frac"], res["behind_frac"]
    if cf < min_covered:
        return False, f"covered fraction {cf:.3f} below floor {min_covered} (shift too extreme)"
    if 1.0 - cf > max_holes:
        return False, f"disocclusion fraction {1-cf:.3f} above cap {max_holes}"
    if hf > 0.5:
        return False, f"{hf*100:.1f}% of points fall behind new camera"
    if not np.isfinite(res["warped_z"].cpu().numpy()).all():
        return False, "non-finite values in warped depth"
    return True, "ok"


def render_preview_with_gaps(warped: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    """uint8 RGB visualization: gaps tinted magenta."""
    img = (warped.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img[mask > 0] = (255, 0, 255)
    return img


def fill_pseudo_depth(warped_z: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Nearest-valid-pixel fill of masked regions (scipy distance transform)."""
    from scipy.ndimage import distance_transform_edt
    z = warped_z.cpu().numpy().astype(np.float32)
    hole = mask > 0
    if not hole.any():
        return torch.from_numpy(z)
    ind = distance_transform_edt(hole, return_distances=False, return_indices=True)
    filled = z[tuple(ind)].reshape(z.shape)
    return torch.from_numpy(np.nan_to_num(filled,
                                          nan=float(np.nanmedian(z)))).float()