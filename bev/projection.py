from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import cv2

from bev_grid import BEVGrid


def project_ego_points_to_image(
    pts_ego: np.ndarray,
    K: np.ndarray,
    T_cam_from_ego: np.ndarray,
    min_depth: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects ego-frame 3D points to image pixel coordinates.

    Args:
      pts_ego: (..., 3) float array in ego frame (meters).
      K: (3,3) camera intrinsics.
      T_cam_from_ego: (4,4) SE(3), transforms ego -> camera.
      min_depth: Z (camera forward) must be > min_depth to be valid.

    Returns:
      u: (...) float pixels
      v: (...) float pixels
      valid: (...) bool mask (in front of camera + finite)
    """
    pts = np.asarray(pts_ego, dtype=np.float32)
    orig_shape = pts.shape[:-1]
    pts = pts.reshape(-1, 3)

    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    pts_cam_h = (T_cam_from_ego.astype(np.float32) @ pts_h.T).T  # (N,4)

    X = pts_cam_h[:, 0]
    Y = pts_cam_h[:, 1]
    Z = pts_cam_h[:, 2]

    # Pinhole projection assumes camera coords are OpenCV-like:
    # x right, y down, z forward.
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (Z > float(min_depth))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (X / (Z + 1e-12)) + cx
    v = fy * (Y / (Z + 1e-12)) + cy

    u = u.reshape(orig_shape)
    v = v.reshape(orig_shape)
    valid = valid.reshape(orig_shape)
    return u, v, valid


def splat_image_to_bev_ground(
    img_bgr: np.ndarray,
    grid: BEVGrid,
    K: np.ndarray,
    T_ego_from_cam: np.ndarray,   # cam -> ego
    z_plane: float = 0.0,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    ih, iw = img_bgr.shape[:2]
    K = K.astype(np.float64)
    Kinv = np.linalg.inv(K)

    # pixel grid (centers)
    us = (np.arange(0, iw, stride, dtype=np.float64) + 0.5)
    vs = (np.arange(0, ih, stride, dtype=np.float64) + 0.5)
    U, V = np.meshgrid(us, vs)  # (h', w')
    h2, w2 = U.shape
    N = h2 * w2

    pix = np.stack([U, V, np.ones_like(U)], axis=-1).reshape(-1, 3).T  # (3, N)

    # rays in cam
    d_cam = Kinv @ pix  # (3, N)

    # cam->ego
    R = T_ego_from_cam[:3, :3].astype(np.float64)
    t = T_ego_from_cam[:3, 3].astype(np.float64)  # (3,)
    d_ego = R @ d_cam  # (3, N)

    # ray-plane intersection: p = t + s*d, with p_z = z_plane
    denom = d_ego[2, :]  # (N,)
    valid = np.abs(denom) > 1e-9

    s = np.zeros((N,), dtype=np.float64)
    s[valid] = (float(z_plane) - t[2]) / denom[valid]

    # require intersection in front of camera
    valid &= (s > 1e-6)

    # intersection points in ego
    X = t[0] + d_ego[0, :] * s
    Y = t[1] + d_ego[1, :] * s

    # map to BEV indices (float)
    ii, jj = grid.ego_xy_to_ij(X, Y)

    # inside BEV (leave 1px margin for bilinear +1)
    valid &= (ii >= 0) & (ii < grid.H - 1) & (jj >= 0) & (jj < grid.W - 1)

    if not np.any(valid):
        return np.zeros((grid.H, grid.W, 3), dtype=np.uint8), np.zeros((grid.H, grid.W), dtype=np.float32)

    # bilinear weights
    i0 = np.floor(ii).astype(np.int32)
    j0 = np.floor(jj).astype(np.int32)
    di = (ii - i0).astype(np.float32)
    dj = (jj - j0).astype(np.float32)

    w00 = (1 - di) * (1 - dj)
    w10 = di * (1 - dj)
    w01 = (1 - di) * dj
    w11 = di * dj

    # sample colors from strided pixel positions matching pix flatten order
    idx = np.arange(N, dtype=np.int64)
    u_idx = (idx % w2)
    v_idx = (idx // w2)
    u_px = (u_idx * stride).astype(np.int32)
    v_px = (v_idx * stride).astype(np.int32)
    cols = img_bgr[v_px, u_px].astype(np.float32)  # (N,3)

    # apply mask once
    vmask = valid
    i0m = i0[vmask]
    j0m = j0[vmask]
    cols_m = cols[vmask]

    w00m = w00[vmask]
    w10m = w10[vmask]
    w01m = w01[vmask]
    w11m = w11[vmask]

    acc = np.zeros((grid.H, grid.W, 3), dtype=np.float32)
    wsum = np.zeros((grid.H, grid.W), dtype=np.float32)

    def add(i, j, w):
        # accumulate per-channel safely with np.add.at
        np.add.at(wsum, (i, j), w)
        for c in range(3):
            np.add.at(acc[..., c], (i, j), cols_m[:, c] * w)

    add(i0m,     j0m,     w00m)
    add(i0m + 1, j0m,     w10m)
    add(i0m,     j0m + 1, w01m)
    add(i0m + 1, j0m + 1, w11m)

    out = np.zeros((grid.H, grid.W, 3), dtype=np.uint8)
    ok = wsum > 1e-6
    out[ok] = (acc[ok] / wsum[ok, None]).clip(0, 255).astype(np.uint8)
    return out, wsum



def warp_image_to_bev(
    img_bgr: np.ndarray,
    grid: BEVGrid,
    K: np.ndarray,
    T_cam_from_ego: np.ndarray,
    z_plane: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
    # --- NEW: ROI gating to reduce IPM pinwheel artifacts ---
    roi_vmin: Optional[float] = None,
    roi_vmax: Optional[float] = None,
    roi_umin: Optional[float] = None,
    roi_umax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-perspective mapping (IPM): for each BEV cell center, sample the
    corresponding camera pixel (u,v) and remap.

    NEW: Optional ROI gating in the SOURCE image (u/v) to reduce radial stretching.
         This is a pragmatic Phase-1 fix for planar IPM: we ignore pixels near the
         horizon/upper image which tend to be non-ground / far-range and cause
         strong "pinwheel" smearing.

    Returns:
      warped_bgr: (H,W,3) uint8
      valid_mask: (H,W) bool (u/v inside image AND depth positive AND inside ROI)
    """
    pts_ego = grid.ego_points(z=z_plane)  # (H,W,3)
    u, v, valid = project_ego_points_to_image(pts_ego, K, T_cam_from_ego)  # (H,W)

    u_map = u.astype(np.float32)
    v_map = v.astype(np.float32)

    ih, iw = img_bgr.shape[:2]

    inside = (u_map >= 0.0) & (u_map <= (iw - 1)) & (v_map >= 0.0) & (v_map <= (ih - 1))

    # ROI gating (in pixel coordinates)
    if roi_vmin is not None:
        inside &= (v_map >= float(roi_vmin))
    if roi_vmax is not None:
        inside &= (v_map <= float(roi_vmax))
    if roi_umin is not None:
        inside &= (u_map >= float(roi_umin))
    if roi_umax is not None:
        inside &= (u_map <= float(roi_umax))

    valid_mask = valid & inside

    warped = cv2.remap(
        img_bgr,
        u_map,
        v_map,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped[~valid_mask] = 0
    return warped, valid_mask
