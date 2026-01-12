from __future__ import annotations

from typing import Tuple
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

    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (Z > float(min_depth))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # OpenCV camera convention: x right, y down, z forward
    u = fx * (X / (Z + 1e-12)) + cx
    v = fy * (Y / (Z + 1e-12)) + cy

    u = u.reshape(orig_shape)
    v = v.reshape(orig_shape)
    valid = valid.reshape(orig_shape)
    return u, v, valid


def warp_image_to_bev(
    img_bgr: np.ndarray,
    grid: BEVGrid,
    K: np.ndarray,
    T_cam_from_ego: np.ndarray,
    z_plane: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-perspective mapping (IPM): for each BEV cell center, sample the
    corresponding camera pixel (u,v) and remap.

    Returns:
      warped_bgr: (H,W,3) uint8
      valid_mask: (H,W) bool (u/v inside image AND depth positive)
    """
    pts_ego = grid.ego_points(z=z_plane)  # (H,W,3)
    u, v, valid = project_ego_points_to_image(pts_ego, K, T_cam_from_ego)  # (H,W)
    u_map = u.astype(np.float32)
    v_map = v.astype(np.float32)

    ih, iw = img_bgr.shape[:2]
    inside = (u_map >= 0.0) & (u_map <= (iw - 1)) & (v_map >= 0.0) & (v_map <= (ih - 1))
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
