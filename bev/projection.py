from __future__ import annotations
import numpy as np

def project_ego_grid_to_image_maps(
    pts_ego: np.ndarray,          # (H,W,3)
    T_cam_from_ego: np.ndarray,   # (4,4)
    K: np.ndarray,                # (3,3)
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      map_u, map_v: float32 (H,W) for cv2.remap
      valid: bool (H,W)
    """
    H, W, _ = pts_ego.shape

    # Homogeneous ego points
    ones = np.ones((H, W, 1), dtype=np.float64)
    pts_h = np.concatenate([pts_ego.astype(np.float64), ones], axis=-1)  # (H,W,4)

    # Ego -> Cam
    pts_cam = pts_h @ T_cam_from_ego.T   # (H,W,4)
    X = pts_cam[..., 0]
    Y = pts_cam[..., 1]
    Z = pts_cam[..., 2]

    # In front of camera
    eps = 1e-6
    in_front = Z > eps

    # Project using K: u = fx*X/Z + cx, v = fy*Y/Z + cy
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]

    u = fx * (X / (Z + eps)) + cx
    v = fy * (Y / (Z + eps)) + cy

    # Bounds check
    inside = (u >= 0) & (u < (img_w - 1)) & (v >= 0) & (v < (img_h - 1))
    valid = in_front & inside

    # cv2.remap wants float32 maps
    map_u = u.astype(np.float32)
    map_v = v.astype(np.float32)

    return map_u, map_v, valid
