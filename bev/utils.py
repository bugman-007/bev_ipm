from __future__ import annotations
import os
import numpy as np
from pyquaternion import Quaternion

def info(msg: str) -> None:
    print(msg, flush=True)

def assert_file_exists(path: str, what: str = "file") -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {what}: {path}")

def assert_dir_exists(path: str, what: str = "directory") -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Missing {what}: {path}")

def quat_to_rotmat(qwqxqyqz) -> np.ndarray:
    # nuScenes stores rotation as [w, x, y, z]
    q = Quaternion(qwqxqyqz)
    return q.rotation_matrix.astype(np.float64)

def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rescale_intrinsics_to_image(K: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    If K was computed for a different image resolution than the one we're sampling,
    rescale fx,fy,cx,cy to the actual image size.

    Heuristic:
      - Assume principal point is near the center.
      - Infer the "original" size from (cx,cy) ~ (W/2, H/2).
    """
    K = np.asarray(K, dtype=np.float64).copy()

    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Infer the calibration image size.
    # (nuScenes cx,cy are very close to W/2,H/2; if they’re not, we still only use it as a scaling heuristic)
    w0 = max(1.0, 2.0 * cx)
    h0 = max(1.0, 2.0 * cy)

    sx = float(img_w) / w0
    sy = float(img_h) / h0

    # Only apply if there is a meaningful mismatch (avoid tiny float noise)
    if abs(sx - 1.0) > 1e-2 or abs(sy - 1.0) > 1e-2:
        K[0, 0] *= sx  # fx
        K[0, 2] *= sx  # cx
        K[1, 1] *= sy  # fy
        K[1, 2] *= sy  # cy

    return K
