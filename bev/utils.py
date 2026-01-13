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
    IMPORTANT:
      nuScenes camera_intrinsic is usually already for the stored image resolution.
      A bad rescale heuristic can *cause* pinwheel/radial stretching.

    This function now only rescales if it is VERY likely that K belongs to a different
    resolution (i.e., inferred size matches a plausible "original size" and differs
    significantly from current img size).

    If cx,cy are not near half-size, we do NOT attempt to "guess" the original size.
    """
    K = np.asarray(K, dtype=np.float64).copy()

    cx, cy = float(K[0, 2]), float(K[1, 2])

    # If principal point is not near the image center, don't guess.
    # (A guess here often breaks projection.)
    if img_w <= 0 or img_h <= 0:
        return K

    if abs(cx - img_w * 0.5) / max(1.0, img_w) > 0.10 or abs(cy - img_h * 0.5) / max(1.0, img_h) > 0.10:
        return K

    # Infer "original" size from center assumption
    w0 = max(1.0, 2.0 * cx)
    h0 = max(1.0, 2.0 * cy)

    sx = float(img_w) / w0
    sy = float(img_h) / h0

    # Only rescale if mismatch is meaningful (>= 5%)
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K

    K[0, 0] *= sx  # fx
    K[0, 2] *= sx  # cx
    K[1, 1] *= sy  # fy
    K[1, 2] *= sy  # cy
    return K
