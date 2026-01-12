import numpy as np
from pyquaternion import Quaternion

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
