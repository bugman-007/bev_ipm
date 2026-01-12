import numpy as np
from typing import Any
from .utils import quat_to_rotmat, make_se3, invert_se3
from dataclasses import dataclass

@dataclass
class CameraCalibration:
    channel: str
    K: np.ndarray                 # 3x3
    T_ego_from_cam: np.ndarray    # 4x4  (camera -> ego)
    T_cam_from_ego: np.ndarray    # 4x4  (ego -> camera)
    T_global_from_ego: np.ndarray # 4x4
    T_ego_from_global: np.ndarray # 4x4


def get_camera_calibration_for_sample(
    nusc: NuScenes,
    selection: SampleSelection
) -> Dict[str, CameraCalibration]:
    """
    Step 2: For each camera in selection, extract intrinsics, extrinsics, ego pose,
    and build SE(3) transforms.
    """
    import numpy as np

    out: Dict[str, CameraCalibration] = {}

    for ch, cam in selection.cameras.items():
        sd = nusc.get("sample_data", cam.sample_data_token)

        # --- calibrated sensor (camera <-> ego) ---
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"], dtype=np.float64)  # 3x3

        R_ego_from_cam = quat_to_rotmat(cs["rotation"])
        t_ego_from_cam = np.array(cs["translation"], dtype=np.float64)
        T_ego_from_cam = make_se3(R_ego_from_cam, t_ego_from_cam)
        T_cam_from_ego = invert_se3(T_ego_from_cam)

        # --- ego pose (ego <-> global) for this sample_data timestamp ---
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        R_global_from_ego = quat_to_rotmat(ep["rotation"])
        t_global_from_ego = np.array(ep["translation"], dtype=np.float64)
        T_global_from_ego = make_se3(R_global_from_ego, t_global_from_ego)
        T_ego_from_global = invert_se3(T_global_from_ego)

        out[ch] = CameraCalibration(
            channel=ch,
            K=K,
            T_ego_from_cam=T_ego_from_cam,
            T_cam_from_ego=T_cam_from_ego,
            T_global_from_ego=T_global_from_ego,
            T_ego_from_global=T_ego_from_global,
        )

    return out
