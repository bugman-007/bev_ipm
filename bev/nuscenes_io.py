from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image

from .utils import (
    assert_dir_exists,
    assert_file_exists,
    quat_to_rotmat,
    make_se3,
    invert_se3,
)

CAM_CHANNELS_6 = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class CameraFrame:
    channel: str
    sample_data_token: str
    filename_rel: str
    filename_abs: str
    width: int
    height: int
    timestamp: int  # microseconds in nuScenes


@dataclass
class SampleSelection:
    sample_token: str
    scene_token: str
    scene_name: str
    timestamp: int  # microseconds
    cameras: Dict[str, CameraFrame]


def init_nuscenes(dataroot: str, version: str, verbose: bool = True) -> NuScenes:
    assert_dir_exists(dataroot, "nuScenes dataroot")
    return NuScenes(version=version, dataroot=dataroot, verbose=verbose)


def resolve_scene_token(nusc: NuScenes, scene_token_or_name: str) -> Tuple[str, str]:
    # Try token
    try:
        scene = nusc.get("scene", scene_token_or_name)
        return scene["token"], scene["name"]
    except Exception:
        pass

    # Try name (e.g., scene-0061)
    matches = [s for s in nusc.scene if s.get("name") == scene_token_or_name]
    if len(matches) == 1:
        return matches[0]["token"], matches[0]["name"]
    if len(matches) > 1:
        raise ValueError(f"Multiple scenes match name={scene_token_or_name}. Please pass a token.")
    raise ValueError(
        f"Could not resolve scene '{scene_token_or_name}'. "
        f"Pass a valid scene token or a name like 'scene-0061'."
    )


def _get_sample_by_scene_index(nusc: NuScenes, scene_token: str, index: int) -> dict:
    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]

    if index < 0:
        raise ValueError("--index must be >= 0")

    cur = nusc.get("sample", token)
    i = 0
    while i < index:
        nxt = cur["next"]
        if not nxt:
            raise IndexError(f"Scene ended before index={index}. Reached i={i} token={cur['token']}")
        cur = nusc.get("sample", nxt)
        i += 1
    return cur


def collect_6_camera_frames(nusc: NuScenes, sample: dict) -> Dict[str, CameraFrame]:
    data_map = sample["data"]
    missing = [ch for ch in CAM_CHANNELS_6 if ch not in data_map]
    if missing:
        raise KeyError(f"Sample {sample['token']} missing channels: {missing}")

    cameras: Dict[str, CameraFrame] = {}
    for ch in CAM_CHANNELS_6:
        sd_token = data_map[ch]
        sd = nusc.get("sample_data", sd_token)
        rel = sd["filename"]
        abs_path = os.path.join(nusc.dataroot, rel)
        assert_file_exists(abs_path, f"image for {ch}")

        with Image.open(abs_path) as im:
            w, h = im.size

        cameras[ch] = CameraFrame(
            channel=ch,
            sample_data_token=sd_token,
            filename_rel=rel,
            filename_abs=abs_path,
            width=w,
            height=h,
            timestamp=int(sd["timestamp"]),
        )
    return cameras


def select_sample(
    nusc: NuScenes,
    sample_token: Optional[str] = None,
    scene: Optional[str] = None,
    index: int = 0,
) -> SampleSelection:
    if sample_token:
        sample = nusc.get("sample", sample_token)
        scene_token = sample["scene_token"]
        scene_name = nusc.get("scene", scene_token)["name"]
    else:
        if not scene:
            raise ValueError("Provide --sample_token OR --scene (token or name).")
        scene_token, scene_name = resolve_scene_token(nusc, scene)
        sample = _get_sample_by_scene_index(nusc, scene_token, index)

    cameras = collect_6_camera_frames(nusc, sample)
    return SampleSelection(
        sample_token=sample["token"],
        scene_token=scene_token,
        scene_name=scene_name,
        timestamp=int(sample["timestamp"]),
        cameras=cameras,
    )


# ---------------- Step 2 ----------------

@dataclass
class CameraCalibration:
    channel: str
    K: np.ndarray                 # 3x3
    T_ego_from_cam: np.ndarray    # 4x4  (cam -> ego)
    T_cam_from_ego: np.ndarray    # 4x4  (ego -> cam)
    T_global_from_ego: np.ndarray # 4x4
    T_ego_from_global: np.ndarray # 4x4


def get_camera_calibration_for_sample(
    nusc: NuScenes,
    selection: SampleSelection
) -> Dict[str, CameraCalibration]:
    out: Dict[str, CameraCalibration] = {}

    for ch, cam in selection.cameras.items():
        sd = nusc.get("sample_data", cam.sample_data_token)

        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"], dtype=np.float64)

        R_ego_from_cam = quat_to_rotmat(cs["rotation"])
        t_ego_from_cam = np.array(cs["translation"], dtype=np.float64)
        T_ego_from_cam = make_se3(R_ego_from_cam, t_ego_from_cam)
        T_cam_from_ego = invert_se3(T_ego_from_cam)

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
