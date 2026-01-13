# nuscenes_io.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple, Union, List

import numpy as np
from nuscenes.nuscenes import NuScenes

try:
    from utils import quat_to_rotmat, make_se3, invert_se3, assert_dir_exists, assert_file_exists
except ImportError:
    from .utils import quat_to_rotmat, make_se3, invert_se3, assert_dir_exists, assert_file_exists


CAM_CHANNELS_6: List[str] = [
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
    timestamp: int  # microseconds


@dataclass
class SampleSelection:
    sample_token: str
    scene_token: str
    scene_name: str
    timestamp: int  # microseconds
    cameras: Dict[str, CameraFrame]
    
@dataclass(frozen=True)
class CameraModel:
    """
    Geometry for one nuScenes camera sample_data.

    NuScenes conventions:
      - calibrated_sensor rotation/translation: sensor pose relative to ego vehicle.
      - ego_pose rotation/translation: ego pose in global/world coordinates.
    """
    channel: str
    image_path: str
    timestamp: int

    K: np.ndarray  # (3,3)
    T_ego_from_cam: np.ndarray   # (4,4)
    T_cam_from_ego: np.ndarray   # (4,4)

    T_world_from_ego: np.ndarray # (4,4)
    T_ego_from_world: np.ndarray # (4,4)

    T_world_from_cam: np.ndarray # (4,4)
    T_cam_from_world: np.ndarray # (4,4)

def init_nuscenes(dataroot: str, version: str, verbose: bool = True) -> NuScenes:
    assert_dir_exists(dataroot, "nuScenes dataroot")
    return NuScenes(version=version, dataroot=os.path.normpath(dataroot), verbose=verbose)


def resolve_scene_token(nusc: NuScenes, scene_token_or_name: str) -> Tuple[str, str]:
    # Try token first
    try:
        scene = nusc.get("scene", scene_token_or_name)
        return scene["token"], scene["name"]
    except Exception:
        pass

    # Else interpret as scene name
    matches = [s for s in nusc.scene if s.get("name") == scene_token_or_name]
    if len(matches) == 1:
        return matches[0]["token"], matches[0]["name"]
    if len(matches) > 1:
        raise ValueError(f"Multiple scenes match name={scene_token_or_name}. Please pass a token.")
    raise ValueError(
        f"Could not resolve scene '{scene_token_or_name}'. "
        f"Pass a valid scene token or a name like 'scene-0061'."
    )


def sample_token_at_index(nusc: NuScenes, scene_token: str, index: int) -> str:
    if index < 0:
        raise ValueError("index must be >= 0")
    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]
    for _ in range(int(index)):
        s = nusc.get("sample", token)
        if not s["next"]:
            raise IndexError(f"Scene ended before index={index}.")
        token = s["next"]
    return token


def iter_scene_sample_tokens(
    nusc: NuScenes,
    scene_token: str,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
) -> Iterator[str]:
    """
    Yields sample tokens from a scene by index, supporting n*6 processing.

    start: first index (inclusive)
    end: last index (exclusive). If None, goes to end of scene.
    step: stride (>=1)
    """
    if step <= 0:
        raise ValueError("step must be >= 1")
    token = sample_token_at_index(nusc, scene_token, start)
    idx = start
    while True:
        if end is not None and idx >= end:
            break
        yield token
        s = nusc.get("sample", token)
        # advance step times
        nxt = token
        for _ in range(step):
            ss = nusc.get("sample", nxt)
            nxt = ss["next"]
            if not nxt:
                return
        token = nxt
        idx += step


def get_sample_camera_tokens(sample: Mapping, channels: List[str] = CAM_CHANNELS_6) -> Dict[str, str]:
    data_map = sample["data"]
    missing = [ch for ch in channels if ch not in data_map]
    if missing:
        raise KeyError(f"Sample {sample.get('token','?')} missing camera channels: {missing}")
    return {ch: data_map[ch] for ch in channels}

def get_sample_camera_sample_data_tokens(
    nusc: NuScenes,
    sample_token: str,
    channels: List[str] = CAM_CHANNELS_6,
) -> Dict[str, str]:
    """
    Convenience: from sample_token -> dict(channel -> sample_data_token).
    """
    sample = nusc.get("sample", sample_token)
    return get_sample_camera_tokens(sample, channels)

def get_image_path(nusc: NuScenes, sample_data_token: str) -> str:
    sd = nusc.get("sample_data", sample_data_token)
    rel = sd["filename"]
    abs_path = os.path.join(nusc.dataroot, rel)
    assert_file_exists(abs_path, f"image for sample_data={sample_data_token}")
    return abs_path


def collect_6_camera_frames(nusc: NuScenes, sample: Mapping) -> Dict[str, CameraFrame]:
    cams = get_sample_camera_tokens(sample, CAM_CHANNELS_6)
    out: Dict[str, CameraFrame] = {}
    for ch, sd_token in cams.items():
        sd = nusc.get("sample_data", sd_token)
        rel = sd["filename"]
        abs_path = os.path.join(nusc.dataroot, rel)
        assert_file_exists(abs_path, f"image for {ch}")
        out[ch] = CameraFrame(
            channel=ch,
            sample_data_token=sd_token,
            filename_rel=rel,
            filename_abs=abs_path,
            timestamp=int(sd["timestamp"]),
        )
    return out


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
            raise ValueError("Provide sample_token OR scene (token or name).")
        scene_token, scene_name = resolve_scene_token(nusc, scene)
        token = sample_token_at_index(nusc, scene_token, index)
        sample = nusc.get("sample", token)

    cameras = collect_6_camera_frames(nusc, sample)
    return SampleSelection(
        sample_token=sample["token"],
        scene_token=scene_token,
        scene_name=scene_name,
        timestamp=int(sample["timestamp"]),
        cameras=cameras,
    )


# ---------------- Calibration + poses ----------------

@dataclass
class CameraCalibration:
    channel: str
    K: np.ndarray              # 3x3
    T_ego_from_cam: np.ndarray # 4x4 (cam -> ego)
    T_cam_from_ego: np.ndarray # 4x4 (ego -> cam)


def get_camera_calibration_for_sample(
    nusc: NuScenes,
    selection_or_sd_tokens: Union[SampleSelection, Mapping[str, str]],
) -> Dict[str, CameraCalibration]:
    """
    Returns per-camera intrinsics + extrinsics.

    calibrated_sensor rotation/translation in nuScenes represent sensor pose in ego frame:
      p_ego = T_ego_from_cam * p_cam
    """
    calibs: Dict[str, CameraCalibration] = {}

    if isinstance(selection_or_sd_tokens, SampleSelection):
        items = [(ch, cam.sample_data_token) for ch, cam in selection_or_sd_tokens.cameras.items()]
    else:
        items = list(selection_or_sd_tokens.items())

    for ch, sd_token in items:
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        R_ego_from_cam = quat_to_rotmat(cs["rotation"])
        t_ego_from_cam = np.array(cs["translation"], dtype=np.float64)
        T_ego_from_cam = make_se3(R_ego_from_cam, t_ego_from_cam)
        T_cam_from_ego = invert_se3(T_ego_from_cam)

        K = np.array(cs["camera_intrinsic"], dtype=np.float64)

        calibs[ch] = CameraCalibration(
            channel=ch,
            K=K,
            T_ego_from_cam=T_ego_from_cam,
            T_cam_from_ego=T_cam_from_ego,
        )

    return calibs

@dataclass
class EgoPose:
    timestamp: int
    T_world_from_ego: np.ndarray  # 4x4 (ego -> world/global)
    T_ego_from_world: np.ndarray  # 4x4 (world -> ego)


def get_ego_pose_for_sample_data(nusc: NuScenes, sample_data_token: str) -> EgoPose:
    """
    ego_pose rotation/translation represent ego pose in global/world frame:
      p_world = T_world_from_ego * p_ego
    """
    sd = nusc.get("sample_data", sample_data_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])

    R_world_from_ego = quat_to_rotmat(ep["rotation"])
    t_world_from_ego = np.array(ep["translation"], dtype=np.float64)

    T_world_from_ego = make_se3(R_world_from_ego, t_world_from_ego)
    T_ego_from_world = invert_se3(T_world_from_ego)

    return EgoPose(
        timestamp=int(sd["timestamp"]),
        T_world_from_ego=T_world_from_ego,
        T_ego_from_world=T_ego_from_world,
    )


def iter_scene_samples(
    nusc: NuScenes,
    scene_token: str,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
) -> Iterator[Mapping]:
    """Convenience: yields full sample dicts for a scene by index."""
    for tok in iter_scene_sample_tokens(nusc, scene_token, start=start, end=end, step=step):
        yield nusc.get("sample", tok)


def get_camera_calibration_for_sample_data(nusc: NuScenes, sample_data_token: str) -> CameraCalibration:
    """Calibration for one camera sample_data token."""
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    R_ego_from_cam = quat_to_rotmat(cs["rotation"])
    t_ego_from_cam = np.array(cs["translation"], dtype=np.float64)
    T_ego_from_cam = make_se3(R_ego_from_cam, t_ego_from_cam)
    T_cam_from_ego = invert_se3(T_ego_from_cam)

    K = np.array(cs["camera_intrinsic"], dtype=np.float64)
    return CameraCalibration(
        channel=sd.get("channel", ""),
        K=K,
        T_ego_from_cam=T_ego_from_cam,
        T_cam_from_ego=T_cam_from_ego,
    )


def get_camera_model_for_sample_data(nusc: NuScenes, sample_data_token: str) -> CameraModel:
    """
    Returns camera intrinsics + full SE(3) chain for one camera sample_data.

    world_from_cam = world_from_ego @ ego_from_cam
    cam_from_world = inv(world_from_cam)
    """
    sd = nusc.get("sample_data", sample_data_token)
    if "CAM" not in sd.get("channel", ""):
        raise ValueError(f"sample_data_token is not a camera: {sample_data_token}")

    calib = get_camera_calibration_for_sample_data(nusc, sample_data_token)
    ep = get_ego_pose_for_sample_data(nusc, sample_data_token)

    T_world_from_cam = ep.T_world_from_ego @ calib.T_ego_from_cam
    T_cam_from_world = invert_se3(T_world_from_cam)

    image_path = os.path.join(nusc.dataroot, sd["filename"])
    channel = sd.get("channel", "")
    timestamp = int(sd["timestamp"])

    return CameraModel(
        channel=channel,
        image_path=image_path,
        timestamp=timestamp,
        K=calib.K,
        T_ego_from_cam=calib.T_ego_from_cam,
        T_cam_from_ego=calib.T_cam_from_ego,
        T_world_from_ego=ep.T_world_from_ego,
        T_ego_from_world=ep.T_ego_from_world,
        T_world_from_cam=T_world_from_cam,
        T_cam_from_world=T_cam_from_world,
    )


def get_camera_models_for_sample(
    nusc: NuScenes,
    sample_token: str,
    channels: Optional[List[str]] = None,
) -> Dict[str, CameraModel]:
    """
    For a sample_token, returns CameraModel for each requested channel.
    """
    sample = nusc.get("sample", sample_token)
    if channels is None:
        channels = list(sample["data"].keys())

    out: Dict[str, CameraModel] = {}
    for ch in channels:
        if ch not in sample["data"]:
            continue
        sd_token = sample["data"][ch]
        sd = nusc.get("sample_data", sd_token)
        if "CAM" not in sd["channel"]:
            continue
        out[ch] = get_camera_model_for_sample_data(nusc, sd_token)
    return out
