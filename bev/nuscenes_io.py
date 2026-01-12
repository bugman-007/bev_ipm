from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from nuscenes.nuscenes import NuScenes
from PIL import Image

from .utils import assert_dir_exists, assert_file_exists


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
    """
    Initialize NuScenes with a local dataset path.
    """
    assert_dir_exists(dataroot, "nuScenes dataroot")
    # NuScenes will validate metadata files under dataroot/version (depending on structure).
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    return nusc


def _get_sample_by_scene_index(nusc: NuScenes, scene_token: str, index: int) -> dict:
    """
    Walk the linked-list of samples in a scene until reaching 'index'.
    """
    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]

    if index < 0:
        raise ValueError("--index must be >= 0")

    cur = nusc.get("sample", token)
    i = 0
    while i < index:
        nxt = cur["next"]
        if nxt == "" or nxt is None:
            raise IndexError(
                f"Scene ended before index={index}. "
                f"Reached i={i} sample_token={cur['token']}"
            )
        cur = nusc.get("sample", nxt)
        i += 1

    return cur


def resolve_scene_token(
    nusc: NuScenes,
    scene_token_or_name: str,
) -> Tuple[str, str]:
    """
    Accept either:
      - a scene token (32-hex-like string)
      - or a scene name like 'scene-0061' (common in nuScenes)

    Returns: (scene_token, scene_name)
    """
    # Fast path: try as token
    try:
        scene = nusc.get("scene", scene_token_or_name)
        return scene["token"], scene["name"]
    except Exception:
        pass

    # Try as name
    matches = []
    for s in nusc.scene:
        if s.get("name") == scene_token_or_name:
            matches.append(s)

    if len(matches) == 1:
        return matches[0]["token"], matches[0]["name"]

    if len(matches) > 1:
        raise ValueError(f"Multiple scenes match name={scene_token_or_name}. Please pass a token.")
    raise ValueError(
        f"Could not resolve scene '{scene_token_or_name}'. "
        f"Pass a valid scene token or a name like 'scene-0061'."
    )


def collect_6_camera_frames(nusc: NuScenes, sample: dict) -> Dict[str, CameraFrame]:
    """
    For the selected sample, gather 6 camera sample_data + absolute file paths + image sizes.
    """
    sample_token = sample["token"]
    data_map = sample["data"]

    missing = [ch for ch in CAM_CHANNELS_6 if ch not in data_map]
    if missing:
        raise KeyError(
            f"Sample {sample_token} is missing camera channels: {missing}. "
            f"Available keys: {sorted(list(data_map.keys()))}"
        )

    cameras: Dict[str, CameraFrame] = {}
    # sample['timestamp'] exists and refers to sample timestamp (microseconds)
    # each sample_data also has its own timestamp (camera timestamp)
    for ch in CAM_CHANNELS_6:
        sd_token = data_map[ch]
        sd = nusc.get("sample_data", sd_token)
        rel = sd["filename"]
        abs_path = os.path.join(nusc.dataroot, rel)
        assert_file_exists(abs_path, f"image for {ch}")

        # Read dimensions (fast, safe)
        with Image.open(abs_path) as im:
            width, height = im.size

        cameras[ch] = CameraFrame(
            channel=ch,
            sample_data_token=sd_token,
            filename_rel=rel,
            filename_abs=abs_path,
            width=width,
            height=height,
            timestamp=int(sd["timestamp"]),
        )

    return cameras


def select_sample(
    nusc: NuScenes,
    sample_token: Optional[str] = None,
    scene: Optional[str] = None,
    index: int = 0,
) -> SampleSelection:
    """
    Select a sample either by sample_token, or by (scene token/name + index).
    Returns a structured object with 6 camera frames collected.

    This is Step 1 deliverable.
    """
    if sample_token:
        sample = nusc.get("sample", sample_token)
        scene_token = sample["scene_token"]
        scene_rec = nusc.get("scene", scene_token)
        scene_name = scene_rec["name"]
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
