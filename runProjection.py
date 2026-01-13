from __future__ import annotations

import os
import sys
import argparse
from typing import List, Dict

import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes

# --- Robust import path setup ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_BEV_DIR = os.path.join(_THIS_DIR, "bev")
if os.path.isdir(_BEV_DIR) and _BEV_DIR not in sys.path:
    sys.path.insert(0, _BEV_DIR)

from bev_grid import BEVGrid
from nuscenes_io import get_camera_calibration_for_sample
from projection import splat_image_to_bev_ground
from stitching import stitch_weighted, wsum_to_vis


CAMERAS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def _find_scene_by_name(nusc: NuScenes, scene_name: str) -> Dict:
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene
    raise ValueError(f"Scene '{scene_name}' not found in {nusc.version}.")


def _sample_token_at_index(nusc: NuScenes, scene_rec: Dict, index: int) -> str:
    token = scene_rec["first_sample_token"]
    for _ in range(int(index)):
        sample = nusc.get("sample", token)
        nxt = sample["next"]
        if nxt == "":
            raise IndexError(f"Scene ended before index={index}.")
        token = nxt
    return token


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", required=True, help="e.g. scene-0061")
    p.add_argument("--index", type=int, default=0, help="sample index within the scene")
    p.add_argument("--x_min", type=float, default=-30.0)
    p.add_argument("--x_max", type=float, default=60.0)
    p.add_argument("--y_min", type=float, default=-30.0)
    p.add_argument("--y_max", type=float, default=30.0)
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--out", default="stitched_bev.png")
    p.add_argument("--out_dir", default="", help="If set, saves all outputs there.")
    p.add_argument("--feather_px", type=int, default=80)
    p.add_argument("--save_per_camera", action="store_true")
    p.add_argument("--save_wsum", action="store_true")
    p.add_argument("--stride", type=int, default=2, help="Splat stride (1 = best, slower)")
    p.add_argument("--z_plane", type=float, default=0.0, help="Ground plane height in ego frame")
    p.add_argument("--no_flip_axes", action="store_true", help="Keep legacy axes (x down, y right).")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    dataroot = os.path.normpath(args.dataroot)
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=not args.quiet)

    scene_rec = _find_scene_by_name(nusc, args.scene)
    sample_token = _sample_token_at_index(nusc, scene_rec, args.index)
    sample = nusc.get("sample", sample_token)

    grid = BEVGrid(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        res=args.res,
        flip_x=not args.no_flip_axes,
        flip_y=not args.no_flip_axes,
    )

    cam_sd_tokens: Dict[str, str] = {cam: sample["data"][cam] for cam in CAMERAS if cam in sample["data"]}
    if not cam_sd_tokens:
        raise RuntimeError("No camera sample_data tokens found for this sample.")

    calibs = get_camera_calibration_for_sample(nusc, cam_sd_tokens)

    warped_by_cam: Dict[str, np.ndarray] = {}
    weight_by_cam: Dict[str, np.ndarray] = {}

    out_dir = args.out_dir or os.path.dirname(args.out)
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(args.out)) if out_dir else args.out

    for cam, sd_token in cam_sd_tokens.items():
        data_path, _, _ = nusc.get_sample_data(sd_token)
        img = cv2.imread(data_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {data_path}")

        calib = calibs[cam]

        warped, wsum_cam = splat_image_to_bev_ground(
            img,
            grid,
            calib.K,
            calib.T_ego_from_cam,
            z_plane=float(args.z_plane),
            stride=int(args.stride),
        )

        warped_by_cam[cam] = warped
        weight_by_cam[cam] = wsum_cam

        if args.save_per_camera:
            cv2.imwrite(os.path.join(out_dir, f"{cam}_bev.png"), warped)

    stitched, wsum = stitch_weighted(
        warped_by_cam,
        weight_by_cam,
        feather_px=args.feather_px,
        weight_blur_sigma=1.0,  # try 0.0 / 1.0 / 2.0
    )

    cv2.imwrite(out_path, stitched)

    if args.save_wsum:
        cv2.imwrite(os.path.join(out_dir, "wsum.png"), wsum_to_vis(wsum))

    print(f"Saved stitched BEV: {out_path}")
    if args.save_wsum:
        print(f"Saved wsum: {os.path.join(out_dir, 'wsum.png')}")


if __name__ == "__main__":
    main()
