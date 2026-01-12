# runProjection.py  (TOP OF FILE - replace from line 1 down through the imports)
import os
import sys

# Ensure imports work no matter where you run this script from (Windows/Linux/EC2).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import argparse
import numpy as np
import cv2

from bev_grid import BEVGrid
from nuscenes_io import init_nuscenes, get_scene_sample, get_camera_sample_data, get_camera_calibrations
from projection import project_points_ego_to_image, sample_image_bilinear
from stitching import stitch_bev_images
from utils import ensure_parent_dir



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
    p.add_argument("--fill_holes", action="store_true", help="Visual-only: inpaint wsum==0 pixels.")
    p.add_argument("--inpaint_radius", type=int, default=5)
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
    
    if not args.quiet:
        print("=== BEV grid ===")
        print(f"Bounds: x[{args.x_min}, {args.x_max}] y[{args.y_min}, {args.y_max}] res={args.res} m/px")
        print(f"Canvas: H={grid.H} W={grid.W}")
        print(f"flip_axes: {'NO' if args.no_flip_axes else 'YES'} (forward is {'down' if args.no_flip_axes else 'up'})")
        print(f"Top-left ego point (approx): x={grid.X[0,0]:.3f}, y={grid.Y[0,0]:.3f}")
        print(f"Bottom-right ego point (approx): x={grid.X[-1,-1]:.3f}, y={grid.Y[-1,-1]:.3f}")


    cam_sd_tokens: Dict[str, str] = {cam: sample["data"][cam] for cam in CAMERAS if cam in sample["data"]}
    if not cam_sd_tokens:
        raise RuntimeError("No camera sample_data tokens found for this sample.")

    calibs = get_camera_calibration_for_sample(nusc, cam_sd_tokens)

    warped_by_cam: Dict[str, any] = {}
    valid_by_cam: Dict[str, any] = {}

    out_dir = args.out_dir or os.path.dirname(args.out)
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(args.out)) if out_dir else args.out

    for cam, sd_token in cam_sd_tokens.items():
        data_path, _, _ = nusc.get_sample_data(sd_token)
        img = cv2.imread(data_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {data_path}")

        calib = calibs[cam]
        warped, valid = warp_image_to_bev(img, grid, calib.K, calib.T_cam_from_ego)

        warped_by_cam[cam] = warped
        valid_by_cam[cam] = valid

        if args.save_per_camera:
            cv2.imwrite(os.path.join(out_dir, f"{cam}_bev.png"), warped)

    stitched, wsum = stitch_warped(
        warped_by_cam,
        valid_by_cam,
        feather_px=args.feather_px,
        fill_holes=args.fill_holes,
        inpaint_radius=args.inpaint_radius,
    )

    cv2.imwrite(out_path, stitched)

    if args.save_wsum:
        cv2.imwrite(os.path.join(out_dir, "wsum.png"), wsum_to_vis(wsum))

    print(f"Saved stitched BEV: {out_path}")
    if args.save_wsum:
        print(f"Saved wsum: {os.path.join(out_dir, 'wsum.png')}")


if __name__ == "__main__":
    main()
