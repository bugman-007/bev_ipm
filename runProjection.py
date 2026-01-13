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
from projection import splat_image_to_bev_ground, warp_image_to_bev
from stitching import stitch_weighted, stitch_warped, wsum_to_vis


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

    p.add_argument("--method", choices=["warp", "splat"], default="warp",
                   help="warp=reverse mapping (dense IPM), splat=forward splat (sparse)")

    # Common
    p.add_argument("--z_plane", type=float, default=0.0, help="Ground plane height in ego frame")
    p.add_argument("--no_flip_axes", action="store_true", help="Keep legacy axes (x down, y right).")
    p.add_argument("--quiet", action="store_true")

    # Stitching / output
    p.add_argument("--feather_px", type=int, default=80)
    p.add_argument("--save_per_camera", action="store_true")
    p.add_argument("--save_wsum", action="store_true")
    p.add_argument("--fill_holes", action="store_true", help="Visual-only: inpaint wsum==0 pixels.")
    p.add_argument("--inpaint_radius", type=int, default=5, help="Inpaint radius for hole filling")

    # SPLAT options
    p.add_argument("--stride", type=int, default=1, help="Splat stride (1 = best, slower)")
    p.add_argument("--post_blur_sigma", type=float, default=2.0,
                   help="(splat only) normalized-convolution blur sigma to fill holes/banding")

    # WARP options (ROI gating in source image to reduce pinwheel artifacts)
    p.add_argument(
    "--roi_vmin_frac", type=float, default=0.35,
    help="(warp only) minimum v fraction to KEEP (0..1). "
         "Example 0.35 keeps bottom 65% of the image (removes horizon)."
    )
    p.add_argument(
        "--roi_vmax_frac", type=float, default=1.00,
        help="(warp only) maximum v fraction to KEEP (0..1). "
            "Example 0.95 removes the bottom 5%."
    )

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

    out_dir = args.out_dir or os.path.dirname(args.out)
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(args.out)) if out_dir else args.out

    if args.method == "warp":
        warped_by_cam: Dict[str, np.ndarray] = {}
        valid_by_cam: Dict[str, np.ndarray] = {}

        for cam, sd_token in cam_sd_tokens.items():
            data_path, _, _ = nusc.get_sample_data(sd_token)
            img = cv2.imread(data_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {data_path}")

            calib = calibs[cam]

            ih, iw = img.shape[:2]
            roi_vmin = float(args.roi_vmin_frac) * ih if args.roi_vmin_frac is not None else None
            roi_vmax = float(args.roi_vmax_frac) * ih if args.roi_vmax_frac is not None else None
            # if roi_vmin is not None and roi_vmax is not None and roi_vmin >= roi_vmax:
            #     raise ValueError(f"Invalid ROI: roi_vmin({roi_vmin}) >= roi_vmax({roi_vmax})")

            warped, valid = warp_image_to_bev(
                img,
                grid,
                calib.K,
                calib.T_cam_from_ego,
                z_plane=float(args.z_plane),
                roi_vmin=roi_vmin,
                roi_vmax=roi_vmax,
            )

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

    else:
        # Forward splat (kept for reference / debugging)
        warped_by_cam: Dict[str, np.ndarray] = {}
        weight_by_cam: Dict[str, np.ndarray] = {}

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
            weight_blur_sigma=0.0,            # weight blur alone doesn't fill holes
            post_blur_sigma=float(args.post_blur_sigma),  # normalized convolution fill
        )

        if args.fill_holes:
            eps = 1e-6
            ok = wsum > eps
            if np.any(~ok):
                hole_mask = (~ok).astype(np.uint8) * 255
                stitched = cv2.inpaint(stitched, hole_mask, float(args.inpaint_radius), cv2.INPAINT_TELEA)

        cv2.imwrite(out_path, stitched)

        if args.save_wsum:
            cv2.imwrite(os.path.join(out_dir, "wsum.png"), wsum_to_vis(wsum))

    print(f"Saved stitched BEV: {out_path}")
    if args.save_wsum:
        print(f"Saved wsum: {os.path.join(out_dir, 'wsum.png')}")


if __name__ == "__main__":
    main()
