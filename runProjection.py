# runProjection.py  (world-map mosaic runner)

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

# --- Robust import path setup (fixes ModuleNotFoundError on Windows/EC2) ---
# Ensure Python can import modules that live either:
#   1) next to runProjection.py   (./bev_grid.py)
#   2) inside a subfolder         (./bev/bev_grid.py)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Always allow importing from the script directory
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Also allow importing from ./bev even if it's not a Python package (no __init__.py)
_BEV_DIR = os.path.join(_THIS_DIR, "bev")
if os.path.isdir(_BEV_DIR) and _BEV_DIR not in sys.path:
    sys.path.insert(0, _BEV_DIR)

from nuscenes_io import (
    init_nuscenes,
    resolve_scene_token,
    CAM_CHANNELS_6,
    get_camera_calibration_for_sample,
    get_ego_pose_for_sample_data,
    iter_scene_samples,
    select_sample,
)
from bev_grid import BEVGrid
from projection import warp_image_to_bev
from stitching import feather_weight, stitch_warped, wsum_to_vis
from utils import invert_se3


def build_auto_bounds_world(nusc, scene_token: str, pad_m: float = 50.0) -> Tuple[float, float, float, float]:
    """
    Auto-compute world XY bounds from the ego trajectory in the scene.
    """
    xs, ys = [], []
    for sample in iter_scene_samples(nusc, scene_token):
        # Use CAM_FRONT sample_data to grab ego_pose (any camera works)
        sd_token = sample["data"][CAM_CHANNELS_6[0]]
        ego_pose = get_ego_pose_for_sample_data(nusc, sd_token)
        xs.append(float(ego_pose.T_world_from_ego[0, 3]))
        ys.append(float(ego_pose.T_world_from_ego[1, 3]))

    x_min = min(xs) - pad_m
    x_max = max(xs) + pad_m
    y_min = min(ys) - pad_m
    y_max = max(ys) + pad_m
    return x_min, x_max, y_min, y_max


def accumulate_scene_world_bev(
    nusc,
    scene_token: str,
    grid: BEVGrid,
    start_index: int = 0,
    end_index: Optional[int] = None,
    step: int = 1,
    z_plane_rel_ego: float = 0.0,
    feather_px: int = 80,
    roi_vmin: Optional[float] = None,
    roi_vmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate warped BEV from many frames into a world-frame mosaic.

    CRITICAL FIX:
      Use ONE constant world Z-plane for the whole scene.
      If z changes per-frame, projections won't align and the mosaic smears.
    """
    acc = np.zeros((grid.H, grid.W, 3), dtype=np.float32)
    wsum = np.zeros((grid.H, grid.W), dtype=np.float32)

    # --- pick a single world ground plane Z for the whole scene ---
    # Use the first valid sample's ego Z as reference (more stable than per-frame).
    z_plane_world0: Optional[float] = None

    i = 0
    for sample in iter_scene_samples(nusc, scene_token):
        if i < start_index:
            i += 1
            continue
        if end_index is not None and i > end_index:
            break
        if (i - start_index) % step != 0:
            i += 1
            continue

        sd_tokens = {ch: sample["data"][ch] for ch in CAM_CHANNELS_6}
        calibs = get_camera_calibration_for_sample(nusc, sd_tokens)

        # Initialize constant plane once (first processed frame)
        if z_plane_world0 is None:
            sd_token0 = sd_tokens[CAM_CHANNELS_6[0]]
            ego_pose0 = get_ego_pose_for_sample_data(nusc, sd_token0)
            z_plane_world0 = float(ego_pose0.T_world_from_ego[2, 3]) + float(z_plane_rel_ego)

        for ch in CAM_CHANNELS_6:
            sd_token = sd_tokens[ch]
            calib = calibs[ch]

            sd = nusc.get("sample_data", sd_token)
            img_path = os.path.join(nusc.dataroot, sd["filename"])
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            ih, iw = img_bgr.shape[:2]
            # roi_vmin = args.roi_vmin_frac * ih if args.roi_vmin_frac is not None else None
            # roi_vmax = args.roi_vmax_frac * ih if args.roi_vmax_frac is not None else None

            ego_pose = get_ego_pose_for_sample_data(nusc, sd_token)
            T_world_from_ego = ego_pose.T_world_from_ego

            # cam -> world
            T_world_from_cam = T_world_from_ego @ calib.T_ego_from_cam
            # world -> cam
            T_cam_from_world = invert_se3(T_world_from_cam)

            warped, valid = warp_image_to_bev(
                img_bgr=img_bgr,
                grid=grid,
                K=calib.K,
                T_cam_from_ego=T_cam_from_world,  # “grid frame → cam frame” (world→cam)
                z_plane=z_plane_world0,           # <-- constant plane across frames
                roi_vmin=roi_vmin,
                roi_vmax=roi_vmax,
            )

            w = feather_weight(valid, feather_px)
            acc += warped.astype(np.float32) * w[..., None]
            wsum += w

        i += 1

    out = np.zeros((grid.H, grid.W, 3), dtype=np.uint8)
    ok = wsum > 1e-6
    out[ok] = (acc[ok] / wsum[ok, None]).clip(0, 255).astype(np.uint8)
    return out, wsum


def project_single_sample(
    nusc,
    scene: str,
    index: int,
    grid: BEVGrid,
    z_plane: float = 0.0,
    feather_px: int = 80,
    fill_holes: bool = False,
    inpaint_radius: int = 5,
    save_per_camera: bool = False,
    out_dir: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a single sample's 6 cameras into ego-frame BEV.
    """
    selection = select_sample(nusc, scene=scene, index=index)
    sample = nusc.get("sample", selection.sample_token)
    
    cam_sd_tokens = {ch: cam.sample_data_token for ch, cam in selection.cameras.items()}
    calibs = get_camera_calibration_for_sample(nusc, cam_sd_tokens)
    
    warped_by_cam: Dict[str, np.ndarray] = {}
    valid_by_cam: Dict[str, np.ndarray] = {}
    
    for ch, cam_frame in selection.cameras.items():
        img_bgr = cv2.imread(cam_frame.filename_abs, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {cam_frame.filename_abs}")
        
        calib = calibs[ch]
        warped, valid = warp_image_to_bev(
            img_bgr=img_bgr,
            grid=grid,
            K=calib.K,
            T_cam_from_ego=calib.T_cam_from_ego,
            z_plane=z_plane,
        )
        
        warped_by_cam[ch] = warped
        valid_by_cam[ch] = valid
        
        if save_per_camera and out_dir:
            cv2.imwrite(os.path.join(out_dir, f"{ch}_bev.png"), warped)
    
    stitched, wsum = stitch_warped(
        warped_by_cam,
        valid_by_cam,
        feather_px=feather_px,
        fill_holes=fill_holes,
        inpaint_radius=inpaint_radius,
    )
    
    return stitched, wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", required=True, help="scene token or name like scene-0061")

    # Single sample mode
    ap.add_argument("--index", type=int, default=None, help="sample index within scene (enables single-sample mode)")
    
    # Output (single sample mode)
    ap.add_argument("--out", default=None, help="output BEV image (single-sample mode)")
    ap.add_argument("--out_dir", default="", help="output directory for single-sample mode")
    ap.add_argument("--save_per_camera", action="store_true", help="save per-camera BEV images")
    ap.add_argument("--save_wsum", action="store_true", help="save weight sum visualization")
    
    # Output (world mosaic mode)
    ap.add_argument("--out_bev", default="world_bev.png")
    ap.add_argument("--out_wsum_file", default="world_wsum.png")

    # Grid / bounds
    ap.add_argument("--res", type=float, default=0.2, help="meters per pixel")
    ap.add_argument("--pad", type=float, default=60.0, help="padding around trajectory for auto-bounds (m)")
    ap.add_argument("--x_min", type=float, default=None)
    ap.add_argument("--x_max", type=float, default=None)
    ap.add_argument("--y_min", type=float, default=None)
    ap.add_argument("--y_max", type=float, default=None)

    # Frame range (world mosaic mode)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)

    # Geometry / blending
    ap.add_argument("--z_plane", type=float, default=0.0, help="ground plane height in ego frame (single-sample mode)")
    ap.add_argument("--z_plane_rel_ego", type=float, default=0.0, help="ground plane relative to ego (world mosaic mode)")
    ap.add_argument("--feather_px", type=int, default=80)
    ap.add_argument("--fill_holes", action="store_true", help="Visual-only: inpaint wsum==0 pixels (single-sample mode)")
    ap.add_argument("--inpaint_radius", type=int, default=5, help="Inpaint radius for hole filling")
    ap.add_argument("--roi_vmin", type=float, default=None)
    ap.add_argument("--roi_vmax", type=float, default=None)
    ap.add_argument("--no_flip_axes", action="store_true", help="Keep legacy axes (x down, y right)")

    args = ap.parse_args()

    nusc = init_nuscenes(args.dataroot, args.version, verbose=True)
    scene_token, scene_name = resolve_scene_token(nusc, args.scene)
    print(f"[INFO] scene={scene_name} token={scene_token}")

    # Single-sample mode
    if args.index is not None:
        # Use provided bounds or defaults for single-sample
        x_min = args.x_min if args.x_min is not None else -30.0
        x_max = args.x_max if args.x_max is not None else 60.0
        y_min = args.y_min if args.y_min is not None else -30.0
        y_max = args.y_max if args.y_max is not None else 30.0
        
        print(f"[INFO] single-sample mode: index={args.index}")
        print(f"[INFO] ego-frame bounds: x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}] res={args.res}")
        
        grid = BEVGrid(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            res=float(args.res),
            flip_x=not args.no_flip_axes,
            flip_y=not args.no_flip_axes,
        )
        
        # Determine output directory
        if args.out_dir:
            out_dir = args.out_dir
        elif args.out:
            out_dir = os.path.dirname(args.out) or ""
        else:
            out_dir = ""
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        # Determine output path
        if args.out:
            out_path = args.out
            # If out_dir is set but out_path doesn't have a directory, put it in out_dir
            if out_dir and not os.path.dirname(out_path):
                out_path = os.path.join(out_dir, os.path.basename(out_path))
        else:
            out_path = os.path.join(out_dir, "stitched_bev.png") if out_dir else "stitched_bev.png"
        
        stitched, wsum = project_single_sample(
            nusc=nusc,
            scene=args.scene,
            index=args.index,
            grid=grid,
            z_plane=args.z_plane,
            feather_px=args.feather_px,
            fill_holes=args.fill_holes,
            inpaint_radius=args.inpaint_radius,
            save_per_camera=args.save_per_camera,
            out_dir=out_dir,
        )
        
        cv2.imwrite(out_path, stitched)
        print(f"Saved stitched BEV: {out_path}")
        
        if args.save_wsum:
            wsum_path = os.path.join(out_dir, "wsum.png") if out_dir else "wsum.png"
            cv2.imwrite(wsum_path, wsum_to_vis(wsum))
            print(f"Saved wsum: {wsum_path}")
        
        return
    
    # World mosaic mode
    # Auto-bounds unless user provides explicit bounds
    if args.x_min is None or args.x_max is None or args.y_min is None or args.y_max is None:
        x_min, x_max, y_min, y_max = build_auto_bounds_world(nusc, scene_token, pad_m=float(args.pad))
    else:
        x_min, x_max, y_min, y_max = args.x_min, args.x_max, args.y_min, args.y_max

    print(f"[INFO] world bounds: x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}] res={args.res}")

    grid = BEVGrid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, res=float(args.res))

    stitched, wsum = accumulate_scene_world_bev(
        nusc=nusc,
        scene_token=scene_token,
        grid=grid,
        start_index=args.start,
        end_index=args.end,
        step=args.step,
        z_plane_rel_ego=args.z_plane_rel_ego,
        feather_px=args.feather_px,
        roi_vmin=args.roi_vmin,
        roi_vmax=args.roi_vmax,
    )

    cv2.imwrite(args.out_bev, stitched)
    m = float(np.max(wsum))
    if m <= 1e-6:
        wsum_vis = np.zeros_like(wsum, dtype=np.uint8)
    else:
        wsum_vis = (np.clip(wsum / m, 0.0, 1.0) * 255.0).astype(np.uint8)

    cv2.imwrite(args.out_wsum_file, wsum_vis)
    print(f"[DONE] wrote {args.out_bev} and {args.out_wsum_file}")


if __name__ == "__main__":
    main()
