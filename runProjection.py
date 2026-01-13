# runProjection.py  (world-map mosaic runner)

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from nuscenes_io import (
    init_nuscenes,
    resolve_scene_token,
    CAM_CHANNELS_6,
    get_camera_calibration_for_sample,
    get_ego_pose_for_sample_data,
    iter_scene_samples,
)
from bev_grid import BEVGrid
from projection import warp_image_to_bev
from stitching import feather_weight
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", required=True, help="scene token or name like scene-0061")

    # Output
    ap.add_argument("--out_bev", default="world_bev.png")
    ap.add_argument("--out_wsum", default="world_wsum.png")

    # World grid / bounds
    ap.add_argument("--res", type=float, default=0.2, help="meters per pixel")
    ap.add_argument("--pad", type=float, default=60.0, help="padding around trajectory for auto-bounds (m)")
    ap.add_argument("--x_min", type=float, default=None)
    ap.add_argument("--x_max", type=float, default=None)
    ap.add_argument("--y_min", type=float, default=None)
    ap.add_argument("--y_max", type=float, default=None)

    # Frame range
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)

    # Geometry / blending
    ap.add_argument("--z_plane_rel_ego", type=float, default=0.0)
    ap.add_argument("--feather_px", type=int, default=80)
    ap.add_argument("--roi_vmin", type=float, default=None)
    ap.add_argument("--roi_vmax", type=float, default=None)

    args = ap.parse_args()

    nusc = init_nuscenes(args.dataroot, args.version, verbose=True)
    scene_token, scene_name = resolve_scene_token(nusc, args.scene)
    print(f"[INFO] scene={scene_name} token={scene_token}")

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

    cv2.imwrite(args.out_wsum, wsum_vis)
    print(f"[DONE] wrote {args.out_bev} and {args.out_wsum}")


if __name__ == "__main__":
    main()
