from __future__ import annotations

import argparse
from typing import Optional

from bev.nuscenes_io import init_nuscenes, select_sample, CAM_CHANNELS_6
from bev.utils import info

import numpy as np
from bev.nuscenes_io import get_camera_calibration_for_sample
from bev.bev_grid import make_bev_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-1 BEV RGB projection prototype (Step 1: select sample + list 6 camera frames)."
    )
    p.add_argument("--dataroot", type=str, required=True, help="Path to nuScenes dataset root")
    p.add_argument("--version", type=str, required=True, help="nuScenes version, e.g. v1.0-mini or v1.0-trainval")

    # selection
    p.add_argument("--sample_token", type=str, default=None, help="Select a specific sample token")
    p.add_argument("--scene", type=str, default=None, help="Scene token or scene name like scene-0061")
    p.add_argument("--index", type=int, default=0, help="Sample index inside scene (0-based)")
    p.add_argument("--x_min", type=float, default=0.0)
    p.add_argument("--x_max", type=float, default=60.0)
    p.add_argument("--y_min", type=float, default=-30.0)
    p.add_argument("--y_max", type=float, default=30.0)
    p.add_argument("--res", type=float, default=0.05, help="meters per pixel")

    # output placeholder (used in later steps; kept for interface stability)
    p.add_argument("--out", type=str, default="output.png", help="Output image path (used in later steps)")

    p.add_argument("--quiet", action="store_true", help="Less verbose nuScenes init")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nusc = init_nuscenes(args.dataroot, args.version, verbose=(not args.quiet))

    selection = select_sample(
        nusc,
        sample_token=args.sample_token,
        scene=args.scene,
        index=args.index,
    )

    info("=== Step 1: Sample selection + 6 camera frames ===")
    info(f"Scene: {selection.scene_name}")
    info(f"Scene token: {selection.scene_token}")
    info(f"Sample token: {selection.sample_token}")
    info(f"Sample timestamp (us): {selection.timestamp}")
    info("")

    for ch in CAM_CHANNELS_6:
        cf = selection.cameras[ch]
        info(f"{ch}:")
        info(f"  sample_data_token: {cf.sample_data_token}")
        info(f"  timestamp (us):    {cf.timestamp}")
        info(f"  image:             {cf.filename_abs}")
        info(f"  size:              {cf.width} x {cf.height}")
        info("")

    info("Step 1 complete: all 6 camera images resolved.")

    calibs = get_camera_calibration_for_sample(nusc, selection)

    info("=== Step 2: Calibration & ego pose matrices ===")
    for ch in CAM_CHANNELS_6:
        c = calibs[ch]
        info(f"{ch}:")
        info("  K (intrinsics):")
        info(str(np.array2string(c.K, precision=3, suppress_small=True)))
        info("  T_cam_from_ego (ego -> cam):")
        info(str(np.array2string(c.T_cam_from_ego, precision=3, suppress_small=True)))
        info("  T_ego_from_cam (cam -> ego):")
        info(str(np.array2string(c.T_ego_from_cam, precision=3, suppress_small=True)))
        info("")
        
    grid = make_bev_grid(args.x_min, args.x_max, args.y_min, args.y_max, args.res)

    info("=== Step 3: BEV grid ===")
    info(f"Bounds: x[{grid.x_min}, {grid.x_max}] y[{grid.y_min}, {grid.y_max}] res={grid.res} m/px")
    info(f"Canvas: H={grid.H} W={grid.W}")

    # Print a few sanity points (meters)
    info(f"Top-left ego point (approx): x={grid.xs[0,0]:.3f}, y={grid.ys[0,0]:.3f}")
    info(f"Bottom-right ego point (approx): x={grid.xs[-1,-1]:.3f}, y={grid.ys[-1,-1]:.3f}")


if __name__ == "__main__":
    main()
