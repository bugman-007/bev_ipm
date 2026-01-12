from __future__ import annotations

import argparse
from typing import Optional

from bev.nuscenes_io import init_nuscenes, select_sample, CAM_CHANNELS_6
from bev.utils import info

import numpy as np
from bev.nuscenes_io import get_camera_calibration_for_sample


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


if __name__ == "__main__":
    main()
