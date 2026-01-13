# runProjection.py

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

# --- Robust import path setup (works on Windows/EC2 and non-package folders) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_BEV_DIR = os.path.join(_THIS_DIR, "bev")
if os.path.isdir(_BEV_DIR) and _BEV_DIR not in sys.path:
    sys.path.insert(0, _BEV_DIR)

from nuscenes_io import (
    CAM_CHANNELS_6,
    init_nuscenes,
    resolve_scene_token,
    select_sample,
    iter_scene_sample_tokens,
    get_camera_calibration_for_sample,
    get_camera_calibration_for_sample_data,
    get_ego_pose_for_sample_data,
)
from bev_grid import BEVGrid
from projection import warp_image_to_bev
from stitching import stitch_warped, wsum_to_vis, feather_weight
from utils import invert_se3


def _safe_imread(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def build_auto_bounds_world(
    nusc,
    scene_token: str,
    *,
    pad_m: float = 50.0,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
) -> Tuple[float, float, float, float, float]:
    """Auto-compute world XY bounds (+ median ego Z) from ego trajectory."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for tok in iter_scene_sample_tokens(nusc, scene_token, start=start, end=end, step=step):
        sample = nusc.get("sample", tok)
        # Use any camera to fetch ego_pose (pose token lives on sample_data)
        sd_token = sample["data"][CAM_CHANNELS_6[0]]
        ego_pose = get_ego_pose_for_sample_data(nusc, sd_token)
        xs.append(float(ego_pose.T_world_from_ego[0, 3]))
        ys.append(float(ego_pose.T_world_from_ego[1, 3]))
        zs.append(float(ego_pose.T_world_from_ego[2, 3]))

    if not xs:
        raise ValueError("No samples found to compute auto bounds.")

    x_min = min(xs) - float(pad_m)
    x_max = max(xs) + float(pad_m)
    y_min = min(ys) - float(pad_m)
    y_max = max(ys) + float(pad_m)
    z_med = float(np.median(np.array(zs, dtype=np.float64)))
    return x_min, x_max, y_min, y_max, z_med


def project_single_sample(
    nusc,
    *,
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
    """Project one nuScenes sample (6 cameras) into ego-frame BEV."""
    selection = select_sample(nusc, scene=scene, index=index)
    cam_sd_tokens = {ch: cam.sample_data_token for ch, cam in selection.cameras.items()}
    calibs = get_camera_calibration_for_sample(nusc, cam_sd_tokens)

    warped_by_cam: Dict[str, np.ndarray] = {}
    valid_by_cam: Dict[str, np.ndarray] = {}

    for ch, cam_frame in selection.cameras.items():
        img_bgr = _safe_imread(cam_frame.filename_abs)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {cam_frame.filename_abs}")

        calib = calibs[ch]
        warped, valid = warp_image_to_bev(
            img_bgr,
            grid,
            calib.K,
            T_cam_from_grid=calib.T_cam_from_ego,
            z_plane=float(z_plane),
        )
        warped_by_cam[ch] = warped
        valid_by_cam[ch] = valid

        if save_per_camera and out_dir:
            cv2.imwrite(os.path.join(out_dir, f"{ch}_bev.png"), warped)

    stitched, wsum = stitch_warped(
        warped_by_cam,
        valid_by_cam,
        feather_px=int(feather_px),
        fill_holes=bool(fill_holes),
        inpaint_radius=int(inpaint_radius),
    )
    return stitched, wsum


def project_sample_token_ego_bev(
    nusc,
    *,
    sample_token: str,
    grid: BEVGrid,
    z_plane: float = 0.0,
    feather_px: int = 80,
    fill_holes: bool = False,
    inpaint_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like project_single_sample, but takes a sample_token (for fast n×6 loops)."""
    sample = nusc.get("sample", sample_token)

    cam_sd_tokens: Dict[str, str] = {
        ch: sample["data"][ch] for ch in CAM_CHANNELS_6 if ch in sample["data"]
    }
    if not cam_sd_tokens:
        raise ValueError(f"Sample has no camera channels: {sample_token}")

    calibs = get_camera_calibration_for_sample(nusc, cam_sd_tokens)

    warped_by_cam: Dict[str, np.ndarray] = {}
    valid_by_cam: Dict[str, np.ndarray] = {}

    for ch, sd_token in cam_sd_tokens.items():
        sd = nusc.get("sample_data", sd_token)
        img_path = os.path.join(nusc.dataroot, sd["filename"])
        img_bgr = _safe_imread(img_path)
        if img_bgr is None:
            continue

        calib = calibs[ch]
        warped, valid = warp_image_to_bev(
            img_bgr,
            grid,
            calib.K,
            T_cam_from_grid=calib.T_cam_from_ego,
            z_plane=float(z_plane),
        )
        warped_by_cam[ch] = warped
        valid_by_cam[ch] = valid

    if not warped_by_cam:
        raise RuntimeError(f"No camera images could be loaded for sample: {sample_token}")

    stitched, wsum = stitch_warped(
        warped_by_cam,
        valid_by_cam,
        feather_px=int(feather_px),
        fill_holes=bool(fill_holes),
        inpaint_radius=int(inpaint_radius),
    )
    return stitched, wsum


def project_scene_ego_bev_sequence(
    nusc,
    *,
    scene_token: str,
    grid: BEVGrid,
    out_dir: str,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    z_plane: float = 0.0,
    feather_px: int = 80,
    fill_holes: bool = False,
    inpaint_radius: int = 5,
    save_wsum: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    idx = int(start)
    for tok in iter_scene_sample_tokens(nusc, scene_token, start=start, end=end, step=step):
        stitched, wsum = project_sample_token_ego_bev(
            nusc,
            sample_token=tok,
            grid=grid,
            z_plane=float(z_plane),
            feather_px=int(feather_px),
            fill_holes=bool(fill_holes),
            inpaint_radius=int(inpaint_radius),
        )

        out_path = os.path.join(out_dir, f"bev_{idx:05d}.png")
        cv2.imwrite(out_path, stitched)

        if save_wsum:
            wsum_path = os.path.join(out_dir, f"wsum_{idx:05d}.png")
            cv2.imwrite(wsum_path, wsum_to_vis(wsum))

        idx += int(step)


def accumulate_scene_world_bev(
    nusc,
    *,
    scene_token: str,
    grid: BEVGrid,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    z_plane_world: float = 0.0,
    feather_px: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate a fixed world-plane mosaic directly in (world X, world Y, z=z_plane_world)."""
    acc = np.zeros((grid.H, grid.W, 3), dtype=np.float32)
    wsum = np.zeros((grid.H, grid.W), dtype=np.float32)

    for tok in iter_scene_sample_tokens(nusc, scene_token, start=start, end=end, step=step):
        sample = nusc.get("sample", tok)

        for ch in CAM_CHANNELS_6:
            sd_token = sample["data"].get(ch)
            if not sd_token:
                continue

            sd = nusc.get("sample_data", sd_token)
            img_path = os.path.join(nusc.dataroot, sd["filename"])
            img_bgr = _safe_imread(img_path)
            if img_bgr is None:
                continue

            calib = get_camera_calibration_for_sample_data(nusc, sd_token)
            ego_pose = get_ego_pose_for_sample_data(nusc, sd_token)

            # cam -> ego -> world
            T_world_from_cam = ego_pose.T_world_from_ego @ calib.T_ego_from_cam
            T_cam_from_world = invert_se3(T_world_from_cam)

            warped, valid = warp_image_to_bev(
                img_bgr,
                grid,
                calib.K,
                T_cam_from_grid=T_cam_from_world,
                z_plane=float(z_plane_world),
            )

            w = feather_weight(valid, int(feather_px))
            acc += warped.astype(np.float32) * w[..., None]
            wsum += w

    out = np.zeros((grid.H, grid.W, 3), dtype=np.uint8)
    ok = wsum > 1e-6
    out[ok] = (acc[ok] / wsum[ok, None]).clip(0, 255).astype(np.uint8)
    return out, wsum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", required=True, help="scene token or name like scene-0061")

    # Common range controls (batch modes)
    ap.add_argument("--start", type=int, default=0, help="Start sample index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="End sample index (exclusive)")
    ap.add_argument("--step", type=int, default=1, help="Stride (>=1)")

    # Ego BEV (single or sequence)
    ap.add_argument("--index", type=int, default=None, help="If set: single-sample ego BEV")
    ap.add_argument("--out", default=None, help="Output path for single-sample ego BEV")
    ap.add_argument("--out_dir", default="", help="Output directory (sequence or per-camera outputs)")
    ap.add_argument("--save_per_camera", action="store_true")
    ap.add_argument("--save_wsum", action="store_true")

    # World mosaic mode
    ap.add_argument("--world", action="store_true", help="Accumulate into a fixed world-plane mosaic")
    ap.add_argument("--out_bev", default="world_bev.png")
    ap.add_argument("--out_wsum_file", default="world_wsum.png")
    ap.add_argument("--z_plane_world", type=float, default=None, help="World Z for mosaic plane")
    ap.add_argument(
        "--z_plane_rel_ego",
        type=float,
        default=0.0,
        help="If --z_plane_world is unset, use median(ego_z)+z_plane_rel_ego",
    )
    ap.add_argument("--pad", type=float, default=60.0, help="Padding for auto world bounds (m)")

    # Grid / bounds (ego or world)
    ap.add_argument("--res", type=float, default=0.2, help="meters per pixel")
    ap.add_argument("--x_min", type=float, default=None)
    ap.add_argument("--x_max", type=float, default=None)
    ap.add_argument("--y_min", type=float, default=None)
    ap.add_argument("--y_max", type=float, default=None)
    ap.add_argument("--no_flip_axes", action="store_true", help="Disable visualization flips")

    # Geometry / blending
    ap.add_argument("--z_plane", type=float, default=0.0, help="Plane Z in grid frame")
    ap.add_argument("--feather_px", type=int, default=80)
    ap.add_argument("--fill_holes", action="store_true")
    ap.add_argument("--inpaint_radius", type=int, default=5)

    args = ap.parse_args()

    nusc = init_nuscenes(args.dataroot, args.version, verbose=True)
    scene_token, scene_name = resolve_scene_token(nusc, args.scene)
    print(f"[INFO] scene={scene_name} token={scene_token}")

    # --- World mosaic mode ---
    if args.world:
        if args.x_min is None or args.x_max is None or args.y_min is None or args.y_max is None:
            x_min, x_max, y_min, y_max, z_med = build_auto_bounds_world(
                nusc,
                scene_token,
                pad_m=float(args.pad),
                start=int(args.start),
                end=args.end,
                step=int(args.step),
            )
        else:
            x_min, x_max, y_min, y_max = float(args.x_min), float(args.x_max), float(args.y_min), float(args.y_max)
            # Still estimate median z (needed if z_plane_world unset)
            _, _, _, _, z_med = build_auto_bounds_world(
                nusc,
                scene_token,
                pad_m=0.0,
                start=int(args.start),
                end=args.end,
                step=int(args.step),
            )

        z_plane_world = float(args.z_plane_world) if args.z_plane_world is not None else (z_med + float(args.z_plane_rel_ego))

        print(
            f"[INFO] world mosaic: x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}] res={args.res} z={z_plane_world:.2f}"
        )

        # Note: this grid uses rows=x and cols=y (image-like). Orientation can be flipped for visualization.
        grid = BEVGrid(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            res=float(args.res),
            flip_x=False,
            flip_y=False,
        )

        stitched, wsum = accumulate_scene_world_bev(
            nusc,
            scene_token=scene_token,
            grid=grid,
            start=int(args.start),
            end=args.end,
            step=int(args.step),
            z_plane_world=z_plane_world,
            feather_px=int(args.feather_px),
        )

        cv2.imwrite(args.out_bev, stitched)
        cv2.imwrite(args.out_wsum_file, wsum_to_vis(wsum))
        print(f"[DONE] wrote {args.out_bev} and {args.out_wsum_file}")
        return

    # --- Ego BEV mode ---
    x_min = float(args.x_min) if args.x_min is not None else -30.0
    x_max = float(args.x_max) if args.x_max is not None else 60.0
    y_min = float(args.y_min) if args.y_min is not None else -30.0
    y_max = float(args.y_max) if args.y_max is not None else 30.0

    grid = BEVGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=float(args.res),
        flip_x=not bool(args.no_flip_axes),
        flip_y=not bool(args.no_flip_axes),
    )

    # Single-sample
    if args.index is not None:
        out_dir = args.out_dir or (os.path.dirname(args.out) if args.out else "")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out_path = args.out or (os.path.join(out_dir, "stitched_bev.png") if out_dir else "stitched_bev.png")
        stitched, wsum = project_single_sample(
            nusc,
            scene=args.scene,
            index=int(args.index),
            grid=grid,
            z_plane=float(args.z_plane),
            feather_px=int(args.feather_px),
            fill_holes=bool(args.fill_holes),
            inpaint_radius=int(args.inpaint_radius),
            save_per_camera=bool(args.save_per_camera),
            out_dir=out_dir,
        )

        cv2.imwrite(out_path, stitched)
        print(f"[DONE] wrote {out_path}")
        if args.save_wsum:
            wsum_path = os.path.join(out_dir, "wsum.png") if out_dir else "wsum.png"
            cv2.imwrite(wsum_path, wsum_to_vis(wsum))
            print(f"[DONE] wrote {wsum_path}")
        return

    # Sequence n×6
    out_dir = args.out_dir or "bev_seq"
    print(
        f"[INFO] sequence ego BEV: out_dir={out_dir} start={args.start} end={args.end} step={args.step}"
    )
    project_scene_ego_bev_sequence(
        nusc,
        scene_token=scene_token,
        grid=grid,
        out_dir=out_dir,
        start=int(args.start),
        end=args.end,
        step=int(args.step),
        z_plane=float(args.z_plane),
        feather_px=int(args.feather_px),
        fill_holes=bool(args.fill_holes),
        inpaint_radius=int(args.inpaint_radius),
        save_wsum=bool(args.save_wsum),
    )
    print(f"[DONE] wrote sequence to {out_dir}")


if __name__ == "__main__":
    main()
