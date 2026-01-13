from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import cv2


def feather_weight(valid_mask: np.ndarray, feather_px: int) -> np.ndarray:
    """
    Soft weight that decays toward invalid regions using distance transform.
    valid_mask: (H,W) bool
    returns: (H,W) float32 in [0,1]
    """
    if feather_px <= 0:
        return valid_mask.astype(np.float32)

    valid_u8 = valid_mask.astype(np.uint8)
    dist = cv2.distanceTransform(valid_u8, cv2.DIST_L2, 3)  # float32, px units
    w = np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32)
    return w


def stitch_warped(
    warped_bgr_by_cam: Dict[str, np.ndarray],
    valid_mask_by_cam: Dict[str, np.ndarray],
    feather_px: int = 80,
    fill_holes: bool = False,
    inpaint_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted blend of already-warped BEV images.

    Returns:
      stitched_bgr: (H,W,3) uint8
      wsum: (H,W) float32 total weight
    """
    cams = list(warped_bgr_by_cam.keys())
    if not cams:
        raise ValueError("No cameras provided.")

    H, W = warped_bgr_by_cam[cams[0]].shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    for cam in cams:
        img = warped_bgr_by_cam[cam]
        msk = valid_mask_by_cam[cam]
        if img.shape[:2] != (H, W) or msk.shape[:2] != (H, W):
            raise ValueError(f"Shape mismatch for {cam}: img {img.shape}, mask {msk.shape} != {(H,W)}")

        w = feather_weight(msk, feather_px)  # (H,W) float32
        acc += img.astype(np.float32) * w[..., None]
        wsum += w

    eps = 1e-6
    stitched = np.zeros((H, W, 3), dtype=np.uint8)
    ok = wsum > eps
    stitched[ok] = (acc[ok] / wsum[ok, None]).clip(0, 255).astype(np.uint8)

    # Visual-only (not “geometrically correct”) fill for unobserved regions
    if fill_holes and np.any(~ok):
        hole_mask = (~ok).astype(np.uint8) * 255
        stitched = cv2.inpaint(stitched, hole_mask, float(inpaint_radius), cv2.INPAINT_TELEA)

    return stitched, wsum


def stitch_weighted(
    warped_bgr_by_cam: Dict[str, np.ndarray],
    weight_by_cam: Dict[str, np.ndarray],
    feather_px: int = 80,
    weight_blur_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend warped BEV images using float weight maps (e.g., wsum from splat).
    This is the correct stitcher for forward-splat pipelines.

    Returns:
      stitched_bgr: (H,W,3) uint8
      wsum: (H,W) float32
    """
    cams = list(warped_bgr_by_cam.keys())
    if not cams:
        raise ValueError("No cameras provided.")

    H, W = warped_bgr_by_cam[cams[0]].shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    eps = 1e-6

    for cam in cams:
        img = warped_bgr_by_cam[cam]
        w = weight_by_cam[cam].astype(np.float32)

        if img.shape[:2] != (H, W) or w.shape[:2] != (H, W):
            raise ValueError(f"Shape mismatch for {cam}")

        # Optional seam feathering based on support region
        if feather_px > 0:
            support = (w > eps).astype(np.uint8)
            if np.any(support):
                dist = cv2.distanceTransform(support, cv2.DIST_L2, 3)
                feather = np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32)
                w = w * feather

        # Optional blur to fill sparse splat holes a bit
        if weight_blur_sigma and weight_blur_sigma > 0:
            w = cv2.GaussianBlur(w, (0, 0), sigmaX=float(weight_blur_sigma), sigmaY=float(weight_blur_sigma))

        acc += img.astype(np.float32) * w[..., None]
        wsum += w

    stitched = np.zeros((H, W, 3), dtype=np.uint8)
    ok = wsum > eps
    stitched[ok] = (acc[ok] / wsum[ok, None]).clip(0, 255).astype(np.uint8)
    return stitched, wsum


def wsum_to_vis(wsum: np.ndarray) -> np.ndarray:
    w = np.asarray(wsum, dtype=np.float32)
    vmax = float(np.max(w)) if w.size else 0.0
    if vmax <= 1e-8:
        return np.zeros(w.shape, dtype=np.uint8)
    vis = np.clip(w / vmax, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)