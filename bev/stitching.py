from __future__ import annotations
import numpy as np
import cv2

def feather_weight(mask: np.ndarray, ksize: int = 51) -> np.ndarray:
    """
    mask: bool (H,W)
    returns float32 (H,W) blurred weight in [0,1]
    """
    w = mask.astype(np.float32)
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        return w
    w = cv2.GaussianBlur(w, (ksize, ksize), 0)
    return w

def blend_bev(images: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """
    images: list of uint8 (H,W,3) or float32
    weights: list of float32 (H,W)
    returns uint8 (H,W,3)
    """
    assert len(images) == len(weights) and len(images) > 0
    H, W = weights[0].shape

    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    for img, w in zip(images, weights):
        if img.dtype != np.float32:
            img_f = img.astype(np.float32)
        else:
            img_f = img
        acc += img_f * w[..., None]
        wsum += w

    eps = 1e-6
    out = acc / (wsum[..., None] + eps)

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    out_u8[wsum <= eps] = 0
    return out_u8
