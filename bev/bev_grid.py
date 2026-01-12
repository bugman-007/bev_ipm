from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class BEVGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    res: float
    H: int
    W: int
    xs: np.ndarray      # (H, W) ego-x in meters
    ys: np.ndarray      # (H, W) ego-y in meters
    pts_ego: np.ndarray # (H, W, 3) (x,y,0)

def make_bev_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: float
) -> BEVGrid:
    if res <= 0:
        raise ValueError("res must be > 0")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bounds")

    # H corresponds to x (forward), W corresponds to y (left/right)
    H = int(np.ceil((x_max - x_min) / res))
    W = int(np.ceil((y_max - y_min) / res))

    # Pixel centers: x increases down the image (rows), y increases right (cols)
    x_coords = x_min + (np.arange(H) + 0.5) * res
    y_coords = y_min + (np.arange(W) + 0.5) * res

    xs, ys = np.meshgrid(x_coords, y_coords, indexing="ij")  # (H,W)

    pts_ego = np.stack([xs, ys, np.zeros_like(xs)], axis=-1).astype(np.float64)  # (H,W,3)

    return BEVGrid(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, res=res,
        H=H, W=W, xs=xs, ys=ys, pts_ego=pts_ego
    )
