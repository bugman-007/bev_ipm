from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BEVGrid:
    """
    BEV grid definition in the nuScenes ego frame:
      - Ego: x forward, y left, z up (meters).
      - Image: rows increase downward, cols increase rightward.

    By default we choose a visualization-friendly convention:
      - Forward (x+) is UP in the BEV image  -> flip_x=True
      - Left (y+) is LEFT in the BEV image   -> flip_y=True

    This matches most BEV viewers (top = forward, left = left).
    Set flip_x/flip_y to False to keep the raw increasing order.
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    res: float
    flip_x: bool = True
    flip_y: bool = True

    @property
    def H(self) -> int:
        return int(np.ceil((self.x_max - self.x_min) / self.res))

    @property
    def W(self) -> int:
        return int(np.ceil((self.y_max - self.y_min) / self.res))

    def x_coords(self) -> np.ndarray:
        # cell centers
        xs = self.x_min + (np.arange(self.H, dtype=np.float32) + 0.5) * self.res
        if self.flip_x:
            xs = xs[::-1].copy()
        return xs

    def y_coords(self) -> np.ndarray:
        ys = self.y_min + (np.arange(self.W, dtype=np.float32) + 0.5) * self.res
        if self.flip_y:
            ys = ys[::-1].copy()
        return ys

    def points(self, z: float = 0.0) -> np.ndarray:
        """
        Returns (H, W, 3) float32 points on plane z in the grid's frame.
        (This grid frame can be ego OR world.)
        """
        xs = self.x_coords()
        ys = self.y_coords()
        X, Y = np.meshgrid(xs, ys, indexing="ij")  # (H, W)
        Z = np.full_like(X, float(z), dtype=np.float32)
        return np.stack([X, Y, Z], axis=-1)

    def ego_points(self, z: float = 0.0) -> np.ndarray:
        """
        Backward-compatible alias. Historically this grid was used in ego frame.
        """
        return self.points(z=z)
    
    def ego_xy_to_ij(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert ego-frame X (forward), Y (left) into BEV image indices (i,j).
        i corresponds to x axis, j corresponds to y axis.
        Applies flip_x/flip_y consistently with ego_points().
        """
        # continuous index in non-flipped coordinates
        i = (X - self.x_min) / self.res - 0.5
        j = (Y - self.y_min) / self.res - 0.5

        if self.flip_x:
            i = (self.H - 1) - i
        if self.flip_y:
            j = (self.W - 1) - j

        return i, j
