# roi/roi_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RoiWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    def width(self) -> int:
        return self.x1 - self.x0

    def height(self) -> int:
        return self.y1 - self.y0


class RoiManager:
    def __init__(self, cfg: Dict[str, Any]):
        r = cfg.get("roi", {})
        self.base_size = int(r.get("base_size", 384))
        self.min_size = int(r.get("min_size", 256))
        self.max_size = int(r.get("max_size", 640))

        # arena rect (optional)
        arena = cfg.get("arena", {})
        mr = arena.get("manual_rect", {})
        if arena.get("enabled", False) and mr.get("enabled", False):
            self.arena_rect: Optional[Tuple[int, int, int, int]] = (
                int(mr["x0"]), int(mr["y0"]), int(mr["x1"]), int(mr["y1"])
            )
        else:
            self.arena_rect = None

    def make_roi(self, cx: float, cy: float, frame_size: Tuple[int, int],
                 expansion_factor: float = 1.0) -> RoiWindow:
        """
        Kalman 예측 중심 (cx, cy) 기반 ROI 생성.

        Args:
            cx, cy: 중심 좌표
            frame_size: (W, H)
            expansion_factor: ROI 확장 배율 (1.0 = base_size, 1.5 = 1.5배 확장)
                             UNCERTAIN/OCCLUDED 상태에서 재획득을 위해 확장
        """
        W, H = frame_size
        expanded_size = int(self.base_size * expansion_factor)
        size = max(self.min_size, min(expanded_size, self.max_size))
        half = size // 2

        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x1 = x0 + size
        y1 = y0 + size

        # frame boundary clamp
        if x0 < 0:
            x1 -= x0; x0 = 0
        if y0 < 0:
            y1 -= y0; y0 = 0
        if x1 > W:
            x0 -= (x1 - W); x1 = W
        if y1 > H:
            y0 -= (y1 - H); y1 = H
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)

        # arena clamp (선택)
        if self.arena_rect is not None:
            ax0, ay0, ax1, ay1 = self.arena_rect
            x0 = max(x0, ax0)
            y0 = max(y0, ay0)
            x1 = min(x1, ax1)
            y1 = min(y1, ay1)

        return RoiWindow(x0=x0, y0=y0, x1=x1, y1=y1)

    @staticmethod
    def crop(frame: np.ndarray, roi: RoiWindow) -> np.ndarray:
        return frame[roi.y0:roi.y1, roi.x0:roi.x1].copy()

    @staticmethod
    def to_full_coords(x_roi: float, y_roi: float, roi: RoiWindow) -> Tuple[float, float]:
        return (x_roi + roi.x0, y_roi + roi.y0)

    @staticmethod
    def to_full_coords_bbox(bbox_roi: Tuple[int, int, int, int], roi: RoiWindow) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox_roi
        return (x0 + roi.x0, y0 + roi.y0, x1 + roi.x0, y1 + roi.y0)
