# detection/background_model.py
from __future__ import annotations

from typing import Optional
import numpy as np


class RunningAvgBackground:
    """
    단순하지만 실험실 영상에서 튼튼한 배경모델:
    - background: float32
    - fg_mask에서 전경 픽셀은 업데이트 제외 가능
    """
    def __init__(self, alpha: float = 0.02, freeze_fg_update: bool = True):
        self.alpha = float(alpha)
        self.freeze_fg_update = bool(freeze_fg_update)
        self.bg: Optional[np.ndarray] = None  # float32

    def initialize(self, gray_u8: np.ndarray) -> None:
        self.bg = gray_u8.astype(np.float32)

    def update(self, gray_u8: np.ndarray, fg_mask_u8: Optional[np.ndarray] = None) -> None:
        if self.bg is None:
            self.initialize(gray_u8)
            return

        a = self.alpha
        if not self.freeze_fg_update or fg_mask_u8 is None:
            self.bg = (1 - a) * self.bg + a * gray_u8.astype(np.float32)
            return

        # fg(255)로 판단된 픽셀은 업데이트 제외
        bg = self.bg
        g = gray_u8.astype(np.float32)
        mask_bg = (fg_mask_u8 == 0)
        bg[mask_bg] = (1 - a) * bg[mask_bg] + a * g[mask_bg]
        self.bg = bg

    def diff(self, gray_u8: np.ndarray) -> np.ndarray:
        if self.bg is None:
            self.initialize(gray_u8)
        d = np.abs(gray_u8.astype(np.float32) - self.bg)
        return d.astype(np.uint8)
