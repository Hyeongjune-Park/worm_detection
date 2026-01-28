# detection/motion_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

from detection.background_model import RunningAvgBackground


@dataclass
class Blob:
    cx: float
    cy: float
    bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1
    area: int
    aspect_ratio: float
    solidity: float


@dataclass
class MotionDetections:
    fg_mask: np.ndarray  # uint8 0/255
    blobs: List[Blob]


def _clamp_rect(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(W - 1, int(x0)))
    y0 = max(0, min(H - 1, int(y0)))
    x1 = max(1, min(W, int(x1)))
    y1 = max(1, min(H, int(y1)))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


class MotionDetector:
    """
    모션 기반 전경 마스크 + blob 추출기

    추가 기능:
      - detection.arena.manual_rect.enabled == true 인 경우,
        사각형 ROI 안에서만 fg/blob를 계산(영역 밖은 무조건 0)

    config.yaml 예시:

    detection:
      arena:
        enabled: true
        manual_rect:
          enabled: true
          x0: 100
          y0: 50
          x1: 1800
          y1: 1000
    """

    def __init__(self, cfg: Dict[str, Any]):
        d = cfg.get("detection", {})

        # -------- background model --------
        bg_cfg = d.get("bg", {})
        self.bg = RunningAvgBackground(
            alpha=float(bg_cfg.get("alpha", 0.02)),
            freeze_fg_update=bool(bg_cfg.get("freeze_fg_update", True)),
        )

        # -------- fg extraction --------
        fg_cfg = d.get("fg", {})
        self.th = int(fg_cfg.get("threshold", 25))
        self.morph_open = int(fg_cfg.get("morph_open", 3))
        self.morph_close = int(fg_cfg.get("morph_close", 5))

        # -------- blob filtering --------
        bcfg = d.get("blob", {})
        self.min_area = int(bcfg.get("min_area", 40))
        self.max_area = int(bcfg.get("max_area", 30000))
        self.max_aspect = float(bcfg.get("max_aspect_ratio", 12.0))
        self.min_solidity = float(bcfg.get("min_solidity", 0.2))

        # -------- manual arena rect (optional) --------
        arena_cfg = d.get("arena", {})
        self.arena_enabled = bool(arena_cfg.get("enabled", False))

        rect_cfg = arena_cfg.get("manual_rect", {})
        self.manual_rect_enabled = bool(rect_cfg.get("enabled", False))

        self.rx0 = rect_cfg.get("x0", None)
        self.ry0 = rect_cfg.get("y0", None)
        self.rx1 = rect_cfg.get("x1", None)
        self.ry1 = rect_cfg.get("y1", None)

        self._arena_mask: Optional[np.ndarray] = None  # uint8 0/255, lazy init after first frame known

    def _ensure_arena_mask(self, H: int, W: int) -> None:
        """
        manual_rect가 설정돼 있으면 해당 영역만 255인 mask를 1회 생성
        """
        if self._arena_mask is not None:
            return

        if not (self.arena_enabled and self.manual_rect_enabled):
            return

        if None in (self.rx0, self.ry0, self.rx1, self.ry1):
            return

        x0, y0, x1, y1 = map(int, [self.rx0, self.ry0, self.rx1, self.ry1])
        x0, y0, x1, y1 = _clamp_rect(x0, y0, x1, y1, W=W, H=H)

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        self._arena_mask = mask

    def step(self, gray_u8: np.ndarray) -> MotionDetections:
        H, W = gray_u8.shape[:2]

        # 0) arena mask 준비(필요 시)
        self._ensure_arena_mask(H=H, W=W)

        # 1) diff
        diff = self.bg.diff(gray_u8)

        # 2) threshold -> fg
        _, fg = cv2.threshold(diff, self.th, 255, cv2.THRESH_BINARY)

        # 3) morphology
        if self.morph_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_open, self.morph_open))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        if self.morph_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_close, self.morph_close))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)

        # 3.5) arena mask 적용(영역 밖 제거)  <-- 핵심
        if self._arena_mask is not None:
            fg = cv2.bitwise_and(fg, self._arena_mask)

        # 4) update bg (전경 제외 업데이트 가능)
        self.bg.update(gray_u8, fg_mask_u8=fg)

        # 5) find blobs
        blobs: List[Blob] = []
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            aspect = float(max(w, h) / max(1, min(w, h)))
            if aspect > self.max_aspect:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area / hull_area) if hull_area > 1e-6 else 0.0
            if solidity < self.min_solidity:
                continue

            M = cv2.moments(cnt)
            if abs(M["m00"]) < 1e-6:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            blobs.append(
                Blob(
                    cx=cx,
                    cy=cy,
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                    area=area,
                    aspect_ratio=aspect,
                    solidity=solidity,
                )
            )

        return MotionDetections(fg_mask=fg, blobs=blobs)
