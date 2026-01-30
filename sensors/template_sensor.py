# sensors/template_sensor.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from sensors.base import Sensor, SensorResult
from tracking.track import Track
from roi.roi_manager import RoiWindow, RoiManager


class TemplateSensor(Sensor):
    """
    Gray + Edge(Sobel) 기반 NCC 템플릿 매칭 센서.
    - seed에서 patch 저장
    - 매 프레임 ROI 내에서 NCC로 center 추정
    - QA 품질이 높을 때만 천천히 template 업데이트
    """

    def __init__(self, cfg: Dict[str, Any]):
        tcfg = cfg.get("sensors", {}).get("template", {})
        self.enabled = bool(tcfg.get("enabled", True))
        self.patch_size = int(tcfg.get("patch_size", 64))
        self.use_edge = bool(tcfg.get("use_edge", True))
        self.update_thresh = float(tcfg.get("update_quality_thresh", 0.85))
        self.measurement_r = float(tcfg.get("measurement_noise_r", 25.0))

    def initialize(self, track: Track, roi: RoiWindow,
                   crop_bgr: np.ndarray, crop_gray: np.ndarray) -> None:
        if not self.enabled:
            return
        # seed bbox를 ROI 좌표로 변환
        if track.bbox is None:
            return
        x0, y0, x1, y1 = track.bbox
        rx0 = x0 - roi.x0
        ry0 = y0 - roi.y0
        rx1 = x1 - roi.x0
        ry1 = y1 - roi.y0

        # bbox 중심에서 patch 추출
        cx = (rx0 + rx1) // 2
        cy = (ry0 + ry1) // 2
        half = self.patch_size // 2
        h, w = crop_gray.shape[:2]

        px0 = max(0, cx - half)
        py0 = max(0, cy - half)
        px1 = min(w, cx + half)
        py1 = min(h, cy + half)

        patch_gray = crop_gray[py0:py1, px0:px1]
        if patch_gray.size == 0:
            return

        track.template = patch_gray.copy()
        if self.use_edge:
            track.template_edge = self._edge(patch_gray)
        track.template_center = (float(x0 + x1) / 2, float(y0 + y1) / 2)

    def measure(self, track: Track, roi: RoiWindow,
                crop_bgr: np.ndarray, crop_gray: np.ndarray,
                frame_idx: int) -> Optional[SensorResult]:
        if not self.enabled or track.template is None:
            return None

        tpl = track.template
        th, tw = tpl.shape[:2]
        sh, sw = crop_gray.shape[:2]

        if th > sh or tw > sw:
            return None

        # gray NCC
        res_gray = cv2.matchTemplate(crop_gray, tpl, cv2.TM_CCOEFF_NORMED)
        score = float(res_gray.max())

        # edge NCC (optional blend)
        if self.use_edge and track.template_edge is not None:
            edge_crop = self._edge(crop_gray)
            edge_tpl = track.template_edge
            if edge_tpl.shape[0] <= edge_crop.shape[0] and edge_tpl.shape[1] <= edge_crop.shape[1]:
                res_edge = cv2.matchTemplate(edge_crop, edge_tpl, cv2.TM_CCOEFF_NORMED)
                # 두 결과 결합 (같은 크기 보장)
                min_h = min(res_gray.shape[0], res_edge.shape[0])
                min_w = min(res_gray.shape[1], res_edge.shape[1])
                combined = 0.5 * res_gray[:min_h, :min_w] + 0.5 * res_edge[:min_h, :min_w]
            else:
                combined = res_gray
        else:
            combined = res_gray

        _, max_val, _, max_loc = cv2.minMaxLoc(combined)
        # max_loc = (x, y) in search result space
        # template center offset
        match_x = max_loc[0] + tw / 2
        match_y = max_loc[1] + th / 2

        # ROI → full-frame
        full_x = match_x + roi.x0
        full_y = match_y + roi.y0

        confidence = float(max_val)
        # NCC range: clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return SensorResult(
            center=(full_x, full_y),
            confidence=confidence,
            metadata={"ncc_score": float(max_val)},
        )

    def update_template(self, track: Track, roi: RoiWindow,
                        crop_gray: np.ndarray, quality: float) -> None:
        """QA 품질이 높을 때 template을 천천히 블렌딩 업데이트."""
        if not self.enabled or track.template is None:
            return
        if quality < self.update_thresh:
            return

        center = track.last_center
        if center is None:
            return

        # 현재 center 위치에서 새 patch 추출
        cx_roi = center[0] - roi.x0
        cy_roi = center[1] - roi.y0
        half = self.patch_size // 2
        h, w = crop_gray.shape[:2]

        px0 = int(max(0, cx_roi - half))
        py0 = int(max(0, cy_roi - half))
        px1 = int(min(w, cx_roi + half))
        py1 = int(min(h, cy_roi + half))

        new_patch = crop_gray[py0:py1, px0:px1]
        if new_patch.shape != track.template.shape:
            return

        # exponential blend (slow update)
        alpha = 0.1
        track.template = cv2.addWeighted(
            track.template.astype(np.float32), 1.0 - alpha,
            new_patch.astype(np.float32), alpha, 0.0
        ).astype(np.uint8)

        if self.use_edge:
            track.template_edge = self._edge(track.template)

        track.template_center = center

    @staticmethod
    def _edge(gray: np.ndarray) -> np.ndarray:
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx * sx + sy * sy)
        mag = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        return mag
