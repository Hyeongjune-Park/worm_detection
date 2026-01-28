# sensors/klt_sensor.py
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from sensors.base import Sensor, SensorResult
from tracking.track import Track
from roi.roi_manager import RoiWindow


class KltSensor(Sensor):
    """
    KLT Optical Flow 센서.
    - seed bbox 내에서 goodFeaturesToTrack으로 초기화
    - calcOpticalFlowPyrLK + forward-backward error 검증
    - 유효 점의 median → center_klt
    """

    def __init__(self, cfg: Dict[str, Any]):
        kcfg = cfg.get("sensors", {}).get("klt", {})
        self.enabled = bool(kcfg.get("enabled", True))
        self.max_corners = int(kcfg.get("max_corners", 100))
        self.quality_level = float(kcfg.get("quality_level", 0.01))
        self.min_distance = int(kcfg.get("min_distance", 7))
        self.win_size = int(kcfg.get("win_size", 15))
        self.max_level = int(kcfg.get("max_level", 3))
        self.min_valid_points = int(kcfg.get("min_valid_points", 10))
        self.measurement_r = float(kcfg.get("measurement_noise_r", 25.0))

        self._lk_params = dict(
            winSize=(self.win_size, self.win_size),
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
        )

    def initialize(self, track: Track, roi: RoiWindow,
                   crop_bgr: np.ndarray, crop_gray: np.ndarray) -> None:
        if not self.enabled:
            return
        self._init_features(track, roi, crop_gray)

    def _init_features(self, track: Track, roi: RoiWindow, crop_gray: np.ndarray) -> None:
        """ROI 내에서 bbox 영역에 마스크를 걸어 feature 추출."""
        h, w = crop_gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if track.bbox is not None:
            x0, y0, x1, y1 = track.bbox
            rx0 = max(0, x0 - roi.x0)
            ry0 = max(0, y0 - roi.y0)
            rx1 = min(w, x1 - roi.x0)
            ry1 = min(h, y1 - roi.y0)
            mask[ry0:ry1, rx0:rx1] = 255
        else:
            # bbox 없으면 ROI 전체
            mask[:] = 255

        pts = cv2.goodFeaturesToTrack(crop_gray, mask=mask, **self._feature_params)
        if pts is not None and len(pts) > 0:
            track.klt_points = pts.reshape(-1, 2).astype(np.float32)
        else:
            track.klt_points = None
        track.klt_prev_gray = crop_gray.copy()

    def measure(self, track: Track, roi: RoiWindow,
                crop_bgr: np.ndarray, crop_gray: np.ndarray,
                frame_idx: int) -> Optional[SensorResult]:
        if not self.enabled:
            return None

        if track.klt_points is None or track.klt_prev_gray is None:
            # reinit
            self._init_features(track, roi, crop_gray)
            return None

        prev_gray = track.klt_prev_gray
        prev_pts = track.klt_points.reshape(-1, 1, 2)

        # ROI 크기가 달라질 수 있으므로 체크
        if prev_gray.shape != crop_gray.shape:
            self._init_features(track, roi, crop_gray)
            return None

        # forward flow
        next_pts, status_f, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, crop_gray, prev_pts, None, **self._lk_params
        )
        if next_pts is None:
            self._init_features(track, roi, crop_gray)
            return None

        # backward flow (forward-backward check)
        back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(
            crop_gray, prev_gray, next_pts, None, **self._lk_params
        )

        # 유효 점 필터링
        valid = np.ones(len(prev_pts), dtype=bool)
        valid &= (status_f.ravel() == 1)
        if back_pts is not None and status_b is not None:
            valid &= (status_b.ravel() == 1)
            # round-trip error < 1px
            fb_err = np.linalg.norm(
                prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            valid &= (fb_err < 1.0)

        good_pts = next_pts.reshape(-1, 2)[valid]

        if len(good_pts) < self.min_valid_points:
            # 유효 점 부족 → reinit
            self._init_features(track, roi, crop_gray)
            return None

        # median center (ROI 좌표)
        cx_roi = float(np.median(good_pts[:, 0]))
        cy_roi = float(np.median(good_pts[:, 1]))

        # full-frame 변환
        full_x = cx_roi + roi.x0
        full_y = cy_roi + roi.y0

        # 상태 갱신
        track.klt_points = good_pts.astype(np.float32)
        track.klt_prev_gray = crop_gray.copy()

        valid_ratio = float(len(good_pts)) / max(1, len(prev_pts))

        return SensorResult(
            center=(full_x, full_y),
            confidence=valid_ratio,
            metadata={"valid_points": len(good_pts), "total_points": len(prev_pts)},
        )
