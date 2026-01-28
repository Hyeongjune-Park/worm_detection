# sensors/head_estimator.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from sensors.base import SensorResult


class HeadEstimator:
    """
    Head 추정기: PCA major axis endpoints + 이동 방향 벡터로 head 선택.

    1차: SAM2 mask endpoints 사용 (있을 때)
    2차: ROI crop의 edge-based PCA fallback (SAM2 없을 때)
    - 정지/모호하면: head = None
    """

    def __init__(self, cfg: Dict[str, Any]):
        hcfg = cfg.get("head", {})
        self.enabled = bool(hcfg.get("enabled", True))
        self.min_movement_px = float(hcfg.get("min_movement_px", 2.0))

    def estimate(
        self,
        sam2_result: Optional[SensorResult],
        velocity: Tuple[float, float],
        crop_gray: Optional[np.ndarray] = None,
        roi_origin: Optional[Tuple[int, int]] = None,
        fused_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Args:
            sam2_result: SAM2 센서 결과 (endpoints 포함, 없을 수 있음)
            velocity: Kalman velocity (vx, vy) in pixels/frame
            crop_gray: ROI grayscale crop (edge fallback용)
            roi_origin: (roi.x0, roi.y0) for coordinate conversion
            fused_center: fusion 결과 center (edge fallback 시 기준점)
        Returns:
            head (x, y) full-frame 또는 None
        """
        if not self.enabled:
            return None

        vx, vy = velocity
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < self.min_movement_px:
            return None

        # 이동 방향 단위 벡터
        dir_x = vx / (speed + 1e-8)
        dir_y = vy / (speed + 1e-8)

        # 1차: SAM2 mask endpoints
        if sam2_result is not None and sam2_result.endpoints is not None:
            center = sam2_result.center
            if center is not None:
                return self._pick_head(sam2_result.endpoints, center, dir_x, dir_y)

        # 2차: edge-based PCA fallback
        if crop_gray is not None and roi_origin is not None and fused_center is not None:
            endpoints = self._edge_pca(crop_gray, roi_origin)
            if endpoints is not None:
                return self._pick_head(endpoints, fused_center, dir_x, dir_y)

        return None

    @staticmethod
    def _pick_head(
        endpoints: Tuple[Tuple[float, float], Tuple[float, float]],
        center: Tuple[float, float],
        dir_x: float,
        dir_y: float,
    ) -> Optional[Tuple[float, float]]:
        """이동 방향과 더 정렬된 endpoint를 head로 선택."""
        ep1, ep2 = endpoints

        def alignment(ep):
            dx = ep[0] - center[0]
            dy = ep[1] - center[1]
            norm = (dx * dx + dy * dy) ** 0.5
            if norm < 1e-6:
                return 0.0
            return (dx * dir_x + dy * dir_y) / norm

        a1 = alignment(ep1)
        a2 = alignment(ep2)

        if a1 > a2:
            return ep1
        elif a2 > a1:
            return ep2
        return None

    @staticmethod
    def _edge_pca(
        crop_gray: np.ndarray,
        roi_origin: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Edge-based PCA fallback: Canny edge에서 PCA major axis endpoints 추출.
        SAM2 mask 없이도 elongated object의 장축을 추정.
        """
        edges = cv2.Canny(crop_gray, 50, 150)
        ys, xs = np.where(edges > 0)
        if len(xs) < 20:
            return None

        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        mean = pts.mean(axis=0)
        centered = pts - mean

        cov = np.cov(centered.T)
        if cov.shape != (2, 2):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, np.argmax(eigvals)]

        projections = centered @ major
        idx_min = int(np.argmin(projections))
        idx_max = int(np.argmax(projections))

        ox, oy = roi_origin
        ep1 = (float(pts[idx_min, 0]) + ox, float(pts[idx_min, 1]) + oy)
        ep2 = (float(pts[idx_max, 0]) + ox, float(pts[idx_max, 1]) + oy)
        return (ep1, ep2)
