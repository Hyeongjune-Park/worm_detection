# sensors/sam2_sensor.py
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from sensors.base import Sensor, SensorResult
from tracking.track import Track
from roi.roi_manager import RoiWindow


class Sam2Sensor(Sensor):
    """
    SAM2 기반 센서: ROI crop에서 mask → centroid / PCA axis / endpoints / border_touch.
    SAM2 사용 불가 시 None 반환 → Template+KLT+Kalman만으로 운용.
    (MotionMask fallback 제거: 전역 모션 기반은 설계 철학과 충돌)

    Fix 7: prev_mask/points 힌트 캐싱 — QA 좋은 프레임의 mask center를
    다음 프레임 point prompt로 사용하여 안정성 향상.
    """

    def __init__(self, cfg: Dict[str, Any]):
        scfg = cfg.get("sensors", {}).get("sam2", {})
        self.enabled = bool(scfg.get("enabled", True))
        self.model_type = str(scfg.get("model_type", "sam2_hiera_tiny"))
        self.ckpt = str(scfg.get("checkpoint_path", ""))
        self.update_every_n = int(scfg.get("update_every_n", 1))
        self.measurement_r = float(scfg.get("measurement_noise_r", 10.0))

        self._available = False
        self._predictor = None
        self._init_error = ""
        if self.enabled:
            self._init_sam2()

        # prev_mask 캐시 (track_id → (mask_u8, center_full))
        self._prev_masks: Dict[int, np.ndarray] = {}
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        # 임시 저장 (QA 확정 전)
        self._pending: Dict[int, Tuple[np.ndarray, Tuple[float, float]]] = {}

    def _init_sam2(self):
        sam2_dir = os.path.abspath("sam2")
        if os.path.isdir(sam2_dir) and sam2_dir not in sys.path:
            sys.path.insert(0, sam2_dir)

        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            self._available = False
            self._init_error = f"IMPORT_FAIL: {repr(e)}"
            print(f"[SAM2] {self._init_error}")
            return

        if not self.ckpt or not os.path.exists(self.ckpt):
            self._available = False
            self._init_error = f"CKPT_NOT_FOUND: {self.ckpt}"
            print(f"[SAM2] {self._init_error}")
            return

        try:
            self._torch = torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Hydra config search path 설정 (SAM2.1 requires hydra configs)
            config_dir = os.path.join(sam2_dir, "sam2", "configs")
            if os.path.isdir(config_dir):
                from hydra.core.global_hydra import GlobalHydra  # type: ignore
                from hydra import initialize_config_dir  # type: ignore
                GlobalHydra.instance().clear()
                ctx = initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.2")
                ctx.__enter__()
                # keep context open for lifetime of process

            model = build_sam2(self.model_type, self.ckpt, device=device)
            self._predictor = SAM2ImagePredictor(model)
            self._available = True
            print(f"[SAM2] loaded on {device}")
        except Exception as e:
            self._available = False
            self._init_error = f"INIT_FAIL: {repr(e)}"
            print(f"[SAM2] {self._init_error}")

    @property
    def available(self) -> bool:
        return self._available

    def initialize(self, track: Track, roi: RoiWindow,
                   crop_bgr: np.ndarray, crop_gray: np.ndarray) -> None:
        pass

    def measure(self, track: Track, roi: RoiWindow,
                crop_bgr: np.ndarray, crop_gray: np.ndarray,
                frame_idx: int) -> Optional[SensorResult]:
        if not self.enabled or not self._available:
            return None
        if self.update_every_n > 1 and (frame_idx % self.update_every_n) != 0:
            return None

        try:
            return self._measure_sam2(track, roi, crop_bgr)
        except Exception:
            return None

    def _measure_sam2(self, track: Track, roi: RoiWindow,
                      crop_bgr: np.ndarray) -> Optional[SensorResult]:
        center = track.center()
        if center[0] is None:
            return None

        h, w = crop_bgr.shape[:2]

        # box prompt
        if track.bbox is not None:
            bx0, by0, bx1, by1 = track.bbox
            box = np.array([
                bx0 - roi.x0, by0 - roi.y0,
                bx1 - roi.x0, by1 - roi.y0
            ], dtype=np.float32)
        else:
            cx = center[0] - roi.x0
            cy = center[1] - roi.y0
            box = np.array([cx - 30, cy - 30, cx + 30, cy + 30], dtype=np.float32)

        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(w, box[2])
        box[3] = min(h, box[3])

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(crop_rgb)

        # Fix 7: 이전 QA 좋은 프레임의 mask center → point prompt 힌트
        point_coords = None
        point_labels = None
        prev_center = self._prev_centers.get(track.id)
        if prev_center is not None:
            px = prev_center[0] - roi.x0
            py = prev_center[1] - roi.y0
            if 0 <= px < w and 0 <= py < h:
                point_coords = np.array([[px, py]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = self._predictor.predict(
            box=box[None, :],
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        mask_u8 = (mask.astype(np.uint8)) * 255

        if np.count_nonzero(mask_u8) == 0:
            return None

        result = self._analyze_mask(mask_u8, roi, (h, w))

        # pending에 임시 저장 (QA 좋으면 cache_good_result() 호출)
        if result.center is not None:
            self._pending[track.id] = (mask_u8, result.center)

        return result

    def cache_good_result(self, track_id: int) -> None:
        """QA 통과 시 호출 → prev_mask/center 확정 캐시."""
        pending = self._pending.pop(track_id, None)
        if pending is not None:
            mask_u8, center = pending
            self._prev_masks[track_id] = mask_u8
            self._prev_centers[track_id] = center

    def _analyze_mask(self, mask_u8: np.ndarray, roi: RoiWindow,
                      crop_shape: Tuple[int, int]) -> SensorResult:
        h, w = crop_shape

        ys, xs = np.where(mask_u8 > 0)
        cx_roi = float(np.mean(xs))
        cy_roi = float(np.mean(ys))
        full_cx = cx_roi + roi.x0
        full_cy = cy_roi + roi.y0

        endpoints = None
        if len(xs) > 10:
            endpoints = self._pca_endpoints(xs, ys, roi)

        total_fg = len(xs)
        border_pixels = 0
        if total_fg > 0:
            on_border = (
                (xs <= 1) | (xs >= w - 2) |
                (ys <= 1) | (ys >= h - 2)
            )
            border_pixels = int(np.sum(on_border))
        border_touch = float(border_pixels / max(1, total_fg))

        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        bbox_roi = (x_min, y_min, x_max + 1, y_max + 1)

        return SensorResult(
            center=(full_cx, full_cy),
            confidence=1.0 - border_touch,
            mask=mask_u8,
            endpoints=endpoints,
            border_touch=border_touch,
            bbox_roi=bbox_roi,
            area=int(total_fg),
        )

    @staticmethod
    def _pca_endpoints(xs: np.ndarray, ys: np.ndarray,
                       roi: RoiWindow) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        mean = pts.mean(axis=0)
        centered = pts - mean

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, np.argmax(eigvals)]

        projections = centered @ major
        idx_min = int(np.argmin(projections))
        idx_max = int(np.argmax(projections))

        ep1 = (float(pts[idx_min, 0]) + roi.x0, float(pts[idx_min, 1]) + roi.y0)
        ep2 = (float(pts[idx_max, 0]) + roi.x0, float(pts[idx_max, 1]) + roi.y0)

        return (ep1, ep2)


def build_sam2_sensor(cfg: Dict[str, Any]) -> Sam2Sensor:
    """
    SAM2 센서 생성. 실패해도 Sam2Sensor 반환 (available=False → measure()는 None).
    MotionMask fallback 제거: TPL+KLT+Kalman만으로 운용.
    """
    sensor = Sam2Sensor(cfg)
    if sensor.available:
        return sensor
    print(f"[SAM2] not available ({sensor._init_error}) -> TPL+KLT+Kalman only")
    return sensor
