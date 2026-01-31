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
        self._prev_areas: Dict[int, int] = {}  # 이전 프레임 면적 (연속성 검사용)
        # 임시 저장 (QA 확정 전)
        self._pending: Dict[int, Tuple[np.ndarray, Tuple[float, float], int]] = {}  # (mask, center, area)

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
                frame_idx: int, box_expansion: float = 1.0) -> Optional[SensorResult]:
        """
        Args:
            box_expansion: Box prompt 확장 배율 (1.0 = 기본, 1.5 = 1.5배 확장)
                          UNCERTAIN/OCCLUDED 상태에서 재획득을 위해 확장
        """
        if not self.enabled or not self._available:
            return None
        if self.update_every_n > 1 and (frame_idx % self.update_every_n) != 0:
            return None

        try:
            return self._measure_sam2(track, roi, crop_bgr, box_expansion)
        except Exception as e:
            print(f"[SAM2] measure exception track={track.id} frame={frame_idx}: {e}")
            return None

    def _measure_sam2(self, track: Track, roi: RoiWindow,
                      crop_bgr: np.ndarray, box_expansion: float = 1.0) -> Optional[SensorResult]:
        h, w = crop_bgr.shape[:2]

        # Box prompt: Kalman 예측 위치 중심으로 생성
        # 핵심: 빠르게 움직이는 벌레도 Kalman이 예측한 위치에서 찾아야 함
        pred_x, pred_y = track.kf.get_position()
        cx_roi = pred_x - roi.x0
        cy_roi = pred_y - roi.y0

        # Box 크기: seed bbox 크기 기반 (box_expansion으로 확장 가능)
        # 사용자가 시드 지정 시 벌레 전체를 감싸도록 그리므로 그 크기가 기준
        # UNCERTAIN/OCCLUDED 상태에서는 box_expansion=1.5로 확장하여 재획득 시도
        if track.seed_bbox_size is not None:
            seed_w, seed_h = track.seed_bbox_size
            half_w = int(seed_w * box_expansion) // 2
            half_h = int(seed_h * box_expansion) // 2
        else:
            half_w = int(50 * box_expansion)  # fallback: 100x100 * expansion
            half_h = int(50 * box_expansion)

        box = np.array([
            cx_roi - half_w, cy_roi - half_h,
            cx_roi + half_w, cy_roi + half_h
        ], dtype=np.float32)

        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(w, box[2])
        box[3] = min(h, box[3])

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(crop_rgb)

        # Point prompt: Kalman 예측 위치 사용
        # 핵심: 이전 위치가 아닌 현재 예측 위치에서 벌레를 찾음
        point_coords = None
        point_labels = None
        if 0 <= cx_roi < w and 0 <= cy_roi < h:
            point_coords = np.array([[cx_roi, cy_roi]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = self._predictor.predict(
            box=box[None, :],
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # 복합 점수 기반 mask 선택 (score만이 아닌 여러 요소 고려)
        best_idx = self._select_best_mask(
            masks, scores,
            pred_center_roi=(cx_roi, cy_roi),
            prev_area=self._prev_areas.get(track.id),
            crop_shape=(h, w)
        )
        mask = masks[best_idx]
        mask_u8 = (mask.astype(np.uint8)) * 255
        area = int(np.count_nonzero(mask_u8))

        if area == 0:
            return None

        result = self._analyze_mask(mask_u8, roi, (h, w))

        # pending에 임시 저장 (QA 좋으면 cache_good_result() 호출)
        if result.center is not None:
            self._pending[track.id] = (mask_u8, result.center, area)

        return result

    def _select_best_mask(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        pred_center_roi: Tuple[float, float],
        prev_area: Optional[int],
        crop_shape: Tuple[int, int]
    ) -> int:
        """
        복합 점수 기반 mask 선택.

        기존: argmax(scores) - SAM2 점수만 고려
        개선: score + pred 근접성 + 면적 일관성 + border_touch 종합 고려

        이렇게 하면 "점수만 높고 엉뚱한 mask"가 선택되는 것을 방지.
        하드 거부가 아닌 소프트 스코어링이므로 오탐 위험이 낮음.
        """
        h, w = crop_shape
        cx_pred, cy_pred = pred_center_roi

        candidates = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_u8 = mask.astype(np.uint8)
            area = int(np.count_nonzero(mask_u8))

            if area == 0:
                candidates.append((float('-inf'), i))
                continue

            # 1. SAM2 score (0~1, 높을수록 좋음)
            sam2_score = float(score)

            # 2. Center 계산 및 pred 근접성 (가까울수록 좋음)
            ys, xs = np.where(mask_u8 > 0)
            cx_mask = float(np.mean(xs))
            cy_mask = float(np.mean(ys))

            dist_to_pred = np.sqrt((cx_mask - cx_pred)**2 + (cy_mask - cy_pred)**2)
            roi_diag = np.sqrt(h**2 + w**2)
            dist_penalty = min(dist_to_pred / roi_diag, 1.0)  # 정규화 (0~1)

            # 3. 면적 일관성 (이전 면적과 비슷할수록 좋음)
            area_penalty = 0.0
            if prev_area is not None and prev_area > 0:
                area_ratio = area / prev_area
                # 0.5배 ~ 2배 범위를 벗어나면 페널티 증가
                if area_ratio < 0.5:
                    area_penalty = (0.5 - area_ratio) * 0.5  # 너무 작으면 감점
                elif area_ratio > 2.0:
                    area_penalty = (area_ratio - 2.0) * 0.3  # 너무 크면 감점
                area_penalty = min(area_penalty, 0.5)  # 최대 0.5 감점

            # 4. Border touch (경계 접촉 비율, 낮을수록 좋음)
            on_border = (
                (xs <= 1) | (xs >= w - 2) |
                (ys <= 1) | (ys >= h - 2)
            )
            border_touch = float(np.sum(on_border)) / max(1, area)

            # 복합 점수 계산
            # 가중치: SAM2 점수 기본, 거리/면적/경계는 페널티로 적용
            combined = (
                sam2_score * 1.0           # SAM2 점수 (0~1)
                - dist_penalty * 0.3       # 거리 페널티 (최대 -0.3)
                - area_penalty             # 면적 페널티 (최대 -0.5)
                - border_touch * 0.3       # 경계 페널티 (최대 -0.3)
            )

            candidates.append((combined, i))

        # 가장 높은 복합 점수의 mask 선택
        candidates.sort(reverse=True)
        return candidates[0][1]

    def cache_good_result(self, track_id: int) -> None:
        """QA 통과 시 호출 → prev_mask/center/area 확정 캐시."""
        pending = self._pending.pop(track_id, None)
        if pending is not None:
            mask_u8, center, area = pending
            self._prev_masks[track_id] = mask_u8
            self._prev_centers[track_id] = center
            self._prev_areas[track_id] = area

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
