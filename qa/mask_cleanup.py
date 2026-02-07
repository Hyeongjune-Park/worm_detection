# qa/mask_cleanup.py
"""SAM2 마스크 후처리 — 잎맥 번짐(vein bleeding) 정리.

CCF(Connected Component Filtering) + 조건부 Erosion + Expected Region AND.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from sensors.base import SensorResult
from roi.roi_manager import RoiWindow


def cleanup_mask(
    mask_u8: np.ndarray,
    pred_center_roi: Tuple[float, float],
    baseline_area: Optional[int],
    baseline_aspect_ratio: float,
    roi_shape: Tuple[int, int],
) -> Tuple[np.ndarray, bool]:
    """번짐 의심 마스크를 정리하여 벌레 부분만 추출 시도.

    Args:
        mask_u8: SAM2 출력 mask (ROI coords, 0/255)
        pred_center_roi: Kalman 예측 위치 (ROI 내 좌표)
        baseline_area: 첫 프레임 면적 (없으면 CCF만)
        baseline_aspect_ratio: 첫 프레임 종횡비
        roi_shape: (H, W)

    Returns:
        (cleaned_mask, was_cleaned) — was_cleaned=False면 원본 그대로
    """
    original_area = int(np.count_nonzero(mask_u8))
    if original_area == 0:
        return mask_u8, False

    # Step 1: CCF — pred_center에 가장 가까운 컴포넌트만 유지
    cleaned = _keep_nearest_component(mask_u8, pred_center_roi)
    n_components_step1 = _count_components(mask_u8)

    # Step 2: 컴포넌트가 1개였으면 (전부 연결) erosion으로 끊기 시도
    if n_components_step1 <= 1:
        eroded = _erode_and_filter(mask_u8, pred_center_roi)
        if eroded is not None:
            cleaned = eroded

    # Step 3: Expected Region AND (baseline 있을 때만)
    if baseline_area is not None and baseline_area > 0:
        cleaned = _apply_expected_region(
            cleaned, pred_center_roi, baseline_area,
            baseline_aspect_ratio, roi_shape
        )

    cleaned_area = int(np.count_nonzero(cleaned))
    if cleaned_area == 0:
        return mask_u8, False

    was_cleaned = not np.array_equal(cleaned, mask_u8)
    return cleaned, was_cleaned


def rebuild_sensor_result(
    mask_u8: np.ndarray,
    roi: RoiWindow,
    roi_shape: Tuple[int, int],
) -> SensorResult:
    """정리된 mask에서 SensorResult 재구성."""
    h, w = roi_shape
    ys, xs = np.where(mask_u8 > 0)
    total_fg = len(xs)

    if total_fg == 0:
        return SensorResult()

    cx_roi = float(np.mean(xs))
    cy_roi = float(np.mean(ys))
    full_cx = cx_roi + roi.x0
    full_cy = cy_roi + roi.y0

    # PCA endpoints
    endpoints = None
    if total_fg > 10:
        endpoints = _pca_endpoints(xs, ys, roi)

    # Border touch
    on_border = (
        (xs <= 1) | (xs >= w - 2) |
        (ys <= 1) | (ys >= h - 2)
    )
    border_pixels = int(np.sum(on_border))
    border_touch = float(border_pixels / max(1, total_fg))

    # BBox
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


# ── Internal helpers ──────────────────────────────────────────────

def _count_components(mask_u8: np.ndarray) -> int:
    """foreground 컴포넌트 수 (background 제외)."""
    binary = (mask_u8 > 0).astype(np.uint8)
    n, _ = cv2.connectedComponents(binary)
    return n - 1  # label 0 = background


def _keep_nearest_component(
    mask_u8: np.ndarray,
    pred_center_roi: Tuple[float, float],
) -> np.ndarray:
    """pred_center에 가장 가까운 connected component만 유지."""
    binary = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    if n <= 2:  # background + 1 component → 분리 불가
        return mask_u8

    cx, cy = pred_center_roi
    best_label = -1
    best_dist = float('inf')

    for label_id in range(1, n):  # skip background (0)
        ccx, ccy = centroids[label_id]
        dist = math.hypot(ccx - cx, ccy - cy)
        if dist < best_dist:
            best_dist = dist
            best_label = label_id

    if best_label < 0:
        return mask_u8

    result = np.zeros_like(mask_u8)
    result[labels == best_label] = 255
    return result


def _erode_and_filter(
    mask_u8: np.ndarray,
    pred_center_roi: Tuple[float, float],
) -> Optional[np.ndarray]:
    """Erosion으로 얇은 연결 끊은 후 CCF. 실패 시 None."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=2)

    if np.count_nonzero(eroded) == 0:
        return None  # erosion으로 전부 사라짐

    # eroded mask에서 CCF
    nearest = _keep_nearest_component(eroded, pred_center_roi)

    if np.count_nonzero(nearest) == 0:
        return None

    # dilate로 크기 복원
    restored = cv2.dilate(nearest, kernel, iterations=2)
    return restored


def _apply_expected_region(
    mask_u8: np.ndarray,
    pred_center_roi: Tuple[float, float],
    baseline_area: int,
    baseline_aspect_ratio: float,
    roi_shape: Tuple[int, int],
) -> np.ndarray:
    """Kalman 예측 + baseline 크기로 예상 영역 제한 (AND)."""
    h, w = roi_shape
    ar = max(baseline_aspect_ratio, 1.0)

    # 타원 반지름 계산: area = pi * a * b, a = ar * b
    b = math.sqrt(baseline_area / (math.pi * ar))
    a = ar * b

    # 1.5x 안전 마진
    a_exp = int(a * 1.5) + 1
    b_exp = int(b * 1.5) + 1

    cx, cy = int(round(pred_center_roi[0])), int(round(pred_center_roi[1]))

    expected = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(expected, (cx, cy), (a_exp, b_exp), 0, 0, 360, 255, -1)

    result = cv2.bitwise_and(mask_u8, expected)

    # 과도 절삭 방지: 결과가 baseline의 20% 미만이면 적용 취소
    result_area = int(np.count_nonzero(result))
    if result_area < baseline_area * 0.2:
        return mask_u8

    return result


def _pca_endpoints(
    xs: np.ndarray, ys: np.ndarray, roi: RoiWindow,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """PCA 기반 양 끝점 (full-frame 좌표). sam2_sensor._pca_endpoints와 동일."""
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
