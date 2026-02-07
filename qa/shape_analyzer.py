# qa/shape_analyzer.py
"""마스크 형태 분석 — 벌레 vs 배경/풀마스크 판별."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as _skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


@dataclass
class ShapeStats:
    """마스크 형태 통계."""
    area: int = 0
    perimeter: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 1.0
    p2_over_a: float = 0.0  # perimeter^2 / area
    thickness_med: float = 0.0  # 2 * median(distance_transform)
    thickness_p90: float = 0.0  # 2 * p90(distance_transform)
    border_touch: float = 0.0
    # Skeleton 기반 특성
    skel_length: int = 0
    skel_worm_ratio: float = 0.0  # skeleton_length / mean_radius
    skel_width_cv: float = 0.0  # std(width) / mean(width) along skeleton
    skel_tube_fit: float = 0.0  # area / (skeleton_length * mean_width)
    skel_branch_points: int = 0
    # 최종 평가
    shape_mode: str = "UNKNOWN"  # ELONGATED / COILED / UNKNOWN
    shape_score: float = 0.0  # 0..1
    temporal_score: float = 1.0  # 0..1 (연속성)

    def to_dict(self) -> Dict:
        return {
            "area": self.area,
            "perimeter": round(self.perimeter, 2),
            "solidity": round(self.solidity, 3),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "p2_over_a": round(self.p2_over_a, 2),
            "thickness_med": round(self.thickness_med, 2),
            "thickness_p90": round(self.thickness_p90, 2),
            "border_touch": round(self.border_touch, 3),
            "skel_length": self.skel_length,
            "skel_worm_ratio": round(self.skel_worm_ratio, 2),
            "skel_width_cv": round(self.skel_width_cv, 4),
            "skel_tube_fit": round(self.skel_tube_fit, 4),
            "skel_branch_points": self.skel_branch_points,
            "shape_mode": self.shape_mode,
            "shape_score": round(self.shape_score, 3),
            "temporal_score": round(self.temporal_score, 3),
        }


def _compute_skeleton_features(mask_bin: np.ndarray, dist: np.ndarray, stats: ShapeStats):
    """skeleton 기반 특성 계산 (skimage 있을 때만 호출)."""
    skel = _skeletonize(mask_bin > 0)
    skel_u8 = skel.astype(np.uint8)
    skel_length = int(np.sum(skel_u8))

    if skel_length < 3:
        return

    stats.skel_length = skel_length

    # skeleton 위의 distance transform 값 = 각 지점의 반경
    widths = dist[skel]
    if len(widths) == 0:
        return

    mean_radius = float(np.mean(widths))
    mean_width = mean_radius * 2.0

    # Worm ratio
    stats.skel_worm_ratio = skel_length / max(mean_radius, 1.0)

    # Width CV (coefficient of variation)
    if mean_radius > 1e-6:
        stats.skel_width_cv = float(np.std(widths)) / mean_radius

    # Tube fit: area / (skeleton_length * mean_diameter)
    tube_denom = skel_length * mean_width
    if tube_denom > 0:
        stats.skel_tube_fit = stats.area / tube_denom

    # Branch points (skeleton 이웃 > 2)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, kernel) - skel_u8
    stats.skel_branch_points = int(np.sum((skel_u8 > 0) & (neighbor_count > 2)))


def analyze_mask(
    mask: np.ndarray,
    prev_stats: Optional[ShapeStats] = None,
    roi_shape: Optional[Tuple[int, int]] = None,
) -> ShapeStats:
    """
    마스크 형태 분석.

    Args:
        mask: binary mask (0/255 or 0/1)
        prev_stats: 이전 프레임 stats (연속성 평가용)
        roi_shape: (H, W) ROI 크기 (border_touch 계산용)

    Returns:
        ShapeStats with all computed features
    """
    stats = ShapeStats()

    # 마스크 정규화
    if mask is None or mask.size == 0:
        return stats

    mask_bin = (mask > 0).astype(np.uint8)
    stats.area = int(np.sum(mask_bin))

    if stats.area < 10:
        return stats

    # --- Contour 기반 특성 ---
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return stats

    # 가장 큰 contour 사용
    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)
    if cnt_area < 10:
        return stats

    stats.perimeter = cv2.arcLength(cnt, True)

    # P^2 / A (가느다란 구조일수록 큼)
    if cnt_area > 0:
        stats.p2_over_a = (stats.perimeter ** 2) / cnt_area

    # Solidity (area / convex_hull_area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        stats.solidity = cnt_area / hull_area

    # Aspect ratio (회전된 bounding rect)
    if len(cnt) >= 5:
        (cx, cy), (w, h), angle = cv2.fitEllipse(cnt)
        if h > 0:
            stats.aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
    else:
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if h > 0:
            stats.aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

    # --- Distance Transform 기반 두께 ---
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    dist_vals = dist[mask_bin > 0]
    if len(dist_vals) > 0:
        stats.thickness_med = 2.0 * float(np.median(dist_vals))
        stats.thickness_p90 = 2.0 * float(np.percentile(dist_vals, 90))

    # --- Skeleton 기반 특성 ---
    if _HAS_SKIMAGE:
        _compute_skeleton_features(mask_bin, dist, stats)

    # --- Border touch ---
    if roi_shape is not None:
        H, W = roi_shape
        border_mask = np.zeros_like(mask_bin)
        border_mask[0, :] = 1  # top
        border_mask[-1, :] = 1  # bottom
        border_mask[:, 0] = 1  # left
        border_mask[:, -1] = 1  # right
        border_pixels = np.sum(mask_bin & border_mask)
        border_total = 2 * (H + W) - 4
        if border_total > 0:
            stats.border_touch = border_pixels / border_total

    # --- Shape mode 판정 ---
    # skeleton 있으면 tube_fit 기반, 없으면 기존 로직
    if _HAS_SKIMAGE and stats.skel_tube_fit > 0:
        if stats.skel_tube_fit >= 0.7 and stats.skel_width_cv <= 0.8:
            # 깔끔한 튜브 구조
            if stats.aspect_ratio >= 3.0:
                stats.shape_mode = "ELONGATED"
            else:
                stats.shape_mode = "COILED"
        elif stats.border_touch > 0.4:
            stats.shape_mode = "UNKNOWN"
        elif stats.skel_tube_fit < 0.5 and stats.skel_width_cv > 0.85:
            stats.shape_mode = "UNKNOWN"
        else:
            stats.shape_mode = "ELONGATED" if stats.aspect_ratio >= 3.0 else "COILED"
    else:
        # fallback: 기존 로직
        is_elongated = stats.aspect_ratio >= 3.0
        is_coiled = stats.aspect_ratio < 2.5 and stats.solidity > 0.3
        if stats.border_touch > 0.4:
            stats.shape_mode = "UNKNOWN"
        elif is_elongated:
            stats.shape_mode = "ELONGATED"
        elif is_coiled:
            stats.shape_mode = "COILED"
        else:
            stats.shape_mode = "UNKNOWN"

    # --- Shape score 계산 ---
    # 기본 점수: 1.0에서 페널티 차감
    score = 1.0

    # border_touch 페널티
    if stats.border_touch > 0.3:
        score -= min((stats.border_touch - 0.3) * 2.0, 0.5)

    # 면적 기반 페널티 (ROI 대비 너무 크면 풀마스크 의심)
    if roi_shape is not None:
        roi_area = roi_shape[0] * roi_shape[1]
        area_ratio = stats.area / roi_area
        if area_ratio > 0.3:  # ROI 30% 이상 차지
            score -= min((area_ratio - 0.3) * 1.5, 0.5)

    # --- Skeleton 기반 페널티 (핵심 판별) ---
    skel_valid = _HAS_SKIMAGE and stats.skel_length >= 10 and stats.skel_tube_fit > 0
    if skel_valid:
        # tube_fit: 깔끔한 튜브 = 0.7~1.2
        # Good median=0.98, Bad median=0.37
        if stats.skel_tube_fit < 0.5:
            # 번짐/불규칙 마스크 (skeleton은 길지만 면적이 안 맞음)
            score -= min((0.5 - stats.skel_tube_fit) * 2.0, 0.5)
        elif stats.skel_tube_fit > 1.3:
            # blob/compact 덩어리 (skeleton 대비 면적이 너무 큼)
            # tube_fit 2.47 같은 값은 완전히 blob
            score -= min((stats.skel_tube_fit - 1.3) * 0.6, 0.5)

        # width_cv: 체폭 일관성. 애벌레는 낮고, 번짐은 높음
        # Good median=0.60, Bad median=0.99
        if stats.skel_width_cv > 0.85:
            score -= min((stats.skel_width_cv - 0.85) * 1.5, 0.4)
    else:
        # fallback: 기존 두께 기반 페널티
        if roi_shape is not None:
            roi_dim = min(roi_shape)
            thick_ratio = stats.thickness_p90 / max(roi_dim, 1)
            if thick_ratio > 0.08:
                score -= min((thick_ratio - 0.08) * 3.0, 0.5)
        elif stats.thickness_p90 > 50:
            score -= min((stats.thickness_p90 - 50) / 100, 0.3)

        # p2_over_a: 너무 낮으면 뭉툭한 덩어리
        if stats.p2_over_a < 20 and stats.aspect_ratio < 2.0:
            score -= 0.2

    stats.shape_score = max(0.0, min(1.0, score))

    # --- Temporal score (연속성) ---
    if prev_stats is not None and prev_stats.area > 0:
        temp_score = 1.0

        # 면적 변화
        area_ratio = stats.area / prev_stats.area
        if area_ratio > 3.0 or area_ratio < 0.3:
            temp_score -= 0.5
        elif area_ratio > 2.0 or area_ratio < 0.5:
            temp_score -= 0.2

        # 두께 변화
        if prev_stats.thickness_med > 0:
            thick_ratio = stats.thickness_med / prev_stats.thickness_med
            if thick_ratio > 2.0 or thick_ratio < 0.5:
                temp_score -= 0.3

        stats.temporal_score = max(0.0, min(1.0, temp_score))

    return stats
