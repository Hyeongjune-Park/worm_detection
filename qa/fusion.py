# qa/fusion.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import math

from sensors.base import SensorResult
from tracking.track import TrackState


@dataclass
class FusionResult:
    """교차검증 + 융합 결과."""
    center: Tuple[float, float]                     # 채택된 fused center (full-frame)
    head: Optional[Tuple[float, float]] = None      # head 좌표
    sensor_used: str = "PRED"                       # SAM2 / TPL / KLT / PRED
    quality_score: float = 0.0                      # 0..1
    do_kalman_update: bool = False                  # KF update 수행 여부
    measurement_r: float = 25.0                     # KF update 시 관측 노이즈
    state_hint: Optional[TrackState] = None         # 상태머신 힌트
    debug: Dict[str, float] = field(default_factory=dict)


def _dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    if a is None or b is None:
        return float("inf")
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def fuse(
    pred_center: Tuple[float, float],
    sam2: Optional[SensorResult],
    tpl: Optional[SensorResult],
    klt: Optional[SensorResult],
    cfg: Dict[str, Any],
    roi_size: float = 384.0,
    sam2_r: float = 10.0,
    tpl_r: float = 25.0,
    klt_r: float = 25.0,
    is_merged: bool = False,
    prev_area: Optional[int] = None,  # 이전 프레임 SAM2 면적 (연속성 검사용)
) -> FusionResult:
    """
    다중 센서 교차검증 + 융합.

    Fix 2: 거리 임계값을 ROI 크기 대비 비율로 정규화.
    Fix 3: is_merged=True일 때 SAM2 불신, Template/KLT 우선.

    Case A (Good SAM2): d_norm small AND d_tpl_norm small AND border_touch low → SAM2 채택
    Case B (SAM2 suspicious or MERGED): → template 또는 KLT 사용
    Case C (All weak): → predict only
    """
    qa = cfg.get("qa", {})
    # 비율 기반 임계값 (0..1, ROI diagonal 대비)
    dist_pred_ratio = float(qa.get("dist_pred_ratio", 0.12))
    dist_tpl_ratio = float(qa.get("dist_tpl_ratio", 0.12))
    dist_klt_ratio = float(qa.get("dist_klt_ratio", 0.12))
    border_th = float(qa.get("border_touch_thresh", 0.3))

    # ROI diagonal 기반 정규화 기준
    roi_diag = roi_size * math.sqrt(2)
    dist_pred_th = dist_pred_ratio * roi_diag
    dist_tpl_th = dist_tpl_ratio * roi_diag
    dist_klt_th = dist_klt_ratio * roi_diag

    # 센서별 center
    c_sam2 = sam2.center if sam2 is not None else None
    c_tpl = tpl.center if tpl is not None else None
    c_klt = klt.center if klt is not None else None

    # 거리 계산
    d_pred = _dist(c_sam2, pred_center)
    d_tpl = _dist(c_sam2, c_tpl) if c_sam2 is not None and c_tpl is not None else float("inf")
    d_klt = _dist(c_sam2, c_klt) if c_sam2 is not None and c_klt is not None else float("inf")
    border_touch = sam2.border_touch if sam2 is not None else 0.0

    # 정규화 비율도 debug에 포함
    debug = {
        "d_pred": d_pred,
        "d_tpl": d_tpl,
        "d_klt": d_klt,
        "d_pred_norm": d_pred / max(1.0, roi_diag),
        "d_tpl_norm": d_tpl / max(1.0, roi_diag) if d_tpl < float("inf") else -1.0,
        "border_touch": border_touch,
        "roi_size": roi_size,
    }

    # --- Fix 3: MERGED 상태 → SAM2 불신 (붙어 있으면 하나로 나옴) ---
    sam2_trustable = not is_merged

    # --- 면적 연속성 페널티 계산 (소프트 의심 신호) ---
    # 하드 게이트가 아닌 quality 감점으로만 사용
    area_penalty = 0.0
    curr_area = sam2.area if sam2 is not None else None
    if prev_area is not None and curr_area is not None and prev_area > 0:
        area_ratio = curr_area / prev_area
        # 0.5배 ~ 2배 범위를 벗어나면 페널티 증가
        if area_ratio < 0.5:
            area_penalty = min((0.5 - area_ratio) * 0.8, 0.4)  # 너무 작으면 감점 (최대 0.4)
        elif area_ratio > 2.0:
            area_penalty = min((area_ratio - 2.0) * 0.4, 0.4)  # 너무 크면 감점 (최대 0.4)
    debug["area_penalty"] = area_penalty
    debug["curr_area"] = curr_area if curr_area is not None else -1
    debug["prev_area"] = prev_area if prev_area is not None else -1

    # --- Case A: SAM2 신뢰 (MERGED가 아닐 때만) ---
    # 2-of-3 교차검증: SAM2 + (pred 필수) + (tpl OR klt 중 하나 이상 동의 또는 둘 다 None)
    if c_sam2 is not None and sam2_trustable:
        pred_ok = (d_pred <= dist_pred_th)
        tpl_ok = (c_tpl is None) or (d_tpl <= dist_tpl_th)
        klt_ok = (c_klt is None) or (d_klt <= dist_klt_th)
        border_ok = (border_touch <= border_th)

        # tpl/klt 중 하나라도 동의하면 OK (둘 다 None이면 SAM2+pred만으로 통과)
        secondary_ok = tpl_ok or klt_ok

        if pred_ok and secondary_ok and border_ok:
            # 기본 confidence에서 면적 연속성 페널티 적용
            base_quality = min(1.0, sam2.confidence if sam2 else 0.5)
            adjusted_quality = max(0.0, base_quality - area_penalty)

            return FusionResult(
                center=c_sam2,
                sensor_used="SAM2",
                quality_score=adjusted_quality,
                do_kalman_update=True,
                measurement_r=sam2_r,
                state_hint=TrackState.ACTIVE,
                debug=debug,
            )

    # --- Case B: SAM2 불신 또는 MERGED → template/KLT 우선 ---
    if c_tpl is not None:
        tpl_pred_dist = _dist(c_tpl, pred_center)
        if tpl_pred_dist <= dist_pred_th:
            return FusionResult(
                center=c_tpl,
                sensor_used="TPL",
                quality_score=tpl.confidence if tpl else 0.5,
                do_kalman_update=True,
                measurement_r=tpl_r,
                state_hint=TrackState.UNCERTAIN,
                debug=debug,
            )

    if c_klt is not None:
        klt_pred_dist = _dist(c_klt, pred_center)
        if klt_pred_dist <= dist_pred_th:
            return FusionResult(
                center=c_klt,
                sensor_used="KLT",
                quality_score=klt.confidence if klt else 0.5,
                do_kalman_update=True,
                measurement_r=klt_r,
                state_hint=TrackState.UNCERTAIN,
                debug=debug,
            )

    # --- Case C: 모두 불신 → predict only ---
    return FusionResult(
        center=pred_center,
        sensor_used="PRED",
        quality_score=0.0,
        do_kalman_update=False,
        measurement_r=25.0,
        state_hint=TrackState.OCCLUDED,
        debug=debug,
    )
