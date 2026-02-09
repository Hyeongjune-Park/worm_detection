# qa/fusion.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import math

from sensors.base import SensorResult
from tracking.track import TrackState

if TYPE_CHECKING:
    from config_toggles import FeatureToggles


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
    prev_area: Optional[int] = None,
    track_state: Optional[TrackState] = None,
    toggles: Optional["FeatureToggles"] = None,
    shape_score: Optional[float] = None,
) -> FusionResult:
    """
    다중 센서 교차검증 + 융합.

    Fix 2: 거리 임계값을 ROI 크기 대비 비율로 정규화.
    Fix 3: is_merged=True일 때 SAM2 불신, Template/KLT 우선.

    Case A (Good SAM2): SAM2 채택 (토글에 따라 조건 다름)
    Case B (SAM2 suspicious or MERGED): → template 또는 KLT 사용
    Case C (All weak): → predict only
    """
    # 토글 기본값 (None이면 dataclass 기본값 사용)
    if toggles is None:
        from config_toggles import FeatureToggles
        toggles = FeatureToggles()

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

    # --- [F4] 면적 연속성 페널티 계산 ---
    area_penalty = 0.0
    curr_area = sam2.area if sam2 is not None else None
    if toggles.area_continuity_penalty:
        if prev_area is not None and curr_area is not None and prev_area > 0:
            area_ratio = curr_area / prev_area
            if area_ratio < 0.5:
                area_penalty = min((0.5 - area_ratio) * 0.8, 0.4)
            elif area_ratio > 2.0:
                area_penalty = min((area_ratio - 2.0) * 0.4, 0.4)
    debug["area_penalty"] = area_penalty
    debug["curr_area"] = curr_area if curr_area is not None else -1
    debug["prev_area"] = prev_area if prev_area is not None else -1
    debug["shape_score"] = shape_score if shape_score is not None else -1.0

    # --- 모드 판단 ---
    is_reacquire = track_state in (TrackState.UNCERTAIN, TrackState.OCCLUDED)
    debug["is_reacquire"] = 1.0 if is_reacquire else 0.0

    # --- [F3] 센서 합의(consensus) 계산 ---
    if toggles.strict_consensus:
        # 수정: None이면 False (센서가 존재하고 SAM2와 가까워야 True)
        tpl_agrees = (c_tpl is not None) and (d_tpl <= dist_tpl_th)
        klt_agrees = (c_klt is not None) and (d_klt <= dist_klt_th)
        has_consensus = tpl_agrees or klt_agrees
    else:
        # BASE: None이면 ok (원래 동작)
        tpl_agrees = (c_tpl is None) or (d_tpl <= dist_tpl_th)
        klt_agrees = (c_klt is None) or (d_klt <= dist_klt_th)
        has_consensus = tpl_agrees or klt_agrees
    debug["tpl_agrees"] = 1.0 if tpl_agrees else 0.0
    debug["klt_agrees"] = 1.0 if klt_agrees else 0.0
    debug["has_consensus"] = 1.0 if has_consensus else 0.0

    # --- [SH1] shape 기반 전체 센서 거부 (심각한 번짐 시 predict-only) ---
    if toggles.shape_quality_gate and shape_score is not None and shape_score < 0.6:
        debug["shape_reject"] = 1.0
        return FusionResult(
            center=pred_center,
            sensor_used="PRED",
            quality_score=0.0,
            do_kalman_update=False,
            measurement_r=25.0,
            state_hint=TrackState.OCCLUDED,
            debug=debug,
        )
    debug["shape_reject"] = 0.0

    # --- Case A: SAM2 채택 ---
    if c_sam2 is not None and sam2_trustable:
        border_ok = (border_touch <= border_th)

        # 면적 연속성 체크 (하드 게이트용)
        shape_ok = True
        area_check_available = (prev_area is not None and curr_area is not None and prev_area > 0)
        if area_check_available:
            area_ratio = curr_area / prev_area
            shape_ok = 0.3 < area_ratio < 3.0

        debug["border_ok"] = 1.0 if border_ok else 0.0
        debug["shape_ok"] = 1.0 if shape_ok else 0.0
        debug["area_check_available"] = 1.0 if area_check_available else 0.0

        # --- [F1] ACTIVE/REACQUIRE 모드 분리 ---
        if toggles.active_reacquire_split:
            result = _fuse_case_a_split(
                c_sam2, sam2, pred_center, d_pred, roi_diag,
                dist_pred_ratio, dist_pred_th, sam2_r,
                border_ok, shape_ok, area_check_available,
                has_consensus, area_penalty, is_reacquire,
                qa, debug, toggles, shape_score,
            )
        else:
            result = _fuse_case_a_base(
                c_sam2, sam2, pred_center, d_pred, roi_diag,
                dist_pred_ratio, dist_pred_th, sam2_r,
                border_ok, has_consensus,
                area_penalty, qa, debug, toggles,
            )
        if result is not None:
            return result

    # --- Case B: SAM2 불신 또는 MERGED → template/KLT 우선 ---
    result = _fuse_case_b(
        c_tpl, c_klt, tpl, klt, pred_center,
        roi_diag, dist_pred_ratio, dist_pred_th,
        tpl_r, klt_r, debug, toggles,
    )
    if result is not None:
        return result

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


def _fuse_case_a_split(
    c_sam2, sam2, pred_center, d_pred, roi_diag,
    dist_pred_ratio, dist_pred_th, sam2_r,
    border_ok, shape_ok, area_check_available,
    has_consensus, area_penalty, is_reacquire,
    qa, debug, toggles, shape_score=None,
) -> Optional[FusionResult]:
    """[F1=ON] ACTIVE/REACQUIRE 모드 분리 SAM2 채택 로직."""

    # [SH1] shape_score 기반 quality 페널티 (< 0.6 거부는 fuse() 최상단에서 처리)
    shape_penalty = 0.0
    if toggles.shape_quality_gate and shape_score is not None and shape_score < 0.8:
        shape_penalty = min((0.8 - shape_score) * 1.0, 0.3)
    debug["shape_penalty"] = shape_penalty

    if not is_reacquire:
        # --- ACTIVE 모드: 엄격 (consensus 필수, area 없으면 소프트 페널티) ---
        can_adopt = border_ok and shape_ok and has_consensus
        if not can_adopt:
            return None

        base_quality = min(1.0, sam2.confidence if sam2 else 0.5)
        adjusted_quality = max(0.0, base_quality - area_penalty - shape_penalty)

        # area_check 미가용 시 소프트 페널티 (순환 의존 방지)
        if not area_check_available:
            adjusted_quality = max(0.0, adjusted_quality - 0.15)

        # [F2] pred 거리 페널티
        pred_penalty, effective_r = _apply_pred_penalty(
            d_pred, roi_diag, dist_pred_ratio, sam2_r, toggles.soft_pred_penalty,
        )
        if not toggles.soft_pred_penalty and d_pred > dist_pred_ratio * roi_diag:
            # F2 OFF: pred 거리 하드 게이트 실패 → SAM2 거부
            return None
        adjusted_quality = max(0.0, adjusted_quality - pred_penalty)

        quality_uncertain_th = float(qa.get("quality_uncertain_threshold", 0.7))
        hint = TrackState.ACTIVE if adjusted_quality >= quality_uncertain_th else TrackState.UNCERTAIN

        # quality 기반 R 스케일링 (저품질 update 드리프트 억제)
        effective_r = _quality_r_scale(effective_r, adjusted_quality)

        debug["pred_penalty"] = pred_penalty
        debug["effective_r"] = effective_r
        debug["mode"] = 0.0  # ACTIVE

        return FusionResult(
            center=c_sam2,
            sensor_used="SAM2",
            quality_score=adjusted_quality,
            do_kalman_update=True,
            measurement_r=effective_r,
            state_hint=hint,
            debug=debug,
        )

    else:
        # --- REACQUIRE 모드: 관대 (consensus 없어도 채택 가능, 단 R 크게) ---
        can_adopt = border_ok and shape_ok
        if not can_adopt:
            return None

        base_quality = min(1.0, sam2.confidence if sam2 else 0.5)
        adjusted_quality = max(0.0, base_quality - area_penalty - shape_penalty)

        # consensus 없으면 추가 페널티
        if not has_consensus:
            adjusted_quality = max(0.0, adjusted_quality - 0.2)
            consensus_r_mult = 2.0
        else:
            consensus_r_mult = 1.0

        # area_check 없으면 추가 페널티
        if not area_check_available:
            adjusted_quality = max(0.0, adjusted_quality - 0.1)

        # [F2] pred 거리 페널티
        pred_penalty, r_multiplier_base = _apply_pred_penalty(
            d_pred, roi_diag, dist_pred_ratio, sam2_r, toggles.soft_pred_penalty,
        )
        if not toggles.soft_pred_penalty and d_pred > dist_pred_ratio * roi_diag:
            return None
        adjusted_quality = max(0.0, adjusted_quality - pred_penalty)
        effective_r = r_multiplier_base * consensus_r_mult

        # [F5] REACQUIRE → ACTIVE 복귀
        if toggles.reacquire_active_recovery:
            quality_active_th = float(qa.get("quality_active_threshold", 0.8))
            if has_consensus and adjusted_quality >= quality_active_th:
                hint = TrackState.ACTIVE
            else:
                hint = TrackState.UNCERTAIN
        else:
            hint = TrackState.UNCERTAIN

        # quality 기반 R 스케일링 (저품질 update 드리프트 억제)
        effective_r = _quality_r_scale(effective_r, adjusted_quality)

        debug["pred_penalty"] = pred_penalty
        debug["effective_r"] = effective_r
        debug["consensus_r_mult"] = consensus_r_mult
        debug["mode"] = 1.0  # REACQUIRE

        return FusionResult(
            center=c_sam2,
            sensor_used="SAM2",
            quality_score=adjusted_quality,
            do_kalman_update=True,
            measurement_r=effective_r,
            state_hint=hint,
            debug=debug,
        )


def _fuse_case_a_base(
    c_sam2, sam2, pred_center, d_pred, roi_diag,
    dist_pred_ratio, dist_pred_th, sam2_r,
    border_ok, has_consensus,
    area_penalty, qa, debug, toggles,
) -> Optional[FusionResult]:
    """[F1=OFF] BASE 단일 경로 SAM2 채택 로직.

    원래 로직: pred_ok + secondary_ok + border_ok → SAM2 채택.
    """
    # [F2] pred 거리 체크
    if toggles.soft_pred_penalty:
        # 소프트: 거리 멀어도 채택하되 quality 감점 + R 증가
        pred_penalty, effective_r = _apply_pred_penalty(
            d_pred, roi_diag, dist_pred_ratio, sam2_r, True,
        )
        pred_ok = True  # 소프트 모드에서는 항상 통과
    else:
        # BASE: 하드 게이트
        pred_ok = (d_pred <= dist_pred_th)
        pred_penalty = 0.0
        effective_r = sam2_r

    # secondary_ok = has_consensus (F3에 의해 계산됨)
    secondary_ok = has_consensus

    if not (pred_ok and secondary_ok and border_ok):
        return None

    base_quality = min(1.0, sam2.confidence if sam2 else 0.5)
    adjusted_quality = max(0.0, base_quality - area_penalty - pred_penalty)

    hint = TrackState.ACTIVE

    # quality 기반 R 스케일링 (저품질 update 드리프트 억제)
    effective_r = _quality_r_scale(effective_r, adjusted_quality)

    debug["pred_penalty"] = pred_penalty
    debug["effective_r"] = effective_r
    debug["mode"] = -1.0  # BASE (no split)

    return FusionResult(
        center=c_sam2,
        sensor_used="SAM2",
        quality_score=adjusted_quality,
        do_kalman_update=True,
        measurement_r=effective_r,
        state_hint=hint,
        debug=debug,
    )


def _apply_pred_penalty(
    d_pred: float,
    roi_diag: float,
    dist_pred_ratio: float,
    base_r: float,
    soft: bool,
) -> Tuple[float, float]:
    """pred 거리 기반 페널티 계산.

    Returns: (quality_penalty, effective_r)
    """
    d_pred_norm = d_pred / max(1.0, roi_diag)
    if soft and d_pred_norm > dist_pred_ratio:
        excess_ratio = (d_pred_norm - dist_pred_ratio) / dist_pred_ratio
        pred_penalty = min(excess_ratio * 0.3, 0.4)
        r_multiplier = min(1.0 + excess_ratio * 2.0, 4.0)
        effective_r = base_r * r_multiplier
    else:
        pred_penalty = 0.0
        effective_r = base_r
    return pred_penalty, effective_r


def _quality_r_scale(base_r: float, quality: float) -> float:
    """quality 기반 measurement_r 스케일링.

    quality >= 0.8 → R 그대로 (고신뢰)
    0.6 <= quality < 0.8 → R을 최대 3배까지 증가 (저신뢰 update 드리프트 억제)
    quality < 0.6 → R × 5 (거의 무시)
    """
    if quality >= 0.8:
        return base_r
    if quality >= 0.6:
        scale = 1.0 + 2.0 * (0.8 - quality) / 0.2  # 1.0 ~ 3.0
        return base_r * scale
    return base_r * 5.0


def _fuse_case_b(
    c_tpl, c_klt, tpl, klt, pred_center,
    roi_diag, dist_pred_ratio, dist_pred_th,
    tpl_r, klt_r, debug, toggles,
) -> Optional[FusionResult]:
    """Case B: SAM2 불신 → template/KLT 우선."""

    # --- Template ---
    if c_tpl is not None:
        tpl_pred_dist = _dist(c_tpl, pred_center)

        if toggles.soft_pred_penalty:
            # [F2=ON] 소프트 페널티: 거리 멀어도 채택하되 R 증가
            tpl_pred_norm = tpl_pred_dist / max(1.0, roi_diag)
            if tpl_pred_norm > dist_pred_ratio:
                excess_ratio = (tpl_pred_norm - dist_pred_ratio) / dist_pred_ratio
                r_multiplier = min(1.0 + excess_ratio * 2.0, 4.0)
                effective_tpl_r = tpl_r * r_multiplier
                tpl_quality = max(0.0, (tpl.confidence if tpl else 0.5) - min(excess_ratio * 0.3, 0.4))
            else:
                effective_tpl_r = tpl_r
                tpl_quality = tpl.confidence if tpl else 0.5
            return FusionResult(
                center=c_tpl,
                sensor_used="TPL",
                quality_score=tpl_quality,
                do_kalman_update=True,
                measurement_r=effective_tpl_r,
                state_hint=TrackState.UNCERTAIN,
                debug=debug,
            )
        else:
            # [F2=OFF] BASE: 하드 게이트
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
            # 하드 게이트 실패 → KLT로 진행

    # --- KLT ---
    if c_klt is not None:
        klt_pred_dist = _dist(c_klt, pred_center)

        if toggles.soft_pred_penalty:
            # [F2=ON] 소프트 페널티
            klt_pred_norm = klt_pred_dist / max(1.0, roi_diag)
            if klt_pred_norm > dist_pred_ratio:
                excess_ratio = (klt_pred_norm - dist_pred_ratio) / dist_pred_ratio
                r_multiplier = min(1.0 + excess_ratio * 2.0, 4.0)
                effective_klt_r = klt_r * r_multiplier
                klt_quality = max(0.0, (klt.confidence if klt else 0.5) - min(excess_ratio * 0.3, 0.4))
            else:
                effective_klt_r = klt_r
                klt_quality = klt.confidence if klt else 0.5
            return FusionResult(
                center=c_klt,
                sensor_used="KLT",
                quality_score=klt_quality,
                do_kalman_update=True,
                measurement_r=effective_klt_r,
                state_hint=TrackState.UNCERTAIN,
                debug=debug,
            )
        else:
            # [F2=OFF] BASE: 하드 게이트
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

    return None
