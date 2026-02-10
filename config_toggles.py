# config_toggles.py — Feature Toggle 관리
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass(frozen=True)
class FeatureToggles:
    """불변 토글 상태. 초기화 시 한 번 읽고 이후 변경 불가."""

    # --- Fusion (qa/fusion.py) ---
    active_reacquire_split: bool = True       # F1: ACTIVE/REACQUIRE 모드 분리
    soft_pred_penalty: bool = False           # F2: pred 거리 소프트 페널티
    strict_consensus: bool = True             # F3: 엄격한 consensus (버그수정)
    area_continuity_penalty: bool = False     # F4: 면적 연속성 감점
    reacquire_active_recovery: bool = True    # F5: REACQUIRE→ACTIVE 복귀

    # --- Pipeline / ROI (pipeline/runner.py) ---
    speed_roi_expansion: bool = False         # P1: 속도 기반 ROI 확장
    state_roi_expansion: bool = False         # P2: 상태 기반 ROI 1.25x
    expansion_cooldown: bool = False          # P3: 확장 쿨다운 5프레임
    sam2_bbox_sync: bool = False              # P4: SAM2 bbox 독립 동기화
    reacquire_box_expansion: bool = False     # P5: REACQUIRE 시 box prompt 확장

    # --- Sensor ---
    composite_mask_selection: bool = True     # S1: 복합 마스크 선택
    template_update_gating: bool = True       # S2: TPL 업데이트 품질 게이팅
    klt_divergence_reinit: bool = True        # S3: KLT 발산 시 재초기화
    sam2_mask_caching: bool = True            # S4: SAM2 결과 캐싱

    # --- Kalman ---
    velocity_decay_on_predict: bool = True    # K1: predict-only 속도 감쇠
    innovation_gating: bool = False          # K2: Mahalanobis innovation gating (outlier rejection)

    # --- Shape Quality ---
    shape_quality_gate: bool = True              # SH1: ShapeStats 기반 bbox/cache/quality 소프트 게이트

    # --- Mask Cleanup ---
    mask_cleanup: bool = False                   # MC1: 번짐 감지 시 CCF+Expected Region 정리

    # --- State Machine ---
    occluded_to_uncertain_recovery: bool = False  # SM1: OCCLUDED→UNCERTAIN 복귀

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FeatureToggles":
        """config.yaml의 toggles 섹션에서 생성. 없는 키는 기본값 사용."""
        t = cfg.get("toggles", {})
        if t is None:
            t = {}
        valid_fields = {f.name for f in fields(cls)}
        kwargs = {k: bool(v) for k, v in t.items() if k in valid_fields}
        instance = cls(**kwargs)
        instance._warn_dependencies()
        return instance

    def _warn_dependencies(self) -> None:
        """토글 간 의존성 위반 시 경고 출력."""
        if self.reacquire_active_recovery and not self.active_reacquire_split:
            print("[WARN] reacquire_active_recovery는 active_reacquire_split=ON이어야 효과 있음")
        if self.expansion_cooldown and not (self.speed_roi_expansion or self.state_roi_expansion):
            print("[WARN] expansion_cooldown은 speed_roi_expansion 또는 state_roi_expansion=ON이어야 효과 있음")
        if self.area_continuity_penalty and not self.sam2_mask_caching:
            print("[WARN] area_continuity_penalty는 sam2_mask_caching=ON이어야 prev_area 데이터 사용 가능")
        if self.reacquire_box_expansion and not self.state_roi_expansion:
            print("[WARN] reacquire_box_expansion은 state_roi_expansion=ON이어야 확장값이 생성됨")

    def to_dict(self) -> Dict[str, bool]:
        """전체 토글 상태를 딕셔너리로 반환 (로깅용)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        """사람이 읽기 좋은 ON/OFF 요약."""
        lines = []
        for f in fields(self):
            v = getattr(self, f.name)
            marker = "ON " if v else "OFF"
            lines.append(f"  [{marker}] {f.name}")
        return "Feature Toggles:\n" + "\n".join(lines)
