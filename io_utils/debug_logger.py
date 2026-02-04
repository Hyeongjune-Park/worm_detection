# io_utils/debug_logger.py
"""구조화된 디버그 로깅 시스템."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from qa.shape_analyzer import ShapeStats


@dataclass
class FrameDebugRecord:
    """프레임별 디버그 레코드."""
    # 기본 정보
    frame_idx: int = 0
    t_sec: float = 0.0
    track_id: int = 0

    # 상태/결정
    state: str = ""
    sensor_used: str = ""
    do_kalman_update: bool = False
    measurement_r: float = 0.0
    quality_score: float = 0.0

    # 좌표 (full-frame 기준)
    pred_center: Optional[Tuple[float, float]] = None
    sam2_center: Optional[Tuple[float, float]] = None
    tpl_center: Optional[Tuple[float, float]] = None
    klt_center: Optional[Tuple[float, float]] = None
    chosen_center: Optional[Tuple[float, float]] = None

    # 거리
    d_pred_sam2: float = -1.0
    d_pred_tpl: float = -1.0
    d_pred_klt: float = -1.0
    d_sam2_tpl: float = -1.0
    d_sam2_klt: float = -1.0

    # ROI / bbox
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    roi_scale: float = 1.0
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    bbox_update: str = "HOLD"  # HOLD / UPDATE

    # SAM2 마스크 품질/형태
    shape_stats: Optional[ShapeStats] = None
    area_ratio_prev: float = 1.0

    # 이상징후 플래그
    reason_flags: List[str] = field(default_factory=list)

    # Fusion debug 정보
    fusion_debug: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON/CSV용 딕셔너리 변환."""
        d = {
            "frame_idx": self.frame_idx,
            "t_sec": round(self.t_sec, 4),
            "track_id": self.track_id,
            "state": self.state,
            "sensor_used": self.sensor_used,
            "do_kalman_update": self.do_kalman_update,
            "measurement_r": round(self.measurement_r, 2),
            "quality_score": round(self.quality_score, 3),
            # 좌표
            "pred_x": round(self.pred_center[0], 1) if self.pred_center else None,
            "pred_y": round(self.pred_center[1], 1) if self.pred_center else None,
            "sam2_x": round(self.sam2_center[0], 1) if self.sam2_center else None,
            "sam2_y": round(self.sam2_center[1], 1) if self.sam2_center else None,
            "tpl_x": round(self.tpl_center[0], 1) if self.tpl_center else None,
            "tpl_y": round(self.tpl_center[1], 1) if self.tpl_center else None,
            "klt_x": round(self.klt_center[0], 1) if self.klt_center else None,
            "klt_y": round(self.klt_center[1], 1) if self.klt_center else None,
            "chosen_x": round(self.chosen_center[0], 1) if self.chosen_center else None,
            "chosen_y": round(self.chosen_center[1], 1) if self.chosen_center else None,
            # 거리
            "d_pred_sam2": round(self.d_pred_sam2, 1) if self.d_pred_sam2 >= 0 else None,
            "d_pred_tpl": round(self.d_pred_tpl, 1) if self.d_pred_tpl >= 0 else None,
            "d_pred_klt": round(self.d_pred_klt, 1) if self.d_pred_klt >= 0 else None,
            "d_sam2_tpl": round(self.d_sam2_tpl, 1) if self.d_sam2_tpl >= 0 else None,
            "d_sam2_klt": round(self.d_sam2_klt, 1) if self.d_sam2_klt >= 0 else None,
            # ROI / bbox
            "roi_x0": self.roi_xyxy[0] if self.roi_xyxy else None,
            "roi_y0": self.roi_xyxy[1] if self.roi_xyxy else None,
            "roi_x1": self.roi_xyxy[2] if self.roi_xyxy else None,
            "roi_y1": self.roi_xyxy[3] if self.roi_xyxy else None,
            "roi_scale": round(self.roi_scale, 2),
            "bbox_x0": self.bbox_xyxy[0] if self.bbox_xyxy else None,
            "bbox_y0": self.bbox_xyxy[1] if self.bbox_xyxy else None,
            "bbox_x1": self.bbox_xyxy[2] if self.bbox_xyxy else None,
            "bbox_y1": self.bbox_xyxy[3] if self.bbox_xyxy else None,
            "bbox_update": self.bbox_update,
            # 면적
            "area_ratio_prev": round(self.area_ratio_prev, 2),
            # 플래그
            "reason_flags": ",".join(self.reason_flags) if self.reason_flags else "",
        }

        # Shape stats 추가
        if self.shape_stats:
            ss = self.shape_stats.to_dict()
            for k, v in ss.items():
                d[f"shape_{k}"] = v

        # Fusion debug 추가
        for k, v in self.fusion_debug.items():
            d[f"fusion_{k}"] = round(v, 3) if isinstance(v, float) else v

        return d


class DebugLogger:
    """디버그 로거 — JSONL/CSV/events.md/summary 출력."""

    def __init__(self, output_dir: str, fps: float = 25.0):
        self.output_dir = output_dir
        self.fps = fps
        self.records: List[FrameDebugRecord] = []
        self.events: List[Dict[str, Any]] = []
        self.toggles_dict: Dict[str, bool] = {}

        # 이전 상태 추적 (연속성/이벤트 감지)
        self.prev_states: Dict[int, str] = {}  # track_id -> state
        self.prev_centers: Dict[int, Tuple[float, float]] = {}  # track_id -> center
        self.prev_areas: Dict[int, int] = {}  # track_id -> area

        os.makedirs(output_dir, exist_ok=True)

    def record_toggles(self, toggles) -> None:
        """실행 시작 시 호출 — 토글 상태 기록."""
        self.toggles_dict = toggles.to_dict()

    def add_record(self, record: FrameDebugRecord):
        """프레임 레코드 추가 + 이상징후 자동 감지."""
        # 이상징후 플래그 자동 생성
        self._detect_anomalies(record)

        # 상태 전이 이벤트
        self._detect_state_change(record)

        self.records.append(record)

        # 이전 상태 업데이트
        self.prev_states[record.track_id] = record.state
        if record.chosen_center:
            self.prev_centers[record.track_id] = record.chosen_center
        if record.shape_stats and record.shape_stats.area > 0:
            self.prev_areas[record.track_id] = record.shape_stats.area

    def _detect_anomalies(self, record: FrameDebugRecord):
        """이상징후 플래그 자동 감지."""
        flags = []

        # AREA_BLOWUP: 면적 급변
        if record.area_ratio_prev > 3.0:
            flags.append(f"AREA_BLOWUP_UP({record.area_ratio_prev:.1f}x)")
        elif record.area_ratio_prev < 0.3 and record.area_ratio_prev > 0:
            flags.append(f"AREA_BLOWUP_DOWN({record.area_ratio_prev:.2f}x)")

        # BORDER_TOUCH_HIGH
        if record.shape_stats and record.shape_stats.border_touch > 0.35:
            flags.append(f"BORDER_TOUCH_HIGH({record.shape_stats.border_touch:.2f})")

        # THICKNESS_BLOWUP
        if record.shape_stats and record.shape_stats.thickness_p90 > 60:
            flags.append(f"THICKNESS_HIGH({record.shape_stats.thickness_p90:.0f})")

        # CENTER_JUMP_OUTLIER
        prev_center = self.prev_centers.get(record.track_id)
        if prev_center and record.chosen_center:
            jump = ((record.chosen_center[0] - prev_center[0]) ** 2 +
                    (record.chosen_center[1] - prev_center[1]) ** 2) ** 0.5
            if jump > 100:  # 100px 이상 점프
                flags.append(f"CENTER_JUMP({jump:.0f}px)")

        # CONSENSUS_BREAK: SAM2와 TPL/KLT 모두 멀리 떨어짐
        if record.d_pred_sam2 > 50 and record.d_pred_tpl > 50:
            flags.append("CONSENSUS_BREAK")

        # LOW_SHAPE_SCORE
        if record.shape_stats and record.shape_stats.shape_score < 0.5:
            flags.append(f"LOW_SHAPE_SCORE({record.shape_stats.shape_score:.2f})")

        # ROI_EXPAND_DURING_UNCERTAIN
        if record.state in ("UNCERTAIN", "OCCLUDED") and record.roi_scale > 1.2:
            flags.append(f"ROI_EXPAND_UNCERTAIN({record.roi_scale:.1f}x)")

        record.reason_flags = flags

    def _detect_state_change(self, record: FrameDebugRecord):
        """상태 전이 이벤트 감지."""
        prev_state = self.prev_states.get(record.track_id, "ACTIVE")
        if record.state != prev_state:
            event = {
                "frame_idx": record.frame_idx,
                "t_sec": record.t_sec,
                "track_id": record.track_id,
                "event": "STATE_CHANGE",
                "from": prev_state,
                "to": record.state,
                "sensor": record.sensor_used,
                "quality": record.quality_score,
                "flags": record.reason_flags.copy(),
            }
            self.events.append(event)

    def save_all(self):
        """모든 로그 파일 저장."""
        self._save_jsonl()
        self._save_csv()
        self._save_events_md()
        self._save_summary()

    def _save_jsonl(self):
        """debug_frames.jsonl 저장."""
        path = os.path.join(self.output_dir, "debug_frames.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    def _save_csv(self):
        """debug_frames.csv 저장."""
        if not self.records:
            return

        path = os.path.join(self.output_dir, "debug_frames.csv")

        # 모든 레코드에서 발생할 수 있는 모든 키 수집
        all_keys = set()
        for rec in self.records:
            all_keys.update(rec.to_dict().keys())
        fieldnames = sorted(all_keys)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for rec in self.records:
                row = rec.to_dict()
                # 누락된 필드는 빈 값으로
                for k in fieldnames:
                    if k not in row:
                        row[k] = None
                writer.writerow(row)

    def _save_events_md(self):
        """events.md 저장 — 사람이 읽기 쉬운 이벤트 요약."""
        path = os.path.join(self.output_dir, "events.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 이벤트 로그\n\n")

            # 상태 전이
            f.write("## 상태 전이\n\n")
            for ev in self.events:
                if ev["event"] == "STATE_CHANGE":
                    flags_str = " + ".join(ev["flags"]) if ev["flags"] else ""
                    f.write(f"- F{ev['frame_idx']:04d} T{ev['track_id']}: "
                            f"{ev['from']} -> {ev['to']} "
                            f"[{ev['sensor']}, q={ev['quality']:.2f}]")
                    if flags_str:
                        f.write(f" {{{flags_str}}}")
                    f.write("\n")

            # 이상징후 요약
            f.write("\n## 이상징후 플래그 발생 프레임\n\n")
            flagged = [r for r in self.records if r.reason_flags]
            for rec in flagged[:50]:  # 상위 50개만
                f.write(f"- F{rec.frame_idx:04d} T{rec.track_id}: "
                        f"{', '.join(rec.reason_flags)}\n")
            if len(flagged) > 50:
                f.write(f"\n... 외 {len(flagged) - 50}개\n")

    def _save_summary(self):
        """run_summary.json 저장."""
        path = os.path.join(self.output_dir, "run_summary.json")

        # 통계 계산
        total_frames = len(set(r.frame_idx for r in self.records))
        total_records = len(self.records)

        state_counts = {}
        sensor_counts = {}
        flag_counts = {}

        for rec in self.records:
            state_counts[rec.state] = state_counts.get(rec.state, 0) + 1
            sensor_counts[rec.sensor_used] = sensor_counts.get(rec.sensor_used, 0) + 1
            for flag in rec.reason_flags:
                # 플래그에서 숫자 부분 제거 (예: AREA_BLOWUP_UP(4.5x) -> AREA_BLOWUP_UP)
                flag_key = flag.split("(")[0]
                flag_counts[flag_key] = flag_counts.get(flag_key, 0) + 1

        summary = {
            "toggles": self.toggles_dict,
            "total_frames": total_frames,
            "total_records": total_records,
            "state_distribution": state_counts,
            "sensor_distribution": sensor_counts,
            "anomaly_counts": flag_counts,
            "state_changes": len(self.events),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def save_debug_snapshot(
    frame_bgr: np.ndarray,
    record: FrameDebugRecord,
    output_dir: str,
):
    """이벤트 프레임 스냅샷 저장 (오버레이 포함)."""
    snapshots_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    img = frame_bgr.copy()

    # 텍스트 오버레이
    lines = [
        f"F{record.frame_idx:04d} T{record.track_id} {record.state}",
        f"sensor={record.sensor_used} q={record.quality_score:.2f}",
    ]

    if record.shape_stats:
        ss = record.shape_stats
        lines.append(f"area={ss.area} ratio={record.area_ratio_prev:.2f}")
        lines.append(f"border={ss.border_touch:.2f} shape={ss.shape_score:.2f}")
        lines.append(f"thick_med={ss.thickness_med:.1f} p90={ss.thickness_p90:.1f}")
        lines.append(f"mode={ss.shape_mode}")

    if record.reason_flags:
        lines.append(f"FLAGS: {', '.join(record.reason_flags[:3])}")

    y = 30
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        y += 18

    filename = f"frame_{record.frame_idx:04d}_track_{record.track_id}.png"
    cv2.imwrite(os.path.join(snapshots_dir, filename), img)
