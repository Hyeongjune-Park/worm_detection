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


@dataclass
class TrackBaseline:
    """트랙별 첫 프레임 SAM2 기준값 (이상 감지용)."""
    area: int = 0
    thickness_med: float = 0.0
    thickness_p90: float = 0.0
    tube_fit: float = 0.0
    width_cv: float = 0.0
    aspect_ratio: float = 1.0


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

        # 트랙별 baseline (첫 프레임 SAM2 결과 기준)
        self.baselines: Dict[int, TrackBaseline] = {}  # track_id -> baseline

        # 스냅샷 전환점 감지용 상태 (5프레임 지속 필터)
        self.frame_buffer: Dict[int, List[Tuple[np.ndarray, FrameDebugRecord, Optional[np.ndarray]]]] = {}
        # track_id -> 최근 6프레임 버퍼 [(frame_bgr, record, sam2_mask), ...]
        self.flag_streak: Dict[int, int] = {}  # track_id -> 현재 상태 연속 프레임 수
        self.confirmed_flag_set: Dict[int, frozenset] = {}  # track_id -> 확정된 플래그 집합
        self.pending_transition: Dict[int, Tuple[int, str, frozenset]] = {}
        # track_id -> (transition_frame_idx, transition_type, new_flag_set)
        self.event_counter: int = 0  # 전역 이벤트 번호
        self.min_persist_frames: int = 5  # 최소 지속 프레임 수

        os.makedirs(output_dir, exist_ok=True)

    def set_baseline(self, track_id: int, shape_stats: ShapeStats) -> None:
        """첫 프레임에서 baseline 설정."""
        if track_id in self.baselines:
            return  # 이미 설정됨
        if shape_stats is None or shape_stats.area < 10:
            return

        # width_cv에 최소값(floor) 적용: 첫 프레임이 비정상적으로 낮을 경우
        # false positive 방지 (예: 0.22 -> 0.25로 올려서 2x=0.50 임계값 확보)
        width_cv_floor = max(shape_stats.skel_width_cv, 0.25)

        self.baselines[track_id] = TrackBaseline(
            area=shape_stats.area,
            thickness_med=shape_stats.thickness_med,
            thickness_p90=shape_stats.thickness_p90,
            tube_fit=shape_stats.skel_tube_fit,
            width_cv=width_cv_floor,
            aspect_ratio=shape_stats.aspect_ratio,
        )

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
        """이상징후 플래그 자동 감지 (baseline 기반 상대값 사용)."""
        flags = []
        ss = record.shape_stats
        baseline = self.baselines.get(record.track_id)

        # --- Baseline 기반 상대 비교 ---
        if baseline and baseline.area > 0 and ss and ss.area > 0:
            # AREA_BLOWUP: baseline 대비 면적 급변
            area_ratio = ss.area / baseline.area
            if area_ratio > 3.0:
                flags.append(f"AREA_BLOWUP_UP({area_ratio:.1f}x)")
            elif area_ratio < 0.3:
                flags.append(f"AREA_BLOWUP_DOWN({area_ratio:.2f}x)")

            # THICKNESS_BLOWUP: baseline 대비 두께 급변
            if baseline.thickness_p90 > 0:
                thick_ratio = ss.thickness_p90 / baseline.thickness_p90
                if thick_ratio > 2.0:
                    flags.append(f"THICKNESS_BLOWUP({thick_ratio:.1f}x)")

            # TUBE_FIT_DEGRADE: baseline 대비 tube_fit 악화
            if baseline.tube_fit > 0 and ss.skel_tube_fit > 0:
                tf_ratio = ss.skel_tube_fit / baseline.tube_fit
                if tf_ratio < 0.5:
                    flags.append(f"TUBE_FIT_DEGRADE({tf_ratio:.2f}x)")

            # WIDTH_CV_HIGH: 절대값 기준 (상대비교 대신)
            # 번짐 판별: Good median=0.60, Bad median=0.99 → 0.85 이상이면 의심
            if ss.skel_width_cv > 0.85:
                flags.append(f"WIDTH_CV_HIGH({ss.skel_width_cv:.2f})")
        else:
            # baseline 없으면 직전 프레임 대비 (기존 방식)
            if record.area_ratio_prev > 3.0:
                flags.append(f"AREA_BLOWUP_UP({record.area_ratio_prev:.1f}x)")
            elif record.area_ratio_prev < 0.3 and record.area_ratio_prev > 0:
                flags.append(f"AREA_BLOWUP_DOWN({record.area_ratio_prev:.2f}x)")

        # --- 절대값 기반 플래그 (baseline 무관) ---
        # BORDER_TOUCH_HIGH
        if ss and ss.border_touch > 0.35:
            flags.append(f"BORDER_TOUCH_HIGH({ss.border_touch:.2f})")

        # CENTER_JUMP_OUTLIER (baseline 면적의 sqrt 기준)
        prev_center = self.prev_centers.get(record.track_id)
        if prev_center and record.chosen_center:
            jump = ((record.chosen_center[0] - prev_center[0]) ** 2 +
                    (record.chosen_center[1] - prev_center[1]) ** 2) ** 0.5
            # baseline이 있으면 sqrt(area)의 2배 이상 점프 시 이상
            jump_thresh = 100.0
            if baseline and baseline.area > 0:
                jump_thresh = max(2.0 * (baseline.area ** 0.5), 50.0)
            if jump > jump_thresh:
                flags.append(f"CENTER_JUMP({jump:.0f}px)")

        # CONSENSUS_BREAK: SAM2와 TPL/KLT 모두 멀리 떨어짐
        if record.d_pred_sam2 > 50 and record.d_pred_tpl > 50:
            flags.append("CONSENSUS_BREAK")

        # LOW_SHAPE_SCORE
        if ss and ss.shape_score < 0.5:
            flags.append(f"LOW_SHAPE_SCORE({ss.shape_score:.2f})")

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

    def _extract_flag_types(self, reason_flags: List[str]) -> frozenset:
        """플래그 리스트에서 타입만 추출 (괄호 안 값 제거).

        예: ["AREA_BLOWUP_UP(3.5x)", "WIDTH_CV_HIGH(0.91)"]
            -> frozenset({"AREA_BLOWUP_UP", "WIDTH_CV_HIGH"})
        """
        return frozenset(f.split("(")[0] for f in reason_flags if f)

    def handle_snapshot_transition(
        self,
        frame_bgr: np.ndarray,
        record: FrameDebugRecord,
        sam2_mask: Optional[np.ndarray] = None,
    ):
        """플래그 전환점에서만 스냅샷 저장 (5프레임 지속 필터 적용).

        호출 시점: add_record() 후 (reason_flags가 설정된 후)
        전환이 5프레임 이상 지속되어야 실제 전환으로 인정하고 스냅샷 저장.

        전환 조건:
        1. clean -> flagged (problem_start)
        2. flagged -> clean (problem_end)
        3. 플래그 종류가 바뀜 (problem_change) - 새 플래그 등장
        """
        track_id = record.track_id
        current_flags = self._extract_flag_types(record.reason_flags)
        confirmed_flags = self.confirmed_flag_set.get(track_id, frozenset())

        # 1) 프레임 버퍼 업데이트 (최근 6프레임 유지)
        if track_id not in self.frame_buffer:
            self.frame_buffer[track_id] = []
        buffer = self.frame_buffer[track_id]
        buffer.append((frame_bgr.copy(), record, sam2_mask))
        if len(buffer) > 6:
            buffer.pop(0)

        # 2) 상태 변화 감지 (플래그 집합 비교)
        # 전환 타입 결정
        transition_type = None
        if not confirmed_flags and current_flags:
            # clean -> flagged
            transition_type = "problem_start"
        elif confirmed_flags and not current_flags:
            # flagged -> clean
            transition_type = "problem_end"
        elif confirmed_flags and current_flags:
            # 둘 다 있음 → 새 플래그가 나타났는지 확인
            new_flags = current_flags - confirmed_flags
            if new_flags:
                # 새 플래그 등장 = problem_change
                transition_type = "problem_change"

        # 3) 상태 연속성 추적
        if transition_type is None:
            # 확정 상태와 동일 → streak 리셋, pending 취소
            self.flag_streak[track_id] = 0
            if track_id in self.pending_transition:
                del self.pending_transition[track_id]
        else:
            # 전환 발생 → streak 증가
            streak = self.flag_streak.get(track_id, 0) + 1
            self.flag_streak[track_id] = streak

            # pending transition 등록 (아직 없으면)
            if track_id not in self.pending_transition:
                self.pending_transition[track_id] = (record.frame_idx, transition_type, current_flags)

            # 4) 5프레임 지속 확인 → 스냅샷 저장
            if streak >= self.min_persist_frames:
                self._confirm_and_save_transition(track_id, current_flags)

    def _confirm_and_save_transition(self, track_id: int, new_flag_set: frozenset):
        """전환 확정 및 스냅샷 저장."""
        if track_id not in self.pending_transition:
            return

        trans_frame_idx, transition_type, _ = self.pending_transition.pop(track_id)
        buffer = self.frame_buffer.get(track_id, [])

        if len(buffer) < 2:
            self.confirmed_flag_set[track_id] = new_flag_set
            self.flag_streak[track_id] = 0
            return

        # 버퍼에서 frame_idx로 프레임 찾기
        def find_frame(target_idx):
            for img, rec, mask in buffer:
                if rec.frame_idx == target_idx:
                    return (img, rec, mask)
            return None

        snapshots_dir = os.path.join(self.output_dir, "snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)

        self.event_counter += 1
        event_num = self.event_counter

        # before: 전환 직전 프레임
        before_data = find_frame(trans_frame_idx - 1)
        if before_data:
            img, rec, mask = before_data
            self._save_snapshot_frame(
                img, rec, mask, snapshots_dir,
                f"{event_num:03d}_T{track_id}_before_{transition_type}"
            )

        # event: 전환 프레임
        event_data = find_frame(trans_frame_idx)
        if event_data:
            img, rec, mask = event_data
            self._save_snapshot_frame(
                img, rec, mask, snapshots_dir,
                f"{event_num:03d}_T{track_id}_event_{transition_type}"
            )

        # after: 전환 직후 프레임
        after_data = find_frame(trans_frame_idx + 1)
        if after_data:
            img, rec, mask = after_data
            self._save_snapshot_frame(
                img, rec, mask, snapshots_dir,
                f"{event_num:03d}_T{track_id}_after_{transition_type}"
            )

        # 상태 확정 (플래그 집합으로)
        self.confirmed_flag_set[track_id] = new_flag_set
        self.flag_streak[track_id] = 0

    def _save_snapshot_frame(
        self,
        frame_bgr: np.ndarray,
        record: FrameDebugRecord,
        sam2_mask: Optional[np.ndarray],
        snapshots_dir: str,
        filename_prefix: str,
    ):
        """단일 스냅샷 프레임 저장 (시각화 포함)."""
        img = frame_bgr.copy()

        # 트랙별 마스크 색상
        mask_colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255),
        ]
        mask_color = mask_colors[(record.track_id - 1) % len(mask_colors)]

        # 상태별 색상
        state_colors = {
            "ACTIVE": (0, 255, 0), "UNCERTAIN": (0, 255, 255),
            "OCCLUDED": (0, 165, 255), "MERGED": (255, 255, 0),
            "NEEDS_RESEED": (255, 0, 255),
        }
        state_color = state_colors.get(record.state, (200, 200, 200))

        # SAM2 마스크 반투명 오버레이
        if sam2_mask is not None and record.roi_xyxy is not None:
            rx0, ry0, rx1, ry1 = record.roi_xyxy
            mh, mw = sam2_mask.shape[:2]
            roi_h, roi_w = ry1 - ry0, rx1 - rx0
            if mh == roi_h and mw == roi_w:
                roi_region = img[ry0:ry1, rx0:rx1]
                mask_bool = sam2_mask > 0
                overlay = roi_region.copy()
                overlay[mask_bool] = (
                    np.array(mask_color) * 0.4 + roi_region[mask_bool] * 0.6
                ).astype(np.uint8)
                img[ry0:ry1, rx0:rx1] = overlay

        # ROI 사각형
        if record.roi_xyxy:
            rx0, ry0, rx1, ry1 = record.roi_xyxy
            cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (255, 255, 0), 2)

        # 센서별 중심점 마커
        if record.pred_center:
            px, py = int(record.pred_center[0]), int(record.pred_center[1])
            cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        if record.sam2_center:
            sx, sy = int(record.sam2_center[0]), int(record.sam2_center[1])
            cv2.circle(img, (sx, sy), 8, (0, 255, 0), 2)
        if record.tpl_center:
            tx, ty = int(record.tpl_center[0]), int(record.tpl_center[1])
            cv2.rectangle(img, (tx-6, ty-6), (tx+6, ty+6), (255, 0, 0), 2)
        if record.klt_center:
            kx, ky = int(record.klt_center[0]), int(record.klt_center[1])
            cv2.drawMarker(img, (kx, ky), (0, 0, 255), cv2.MARKER_DIAMOND, 10, 2)
        if record.chosen_center:
            cx, cy = int(record.chosen_center[0]), int(record.chosen_center[1])
            cv2.circle(img, (cx, cy), 5, state_color, -1)

        # 트랙 라벨 + shape 정보
        label_x, label_y = 10, 30
        if record.chosen_center:
            label_x = int(record.chosen_center[0])
            label_y = max(20, int(record.chosen_center[1]) - 15)

        line1 = f"T{record.track_id} {record.state}"
        if record.sensor_used:
            line1 += f" [{record.sensor_used}]"

        if record.shape_stats:
            ss = record.shape_stats
            line2 = f"ss={ss.shape_score:.2f} tf={ss.skel_tube_fit:.2f} wcv={ss.skel_width_cv:.2f}"
        else:
            line2 = f"q={record.quality_score:.2f}"

        for i, line in enumerate([line1, line2]):
            ty = label_y + i * 16
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (label_x - 2, ty - th - 2), (label_x + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(img, line, (label_x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)

        # 플래그
        if record.reason_flags:
            flag_text = " | ".join(record.reason_flags[:4])
            h, w = img.shape[:2]
            cv2.putText(img, flag_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # 프레임 번호
        cv2.putText(img, f"F{record.frame_idx:04d}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        filename = f"{filename_prefix}_F{record.frame_idx:04d}.png"
        cv2.imwrite(os.path.join(snapshots_dir, filename), img)

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
    sam2_mask: Optional[np.ndarray] = None,
):
    """이벤트 프레임 스냅샷 저장 (overlay 스타일 시각화).

    Args:
        frame_bgr: 원본 프레임
        record: 디버그 레코드 (좌표, shape_stats 등)
        output_dir: 출력 디렉토리
        sam2_mask: SAM2 마스크 (ROI 좌표 기준, 0/255)
    """
    snapshots_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    img = frame_bgr.copy()

    # 트랙별 마스크 색상
    mask_colors = [
        (0, 255, 0),    # 녹색 (track 1)
        (255, 0, 0),    # 파랑 (track 2)
        (0, 0, 255),    # 빨강 (track 3)
        (255, 255, 0),  # 시안 (track 4)
        (255, 0, 255),  # 마젠타 (track 5)
    ]
    mask_color = mask_colors[(record.track_id - 1) % len(mask_colors)]

    # 상태별 색상
    state_colors = {
        "ACTIVE": (0, 255, 0),
        "UNCERTAIN": (0, 255, 255),
        "OCCLUDED": (0, 165, 255),
        "MERGED": (255, 255, 0),
        "NEEDS_RESEED": (255, 0, 255),
    }
    state_color = state_colors.get(record.state, (200, 200, 200))

    # --- 1) SAM2 마스크 반투명 오버레이 ---
    if sam2_mask is not None and record.roi_xyxy is not None:
        rx0, ry0, rx1, ry1 = record.roi_xyxy
        mh, mw = sam2_mask.shape[:2]
        roi_h, roi_w = ry1 - ry0, rx1 - rx0

        if mh == roi_h and mw == roi_w:
            roi_region = img[ry0:ry1, rx0:rx1]
            mask_bool = sam2_mask > 0
            overlay = roi_region.copy()
            overlay[mask_bool] = (
                np.array(mask_color) * 0.4 +
                roi_region[mask_bool] * 0.6
            ).astype(np.uint8)
            img[ry0:ry1, rx0:rx1] = overlay

    # --- 2) ROI 사각형 (하늘색) ---
    if record.roi_xyxy:
        rx0, ry0, rx1, ry1 = record.roi_xyxy
        cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (255, 255, 0), 2)

    # --- 3) 센서별 중심점 마커 ---
    # PRED center (흰색 X)
    if record.pred_center:
        px, py = int(record.pred_center[0]), int(record.pred_center[1])
        cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)

    # SAM2 center (녹색 원)
    if record.sam2_center:
        sx, sy = int(record.sam2_center[0]), int(record.sam2_center[1])
        cv2.circle(img, (sx, sy), 8, (0, 255, 0), 2)

    # TPL center (파란색 사각형)
    if record.tpl_center:
        tx, ty = int(record.tpl_center[0]), int(record.tpl_center[1])
        cv2.rectangle(img, (tx-6, ty-6), (tx+6, ty+6), (255, 0, 0), 2)

    # KLT center (빨간색 마름모)
    if record.klt_center:
        kx, ky = int(record.klt_center[0]), int(record.klt_center[1])
        cv2.drawMarker(img, (kx, ky), (0, 0, 255), cv2.MARKER_DIAMOND, 10, 2)

    # chosen center (상태 색상 채워진 원)
    if record.chosen_center:
        cx, cy = int(record.chosen_center[0]), int(record.chosen_center[1])
        cv2.circle(img, (cx, cy), 5, state_color, -1)

    # --- 4) 트랙 라벨 + shape 정보 (중심 근처) ---
    label_x, label_y = 10, 30
    if record.chosen_center:
        label_x = int(record.chosen_center[0])
        label_y = max(20, int(record.chosen_center[1]) - 15)

    # 첫 줄: ID + 상태 + 센서
    line1 = f"T{record.track_id} {record.state}"
    if record.sensor_used:
        line1 += f" [{record.sensor_used}]"

    # 둘째 줄: shape 점수 + skeleton 정보
    if record.shape_stats:
        ss = record.shape_stats
        line2 = f"ss={ss.shape_score:.2f} tf={ss.skel_tube_fit:.2f} wcv={ss.skel_width_cv:.2f}"
    else:
        line2 = f"q={record.quality_score:.2f}"

    # 텍스트 배경 (가독성)
    for i, line in enumerate([line1, line2]):
        ty = label_y + i * 16
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (label_x - 2, ty - th - 2), (label_x + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, line, (label_x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)

    # --- 5) 플래그 (화면 좌하단) ---
    if record.reason_flags:
        flag_text = " | ".join(record.reason_flags[:4])
        h, w = img.shape[:2]
        cv2.putText(img, flag_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # --- 6) 프레임 번호 (좌상단) ---
    cv2.putText(img, f"F{record.frame_idx:04d}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    filename = f"frame_{record.frame_idx:04d}_track_{record.track_id}.png"
    cv2.imwrite(os.path.join(snapshots_dir, filename), img)
