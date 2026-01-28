# tracking/state_machine.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tracking.track import Track, TrackState
from qa.fusion import FusionResult


class TrackStateMachine:
    """
    7-state 상태 머신.
    ACTIVE ↔ UNCERTAIN ↔ OCCLUDED → NEEDS_RESEED
    MERGED (겹침 시)
    EXITED (arena 밖)
    DEAD_CANDIDATE (장기 정지)
    """

    def __init__(self, cfg: Dict[str, Any]):
        sm = cfg.get("tracking", {}).get("state_machine", {})
        self.border_margin = int(sm.get("border_exit_margin_px", 5))
        self.immobile_window = int(sm.get("immobile_window", 60))
        self.immobile_speed = float(sm.get("immobile_speed_px_per_s", 1.0))
        self.merge_dist = float(sm.get("merge_distance_px", 30.0))

        qa = cfg.get("qa", {})
        self.uncertain_to_occluded = int(qa.get("uncertain_to_occluded_frames", 5))
        self.occluded_to_reseed = int(qa.get("occluded_to_reseed_frames", 30))

    def update(
        self,
        track: Track,
        fusion: FusionResult,
        frame_idx: int,
        arena_rect: Optional[Tuple[int, int, int, int]],
        frame_size: Tuple[int, int],
    ) -> TrackState:
        """fusion 결과에 따라 트랙 상태 전이. 반환: 새 상태."""
        # EXITED / NEEDS_RESEED는 고정 (사용자 개입 전까지)
        if track.state in (TrackState.EXITED, TrackState.NEEDS_RESEED):
            return track.state

        hint = fusion.state_hint
        W, H = frame_size

        # --- 융합 성공 (ACTIVE) ---
        if hint == TrackState.ACTIVE:
            track.miss_count = 0
            track.last_good_frame_idx = frame_idx
            # MERGED였어도 성공적 분리 시 ACTIVE 복귀
            return TrackState.ACTIVE

        # --- UNCERTAIN ---
        if hint == TrackState.UNCERTAIN:
            track.miss_count += 1
            if track.miss_count >= self.uncertain_to_occluded:
                return TrackState.OCCLUDED
            return TrackState.UNCERTAIN

        # --- OCCLUDED (predict only) ---
        if hint == TrackState.OCCLUDED:
            track.miss_count += 1

            # arena 밖이면 EXITED
            cx, cy = track.center()
            if cx is not None and cy is not None and arena_rect is not None:
                ax0, ay0, ax1, ay1 = arena_rect
                if cx < ax0 or cy < ay0 or cx > ax1 or cy > ay1:
                    return TrackState.EXITED

            # 프레임 가장자리 이탈
            if cx is not None and cy is not None:
                if (cx < self.border_margin or cy < self.border_margin or
                        cx > W - 1 - self.border_margin or cy > H - 1 - self.border_margin):
                    if track.miss_count > self.uncertain_to_occluded:
                        return TrackState.EXITED

            # reseed 필요
            if track.miss_count >= self.occluded_to_reseed:
                return TrackState.NEEDS_RESEED

            return TrackState.OCCLUDED

        # 기본: 현재 상태 유지
        return track.state

    def check_merges(self, tracks: List[Track]) -> List[Dict[str, Any]]:
        """
        트랙 간 거리가 merge_dist 이하이면 MERGED 마킹.
        분리되면 ACTIVE로 복구.
        겹침에서 트랙 통합 금지 — MERGED 상태로 유지.

        Returns: merge 이벤트 리스트 (MERGE_ENTER / MERGE_EXIT)
        """
        active_tracks = [
            t for t in tracks
            if t.state in (TrackState.ACTIVE, TrackState.UNCERTAIN, TrackState.MERGED)
        ]

        merged_ids = set()
        merge_pairs: List[Tuple[int, int]] = []
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                ti = active_tracks[i]
                tj = active_tracks[j]
                ci = ti.center()
                cj = tj.center()
                if ci[0] is None or cj[0] is None:
                    continue
                dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
                if dist < self.merge_dist:
                    merged_ids.add(ti.id)
                    merged_ids.add(tj.id)
                    merge_pairs.append((ti.id, tj.id))

        events: List[Dict[str, Any]] = []
        for t in active_tracks:
            if t.id in merged_ids:
                if t.state != TrackState.MERGED:
                    t.state = TrackState.MERGED
                    events.append({"type": "MERGE_ENTER", "track_id": t.id,
                                   "pairs": [p for p in merge_pairs if t.id in p]})
            else:
                if t.state == TrackState.MERGED:
                    t.state = TrackState.ACTIVE
                    events.append({"type": "MERGE_EXIT", "track_id": t.id})
        return events

    def check_immobility(self, track: Track, fps: float) -> None:
        """장시간 정지 → DEAD_CANDIDATE."""
        if track.state not in (TrackState.ACTIVE, TrackState.UNCERTAIN):
            return
        if len(track._speed_hist) >= self.immobile_window:
            mean_sp = track.mean_speed(self.immobile_window)
            if mean_sp <= self.immobile_speed:
                track.state = TrackState.DEAD_CANDIDATE
