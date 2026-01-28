# tracking/multi_tracker.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import math
import cv2
import numpy as np

from detection.motion_detector import MotionDetections, Blob
from tracking.track import Track, TrackState
from tracking.kalman import KalmanFilter2D
from tracking.association import assign_tracks_to_detections
from roi.roi_manager import RoiManager
from segmentation.base import Segmenter


def bbox_from_blob(b: Blob) -> Tuple[int, int, int, int]:
    return b.bbox


def clamp_bbox(b, W, H):
    x0, y0, x1, y1 = b
    x0 = max(0, min(W - 1, int(x0)))
    y0 = max(0, min(H - 1, int(y0)))
    x1 = max(1, min(W, int(x1)))
    y1 = max(1, min(H, int(y1)))
    if x1 <= x0 + 1:
        x1 = min(W, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(H, y0 + 2)
    return (x0, y0, x1, y1)


class MultiObjectTracker:
    def __init__(self, cfg: Dict[str, Any], frame_size: Tuple[int, int], fps: float):
        self.cfg = cfg
        self.W, self.H = frame_size
        self.fps = float(fps)

        tcfg = cfg.get("tracking", {})
        self.max_tracks = int(tcfg.get("max_tracks", 10))
        self.max_assign_dist = float(tcfg.get("max_assign_dist_px", 60))

        self.occlusion_max_misses = int(tcfg.get("occlusion_max_misses", 15))
        self.border_exit_margin = int(tcfg.get("border_exit_margin_px", 5))

        self.immobile_window = int(tcfg.get("immobile_window", 60))
        self.immobile_speed = float(tcfg.get("immobile_speed_px_per_s", 1.0))

        self.roi_mgr = RoiManager(cfg)

        self.tracks: List[Track] = []
        self._next_id = 1

        self._last_overlay_tracks: List[Track] = []

    def _new_track_from_blob(self, blob: Blob, frame_idx: int) -> Track:
        kf = KalmanFilter2D(blob.cx, blob.cy)
        tr = Track(id=self._next_id, kf=kf)
        self._next_id += 1
        tr.bbox = clamp_bbox(blob.bbox, self.W, self.H)
        tr.area = int(blob.area)
        tr.last_seen_frame = frame_idx
        tr.misses = 0
        tr.state = TrackState.ACTIVE
        return tr

    def step(
        self,
        frame_bgr,
        frame_gray,
        motion: MotionDetections,
        segmenter: Segmenter,
        frame_idx: int,
    ) -> List[Track]:
        # 1) 예측
        for tr in self.tracks:
            tr.kf.predict(dt=1.0)
            tr.update_speed(self.fps)

        # 2) ACTIVE/OCCLUDED만 연관 대상
        live_tracks = [t for t in self.tracks if t.state in (TrackState.ACTIVE, TrackState.OCCLUDED)]
        track_centers = []
        for t in live_tracks:
            px, py = t.kf.get_position()
            track_centers.append((px, py))

        det_centers = [(b.cx, b.cy) for b in motion.blobs]

        matches, unmatched_t_idx, unmatched_d_idx = assign_tracks_to_detections(
            track_centers, det_centers, max_dist=self.max_assign_dist
        )

        # 3) 매칭 업데이트
        used_track_ids = set()
        for ti, di in matches:
            tr = live_tracks[ti]
            b = motion.blobs[di]

            tr.kf.update(b.cx, b.cy)
            tr.bbox = clamp_bbox(bbox_from_blob(b), self.W, self.H)
            tr.area = int(b.area)
            tr.last_seen_frame = frame_idx
            tr.misses = 0
            tr.state = TrackState.ACTIVE
            used_track_ids.add(tr.id)

        # 4) 미매칭 트랙 처리(occlusion/exit)
        for ui in unmatched_t_idx:
            tr = live_tracks[ui]
            tr.misses += 1
            if tr.misses <= self.occlusion_max_misses:
                tr.state = TrackState.OCCLUDED
            else:
                # 경계에 가까운 상태에서 사라졌으면 EXITED로
                cx, cy = tr.kf.get_position()
                near_border = (
                    cx < self.border_exit_margin
                    or cy < self.border_exit_margin
                    or cx > (self.W - 1 - self.border_exit_margin)
                    or cy > (self.H - 1 - self.border_exit_margin)
                )
                tr.state = TrackState.EXITED if near_border else TrackState.OCCLUDED

        # 5) 미매칭 detection -> 새 트랙 생성(최대 개수 제한)
        if len(self.tracks) < self.max_tracks:
            for di in unmatched_d_idx:
                if len(self.tracks) >= self.max_tracks:
                    break
                b = motion.blobs[di]
                self.tracks.append(self._new_track_from_blob(b, frame_idx))

        # 6) ROI 기반 세그먼트 보정(선택) -> bbox/area 품질 갱신
        if segmenter.enabled:
            for tr in self.tracks:
                if tr.state not in (TrackState.ACTIVE, TrackState.OCCLUDED):
                    continue
                if tr.bbox is None:
                    continue

                roi = self.roi_mgr.make_roi(tr, (self.W, self.H))
                crop_bgr = self.roi_mgr.crop(frame_bgr, roi)
                crop_fg = self.roi_mgr.crop(motion.fg_mask, roi)

                seg = segmenter.segment(track=tr, roi=roi, crop_bgr=crop_bgr, crop_fg=crop_fg, frame_idx=frame_idx)

                if seg is None:
                    continue

                # seg: dict {mask_roi(uint8 0/255), bbox_roi, quality}
                bbox_roi = seg.get("bbox_roi")
                quality = seg.get("quality")
                if bbox_roi is not None:
                    x0, y0, x1, y1 = bbox_roi
                    full_bbox = (x0 + roi.x0, y0 + roi.y0, x1 + roi.x0, y1 + roi.y0)
                    tr.bbox = clamp_bbox(full_bbox, self.W, self.H)
                if quality is not None:
                    tr.quality = float(quality)

                mask_roi = seg.get("mask_roi")
                if mask_roi is not None:
                    tr.area = int(np.count_nonzero(mask_roi))

        # 7) 죽음/정지 판정(매우 단순)
        for tr in self.tracks:
            if tr.state in (TrackState.ACTIVE, TrackState.OCCLUDED):
                tr.update_speed(self.fps)
                # 최근 속도 평균이 너무 낮으면 DEAD로(실험 맞게 튜닝 필요)
                if len(tr._speed_hist) >= min(self.immobile_window, tr._speed_hist.maxlen):
                    mean_sp = sum(list(tr._speed_hist)[-self.immobile_window:]) / self.immobile_window
                    if mean_sp <= self.immobile_speed:
                        tr.state = TrackState.DEAD

        # 8) overlay용 캐시
        self._last_overlay_tracks = [t for t in self.tracks]

        return self.tracks

    def draw_overlay(self, frame_bgr, frame_idx: int):
        for tr in self._last_overlay_tracks:
            if tr.bbox is None:
                continue
            x0, y0, x1, y1 = tr.bbox
            color = (0, 255, 0) if tr.state == TrackState.ACTIVE else (0, 255, 255)
            if tr.state == TrackState.EXITED:
                color = (255, 0, 0)
            if tr.state == TrackState.DEAD:
                color = (0, 0, 255)

            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
            cx, cy = tr.center()
            if cx is not None and cy is not None:
                cv2.circle(frame_bgr, (int(cx), int(cy)), 2, color, -1)

            txt = f"id={tr.id} {tr.state.value} miss={tr.misses}"
            if tr.speed_px_s is not None:
                txt += f" v={tr.speed_px_s:.1f}"
            if tr.quality is not None:
                txt += f" q={tr.quality:.2f}"
            cv2.putText(frame_bgr, txt, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.putText(frame_bgr, f"frame={frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame_bgr
