# io_utils/overlay.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.track import Track, TrackState


# 상태별 색상 (BGR)
_STATE_COLORS = {
    TrackState.ACTIVE:         (0, 255, 0),     # green
    TrackState.UNCERTAIN:      (0, 255, 255),   # yellow
    TrackState.OCCLUDED:       (0, 165, 255),   # orange
    TrackState.MERGED:         (255, 255, 0),   # cyan
    TrackState.EXITED:         (255, 0, 0),     # blue
    TrackState.DEAD_CANDIDATE: (0, 0, 255),     # red
    TrackState.NEEDS_RESEED:   (255, 0, 255),   # magenta
}


@dataclass
class DebugInfo:
    """디버그 시각화용 센서 정보."""
    track_id: int
    roi: Optional[Tuple[int, int, int, int]] = None  # (x0, y0, x1, y1)
    pred_center: Optional[Tuple[float, float]] = None
    sam2_center: Optional[Tuple[float, float]] = None
    tpl_center: Optional[Tuple[float, float]] = None
    klt_center: Optional[Tuple[float, float]] = None
    sam2_mask: Optional[np.ndarray] = None  # ROI 좌표 기준 마스크 (0/255)


def draw_overlay(
    frame_bgr,
    tracks: List[Track],
    frame_idx: int,
    arena_rect: Optional[Tuple[int, int, int, int]] = None,
    debug_infos: Optional[Dict[int, DebugInfo]] = None,
):
    """프레임에 트래킹 정보 오버레이."""

    # arena
    if arena_rect is not None:
        ax0, ay0, ax1, ay1 = arena_rect
        cv2.rectangle(frame_bgr, (ax0, ay0), (ax1, ay1), (255, 255, 0), 1)

    for tr in tracks:
        color = _STATE_COLORS.get(tr.state, (200, 200, 200))

        # 디버그 정보 그리기
        if debug_infos and tr.id in debug_infos:
            dbg = debug_infos[tr.id]

            # SAM2 mask (반투명 색상 오버레이)
            if dbg.sam2_mask is not None and dbg.roi is not None:
                rx0, ry0, rx1, ry1 = dbg.roi
                mask = dbg.sam2_mask
                mh, mw = mask.shape[:2]

                # ROI 크기와 마스크 크기가 맞는지 확인
                roi_h, roi_w = ry1 - ry0, rx1 - rx0
                if mh == roi_h and mw == roi_w:
                    # 마스크 영역에 색상 오버레이 (트랙 ID별 색상)
                    mask_colors = [
                        (0, 255, 0),    # 녹색 (track 1)
                        (255, 0, 0),    # 파랑 (track 2)
                        (0, 0, 255),    # 빨강 (track 3)
                        (255, 255, 0),  # 시안 (track 4)
                        (255, 0, 255),  # 마젠타 (track 5)
                    ]
                    mask_color = mask_colors[(tr.id - 1) % len(mask_colors)]

                    # ROI 영역 추출
                    roi_region = frame_bgr[ry0:ry1, rx0:rx1]

                    # 마스크가 있는 곳에 색상 블렌딩
                    mask_bool = mask > 0
                    overlay = roi_region.copy()
                    overlay[mask_bool] = (
                        np.array(mask_color) * 0.4 +
                        roi_region[mask_bool] * 0.6
                    ).astype(np.uint8)
                    frame_bgr[ry0:ry1, rx0:rx1] = overlay

            # ROI (하늘색 사각형)
            if dbg.roi:
                rx0, ry0, rx1, ry1 = dbg.roi
                cv2.rectangle(frame_bgr, (rx0, ry0), (rx1, ry1), (255, 255, 0), 2)

            # PRED center (흰색 X)
            if dbg.pred_center:
                px, py = int(dbg.pred_center[0]), int(dbg.pred_center[1])
                cv2.drawMarker(frame_bgr, (px, py), (255, 255, 255),
                              cv2.MARKER_TILTED_CROSS, 12, 2)

            # SAM2 center (녹색 원)
            if dbg.sam2_center:
                sx, sy = int(dbg.sam2_center[0]), int(dbg.sam2_center[1])
                cv2.circle(frame_bgr, (sx, sy), 8, (0, 255, 0), 2)

            # TPL center (파란색 사각형)
            if dbg.tpl_center:
                tx, ty = int(dbg.tpl_center[0]), int(dbg.tpl_center[1])
                cv2.rectangle(frame_bgr, (tx-6, ty-6), (tx+6, ty+6), (255, 0, 0), 2)

            # KLT center (빨간색 마름모)
            if dbg.klt_center:
                kx, ky = int(dbg.klt_center[0]), int(dbg.klt_center[1])
                cv2.drawMarker(frame_bgr, (kx, ky), (0, 0, 255),
                              cv2.MARKER_DIAMOND, 10, 2)

        # bbox (상태 색상)
        if tr.bbox is not None:
            x0, y0, x1, y1 = tr.bbox
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 1)

        # center
        cx, cy = tr.center()
        if cx is not None and cy is not None:
            cv2.circle(frame_bgr, (int(cx), int(cy)), 4, color, -1)

        # text
        txt = f"id={tr.id} {tr.state.value}"
        if tr.sensor_used:
            txt += f" [{tr.sensor_used}]"
        txt += f" q={tr.quality_score:.2f}"

        tx = int(cx) if cx is not None else 10
        ty = max(15, (int(cy) - 10) if cy is not None else 15)
        cv2.putText(frame_bgr, txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # frame info
    cv2.putText(frame_bgr, f"frame={frame_idx}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame_bgr
