# io_utils/overlay.py
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2

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


def draw_overlay(
    frame_bgr,
    tracks: List[Track],
    frame_idx: int,
    arena_rect: Optional[Tuple[int, int, int, int]] = None,
):
    """프레임에 트래킹 정보 오버레이."""

    # arena
    if arena_rect is not None:
        ax0, ay0, ax1, ay1 = arena_rect
        cv2.rectangle(frame_bgr, (ax0, ay0), (ax1, ay1), (255, 255, 0), 1)

    for tr in tracks:
        color = _STATE_COLORS.get(tr.state, (200, 200, 200))

        # bbox
        if tr.bbox is not None:
            x0, y0, x1, y1 = tr.bbox
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 1)

        # center
        cx, cy = tr.center()
        if cx is not None and cy is not None:
            cv2.circle(frame_bgr, (int(cx), int(cy)), 4, color, -1)

            # head arrow
            if tr.last_head is not None:
                hx, hy = tr.last_head
                cv2.arrowedLine(
                    frame_bgr,
                    (int(cx), int(cy)),
                    (int(hx), int(hy)),
                    (0, 0, 255), 2, tipLength=0.3,
                )

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
