# io/video_writer.py
from __future__ import annotations

import cv2
from typing import Tuple


class OverlayWriter:
    def __init__(self, out_path: str, fps: float, frame_size: Tuple[int, int], fourcc: str = "mp4v"):
        W, H = frame_size
        self.out_path = out_path
        self.writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*fourcc),
            float(fps),
            (int(W), int(H)),
            True,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    def write(self, frame_bgr) -> None:
        self.writer.write(frame_bgr)

    def close(self) -> None:
        self.writer.release()
