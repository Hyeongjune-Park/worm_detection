# io/video_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Union

import cv2


class VideoReader:
    def __init__(self, path: str, fps_fallback: float = 30.0):
        self.path = Path(path)
        self.fps_fallback = float(fps_fallback)

        self._cap: Optional[cv2.VideoCapture] = None
        self._images: Optional[List[Path]] = None
        self._img_idx = 0

        if self.path.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            imgs = [p for p in sorted(self.path.iterdir()) if p.suffix.lower() in exts]
            if not imgs:
                raise FileNotFoundError(f"No image frames in folder: {self.path}")
            self._images = imgs
            first = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
            if first is None:
                raise RuntimeError(f"Failed to read first image: {imgs[0]}")
            self._height, self._width = first.shape[:2]
            self._fps = self.fps_fallback
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.path}")
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            if fps <= 1e-6:
                fps = self.fps_fallback
            self._width, self._height, self._fps = w, h, fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return int(self._width)

    @property
    def height(self) -> int:
        return int(self._height)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self._images is not None:
            if self._img_idx >= len(self._images):
                raise StopIteration
            p = self._images[self._img_idx]
            self._img_idx += 1
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {p}")
            return frame
        else:
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise StopIteration
            return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
