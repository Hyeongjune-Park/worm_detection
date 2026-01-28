# preprocess/preprocess.py
from __future__ import annotations

from typing import Dict, Any
import cv2
import numpy as np


class Preprocessor:
    def __init__(self, cfg: Dict[str, Any]):
        p = cfg.get("preprocess", {})
        self.use_gray = bool(p.get("use_gray", True))

        den = p.get("denoise", {})
        self.denoise_enabled = bool(den.get("enabled", True))
        self.denoise_ksize = int(den.get("ksize", 3))
        if self.denoise_ksize not in (3, 5, 7):
            self.denoise_ksize = 3

        clahe = p.get("clahe", {})
        self.clahe_enabled = bool(clahe.get("enabled", False))
        self.clahe_clip = float(clahe.get("clipLimit", 2.0))
        tgs = clahe.get("tileGridSize", [8, 8])
        self.clahe_tgs = (int(tgs[0]), int(tgs[1]))
        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tgs) if self.clahe_enabled else None

    def apply(self, frame_bgr) -> Dict[str, Any]:
        out: Dict[str, Any] = {"bgr": frame_bgr}

        if self.use_gray:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr[:, :, 1].copy()  # G채널 정도로 대체

        if self.denoise_enabled:
            gray = cv2.GaussianBlur(gray, (self.denoise_ksize, self.denoise_ksize), 0)

        if self._clahe is not None:
            gray = self._clahe.apply(gray)

        out["gray"] = gray
        return out
