# segmentation/sam2_adapter.py
from __future__ import annotations

from typing import Any, Dict, Optional
import os
import sys
import numpy as np
import cv2

from segmentation.base import Segmenter
from tracking.track import Track
from roi.roi_manager import RoiWindow


class MotionMaskSegmenter(Segmenter):
    """
    SAM2가 없을 때도 파이프라인이 돌아가게 하는 fallback.
    ROI 내부의 fg mask에서 가장 큰 blob을 track bbox 주변으로 선택.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(enabled=True)
        s = cfg.get("segmentation", {})
        self.update_every_n = int(s.get("update_every_n", 1))
        q = s.get("quality", {})
        self.max_area_jump = float(q.get("max_area_jump_ratio", 3.0))

    def segment(self, track: Track, roi: RoiWindow, crop_bgr, crop_fg, frame_idx: int) -> Optional[Dict[str, Any]]:
        if self.update_every_n > 1 and (frame_idx % self.update_every_n) != 0:
            return None

        # crop_fg: 0/255
        fg = crop_fg.copy()
        if fg.ndim == 3:
            fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

        # 조금 더 정리
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # track bbox를 ROI 좌표로 변환해서 그 주변 blob을 우선
        if track.bbox is not None:
            tx0, ty0, tx1, ty1 = track.bbox
            tb = (tx0 - roi.x0, ty0 - roi.y0, tx1 - roi.x0, ty1 - roi.y0)
            tcx = 0.5 * (tb[0] + tb[2])
            tcy = 0.5 * (tb[1] + tb[3])
        else:
            tcx, tcy = fg.shape[1] / 2, fg.shape[0] / 2

        best = None
        best_score = -1e18
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area <= 1:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w * 0.5
            cy = y + h * 0.5
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            # 가까울수록, 면적이 클수록
            score = area - 5.0 * dist
            if score > best_score:
                best_score = score
                best = (cnt, (x, y, x + w, y + h), area)

        if best is None:
            return None

        cnt, bbox_roi, area = best
        mask = np.zeros_like(fg, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

        # 품질(아주 단순): 이전 area 대비 급증이면 낮게
        quality = 1.0
        if track.area is not None and track.area > 0:
            ratio = float(area / max(1, track.area))
            if ratio > self.max_area_jump:
                quality = 0.2

        return {"mask_roi": mask, "bbox_roi": bbox_roi, "quality": quality}


class Sam2RoiSegmenter(Segmenter):
    """
    SAM2가 있을 때 ROI에서만 쓰는 어댑터.
    - SAM2 API가 설치/버전에 따라 다를 수 있어서,
      로딩 실패 시 build_segmenter()에서 자동으로 fallback을 선택한다.

    여기서는 '안 터지게' 최소한의 래퍼만 제공.
    실제 SAM2 video predictor 연동은 사용자 코드/가중치 경로에 맞춰 추가 튜닝 필요.
    """
    def __init__(self, cfg: Dict[str, Any], fps: float):
        super().__init__(enabled=True)
        self.cfg = cfg
        self.fps = float(fps)

        s = cfg.get("segmentation", {})
        self.update_every_n = int(s.get("update_every_n", 3))

        self.model_type = cfg.get("sam2", {}).get("model_type", "sam2_hiera_tiny")
        self.ckpt = cfg.get("sam2", {}).get("checkpoint_path", "")

        self._available = False
        self._init_sam2()

    def _init_sam2(self):
        sam2_dir = os.path.abspath("sam2")
        if os.path.isdir(sam2_dir) and sam2_dir not in sys.path:
            sys.path.insert(0, sam2_dir)

        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            self._available = False
            self._init_error = f"IMPORT_FAIL: {repr(e)}"
            return

        if not self.ckpt or not os.path.exists(self.ckpt):
            self._available = False
            self._init_error = f"CKPT_NOT_FOUND: {self.ckpt}"
            return

        try:
            self._torch = torch
            self._predictor = build_sam2_video_predictor(self.model_type, self.ckpt)
            self._available = True
            self._track_states = {}
        except Exception as e:
            self._available = False
            self._init_error = f"PREDICTOR_INIT_FAIL: {repr(e)}"

    @property
    def available(self) -> bool:
        return bool(self._available)

    def segment(self, track: Track, roi: RoiWindow, crop_bgr, crop_fg, frame_idx: int) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        if self.update_every_n > 1 and (frame_idx % self.update_every_n) != 0:
            return None

        # SAM2 API는 구현/버전에 따라 다름.
        # 여기서는 "안 터지는 구조"만 제공하고,
        # 실제로는 사용자 기존 SAM2 호출부를 이 안으로 옮기는 것을 권장.
        #
        # 즉, 아래는 placeholder에 가깝고, 실패 시 None 반환.
        try:
            # ROI에서 프롬프트 박스는 track.bbox를 roi좌표로 변환해서 사용
            if track.bbox is None:
                return None
            x0, y0, x1, y1 = track.bbox
            box = np.array([x0 - roi.x0, y0 - roi.y0, x1 - roi.x0, y1 - roi.y0], dtype=np.float32)

            # 여기부터는 당신의 기존 SAM2 코드 API에 맞게 교체해야 함.
            # 예시 개념만:
            # - track.id별로 predictor state를 유지
            # - 첫 프레임: add box prompt
            # - 다음 프레임: propagate / update
            #
            # 현재는 "fallback 수준"으로 crop_fg에서 bbox를 만드는 것으로 반환(최소 기능)
            fg = crop_fg if crop_fg.ndim == 2 else cv2.cvtColor(crop_fg, cv2.COLOR_BGR2GRAY)
            # bbox 주변만 조금 좁혀서 사용
            x0i, y0i, x1i, y1i = map(int, box.tolist())
            x0i = max(0, min(fg.shape[1] - 1, x0i))
            y0i = max(0, min(fg.shape[0] - 1, y0i))
            x1i = max(1, min(fg.shape[1], x1i))
            y1i = max(1, min(fg.shape[0], y1i))
            roi_fg_local = fg[y0i:y1i, x0i:x1i]
            if roi_fg_local.size == 0:
                return None

            # 가장 큰 blob
            contours, _ = cv2.findContours(roi_fg_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_roi = (x0i + x, y0i + y, x0i + x + w, y0i + y + h)

            mask = np.zeros_like(fg, dtype=np.uint8)
            cv2.drawContours(mask[y0i:y1i, x0i:x1i], [cnt], -1, 255, thickness=-1)

            return {"mask_roi": mask, "bbox_roi": bbox_roi, "quality": 0.6}
        except Exception:
            return None


def build_segmenter(cfg: Dict[str, Any], fps: float):
    s = cfg.get("segmentation", {})
    enabled = bool(s.get("enabled", True))
    if not enabled:
        # 완전 비활성: segment()가 아무것도 안함
        class Noop(Segmenter):
            def __init__(self): super().__init__(enabled=False)
            def segment(self, *args, **kwargs): return None
        return Noop()

    prefer = bool(s.get("prefer_sam2", True))
    if prefer:
        sam2_seg = Sam2RoiSegmenter(cfg, fps=fps)
        if sam2_seg.available:
            print("[SEG] SAM2 enabled (ROI mode)")
            return sam2_seg
        else:
            print(f"[SEG] SAM2 not available -> {sam2_seg._init_error} -> fallback(MotionMaskSegmenter)")
            return MotionMaskSegmenter(cfg)

    return MotionMaskSegmenter(cfg)
