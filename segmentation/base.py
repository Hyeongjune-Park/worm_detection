# segmentation/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from tracking.track import Track
from roi.roi_manager import RoiWindow


class Segmenter(ABC):
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    @abstractmethod
    def segment(
        self,
        track: Track,
        roi: RoiWindow,
        crop_bgr,
        crop_fg,
        frame_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        반환 dict 예시:
          {
            "mask_roi": uint8 (0/255) or None,
            "bbox_roi": (x0,y0,x1,y1) or None,
            "quality": float or None,
          }
        """
        raise NotImplementedError
