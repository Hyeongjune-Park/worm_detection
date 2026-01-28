# sensors/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tracking.track import Track
from roi.roi_manager import RoiWindow


@dataclass
class SensorResult:
    """센서 관측 결과 (전체 프레임 좌표 기준)."""
    center: Optional[Tuple[float, float]] = None   # (x, y) full-frame
    confidence: float = 0.0                         # 0..1
    mask: Optional[np.ndarray] = None               # ROI 좌표 기준 0/255
    endpoints: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # full-frame
    border_touch: float = 0.0                       # mask가 ROI 경계에 닿는 비율 0..1
    bbox_roi: Optional[Tuple[int, int, int, int]] = None  # ROI 내 bbox
    area: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Sensor(ABC):
    """센서 공통 인터페이스."""

    @abstractmethod
    def initialize(self, track: Track, roi: RoiWindow,
                   crop_bgr: np.ndarray, crop_gray: np.ndarray) -> None:
        """첫 프레임(seed)에서 센서 상태 초기화."""
        ...

    @abstractmethod
    def measure(self, track: Track, roi: RoiWindow,
                crop_bgr: np.ndarray, crop_gray: np.ndarray,
                frame_idx: int) -> Optional[SensorResult]:
        """매 프레임 관측. 실패 시 None."""
        ...
