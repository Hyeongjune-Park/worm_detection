# tracking/track.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Any, Optional, Tuple
from collections import deque

import numpy as np

from tracking.kalman import KalmanFilter2D


class TrackState(str, Enum):
    ACTIVE = "ACTIVE"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    MERGED = "MERGED"
    EXITED = "EXITED"
    DEAD_CANDIDATE = "DEAD_CANDIDATE"
    NEEDS_RESEED = "NEEDS_RESEED"


@dataclass
class Track:
    id: int
    kf: KalmanFilter2D

    state: TrackState = TrackState.ACTIVE

    # --- 좌표 (핵심 출력) ---
    last_center: Optional[Tuple[float, float]] = None
    last_head: Optional[Tuple[float, float]] = None
    head_direction: Optional[np.ndarray] = None  # unit vector

    # --- bbox (디버그용) ---
    bbox: Optional[Tuple[int, int, int, int]] = None
    area: Optional[int] = None
    seed_bbox_size: Optional[Tuple[int, int]] = None  # (width, height) from initial seed

    # --- 센서 메모리 ---
    template: Optional[np.ndarray] = None           # gray+edge patch
    template_edge: Optional[np.ndarray] = None      # edge-only patch
    template_center: Optional[Tuple[float, float]] = None  # center when template captured
    klt_points: Optional[np.ndarray] = None         # Nx2 float32
    klt_prev_gray: Optional[np.ndarray] = None      # previous ROI gray crop

    # --- 품질/상태 ---
    quality_score: float = 1.0
    quality_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    sensor_used: str = "PRED"
    miss_count: int = 0
    reseed_count: int = 0
    last_good_frame_idx: int = 0
    last_seen_frame: int = 0

    # --- 속도 ---
    speed_px_s: Optional[float] = None
    _speed_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def center(self) -> Tuple[Optional[float], Optional[float]]:
        """last_center 우선, 없으면 Kalman 예측."""
        if self.last_center is not None:
            return self.last_center
        return self.kf.get_position()

    def update_speed(self, fps: float):
        vx, vy = self.kf.get_velocity()
        sp = float((vx * vx + vy * vy) ** 0.5) * float(fps)
        self.speed_px_s = sp
        self._speed_hist.append(sp)

    def mean_speed(self, window: int = 60) -> float:
        if not self._speed_hist:
            return 1e9
        hist = list(self._speed_hist)[-window:]
        return float(sum(hist) / len(hist))

    def serialize_state(self) -> Dict[str, Any]:
        """청크 간 상태 직렬화용."""
        return {
            "id": self.id,
            "kf_x": self.kf.x.tolist(),
            "kf_P": self.kf.P.tolist(),
            "state": self.state.value,
            "last_center": self.last_center,
            "last_head": self.last_head,
            "quality_score": self.quality_score,
            "miss_count": self.miss_count,
            "reseed_count": self.reseed_count,
            "last_good_frame_idx": self.last_good_frame_idx,
            "sensor_used": self.sensor_used,
        }
