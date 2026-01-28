# tracking/state_io.py — 청크 간 Track 상태 직렬화/역직렬화
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tracking.track import Track, TrackState
from tracking.kalman import KalmanFilter2D


def save_state(tracks: List[Track], path: str | Path) -> None:
    """트랙 리스트를 JSON으로 직렬화하여 저장."""
    states = [t.serialize_state() for t in tracks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tracks": states}, f, indent=2, ensure_ascii=False)


def load_state(path: str | Path, cfg: Dict[str, Any]) -> List[Track]:
    """JSON에서 트랙 리스트를 복원."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kcfg = cfg.get("tracking", {}).get("kalman", {})
    q = float(kcfg.get("process_noise_q", 2.0))
    r = float(kcfg.get("default_measurement_noise_r", 25.0))

    tracks = []
    for s in data.get("tracks", []):
        kf_x = np.array(s["kf_x"], dtype=np.float64)
        kf_P = np.array(s["kf_P"], dtype=np.float64)

        cx, cy = float(kf_x[0]), float(kf_x[1])
        kf = KalmanFilter2D(cx, cy, q=q, r=r)
        kf.x = kf_x.reshape(-1, 1) if kf_x.ndim == 1 else kf_x
        kf.P = kf_P

        tr = Track(id=int(s["id"]), kf=kf)
        tr.state = TrackState(s["state"])
        tr.last_center = tuple(s["last_center"]) if s.get("last_center") else None
        tr.last_head = tuple(s["last_head"]) if s.get("last_head") else None
        tr.quality_score = float(s.get("quality_score", 1.0))
        tr.miss_count = int(s.get("miss_count", 0))
        tr.reseed_count = int(s.get("reseed_count", 0))
        tr.last_good_frame_idx = int(s.get("last_good_frame_idx", 0))
        tr.sensor_used = str(s.get("sensor_used", "PRED"))
        tracks.append(tr)

    return tracks
