# io_utils/artifacts.py
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tracking.track import Track


_CSV_FIELDS = [
    "frame_idx",
    "time_sec",
    "track_id",
    "center_x",
    "center_y",
    "head_x",
    "head_y",
    "state",
    "quality_score",
    "sensor_used",
    "roi_x0",
    "roi_y0",
    "roi_x1",
    "roi_y1",
    "speed_px_s",
    "area",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
]


class ArtifactWriter:
    def __init__(self, out_dir: Path, save_csv: bool = True, fps: float = 30.0):
        self.out_dir = out_dir
        self.save_csv = bool(save_csv)
        self.fps = float(fps)

        self._csv_path = out_dir / "tracks.csv"
        self._csv_f = open(self._csv_path, "w", newline="", encoding="utf-8") if self.save_csv else None
        self._csv_writer = None

        if self._csv_f is not None:
            self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=_CSV_FIELDS)
            self._csv_writer.writeheader()

    def write_frame(self, frame_idx: int, tracks: List[Track],
                    frame_size: Tuple[int, int], fps: float) -> None:
        if self._csv_writer is None:
            return

        time_sec = frame_idx / max(1.0, fps)

        for t in tracks:
            cx, cy = t.center()
            hx = t.last_head[0] if t.last_head else None
            hy = t.last_head[1] if t.last_head else None
            bx0, by0, bx1, by1 = t.bbox if t.bbox else (None, None, None, None)

            self._csv_writer.writerow({
                "frame_idx": frame_idx,
                "time_sec": f"{time_sec:.4f}",
                "track_id": t.id,
                "center_x": f"{cx:.3f}" if cx is not None else "",
                "center_y": f"{cy:.3f}" if cy is not None else "",
                "head_x": f"{hx:.3f}" if hx is not None else "",
                "head_y": f"{hy:.3f}" if hy is not None else "",
                "state": t.state.value,
                "quality_score": f"{t.quality_score:.3f}",
                "sensor_used": t.sensor_used,
                "roi_x0": "",  # ROI는 runner에서 매 프레임 계산, 여기서는 빈값
                "roi_y0": "",
                "roi_x1": "",
                "roi_y1": "",
                "speed_px_s": f"{t.speed_px_s:.3f}" if t.speed_px_s is not None else "",
                "area": int(t.area) if t.area is not None else "",
                "bbox_x0": int(bx0) if bx0 is not None else "",
                "bbox_y0": int(by0) if by0 is not None else "",
                "bbox_x1": int(bx1) if bx1 is not None else "",
                "bbox_y1": int(by1) if by1 is not None else "",
            })

    def close(self) -> None:
        if self._csv_f is not None:
            self._csv_f.close()
            self._csv_f = None
            self._csv_writer = None


class EventWriter:
    """events.jsonl: 상태 전이, merge, reseed 등 이벤트 로그."""

    def __init__(self, path: Path):
        self._f = open(path, "w", encoding="utf-8")

    def log(self, frame_idx: int, track_id: int, event_type: str,
            details: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "frame": frame_idx,
            "track_id": track_id,
            "event": event_type,
            "details": details or {},
        }
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._f.close()
