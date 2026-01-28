# select_seeds.py — 초기 seed bbox 지정 도구
# 사용법: python select_seeds.py --input video.mp4 --arena arena.yaml --out seeds.yaml
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import yaml


class FrameLoader:
    """비디오 또는 이미지 폴더에서 프레임을 로드. 프레임 탐색 지원."""

    def __init__(self, input_path: str, start_frame: int = 0):
        self.p = Path(input_path)
        self.is_dir = self.p.is_dir()
        self._cap = None
        self._frames_list: List[Path] = []
        self.total_frames = 0
        self.current_idx = 0

        if self.is_dir:
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            self._frames_list = [x for x in sorted(self.p.iterdir()) if x.suffix.lower() in exts]
            if not self._frames_list:
                raise FileNotFoundError(f"No images in folder: {self.p}")
            self.total_frames = len(self._frames_list)
        else:
            self._cap = cv2.VideoCapture(str(self.p))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.p}")
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.current_idx = max(0, min(start_frame, self.total_frames - 1))

    def read(self, idx: int) -> 'np.ndarray':
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_idx = idx
        if self.is_dir:
            img = cv2.imread(str(self._frames_list[idx]), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read: {self._frames_list[idx]}")
            return img
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx}")
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()


def load_arena(arena_path: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not arena_path:
        return None
    with open(arena_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    am = data.get("arena_manual")
    if not isinstance(am, dict):
        return None
    return (int(am["x0"]), int(am["y0"]), int(am["x1"]), int(am["y1"]))


class SeedSelector:
    def __init__(self, frame, arena: Optional[Tuple[int, int, int, int]] = None, max_seeds: int = 10, frame_info: str = ""):
        self.frame = frame
        self.arena = arena
        self.max_seeds = max_seeds
        self.frame_info = frame_info
        self.boxes: List[Tuple[int, int, int, int]] = []  # (x0,y0,x1,y1) normalized
        self.dragging = False
        self.p0: Optional[Tuple[int, int]] = None
        self.p1: Optional[Tuple[int, int]] = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.boxes) >= self.max_seeds:
                print(f"[WARN] max {self.max_seeds} seeds reached")
                return
            self.dragging = True
            self.p0 = (x, y)
            self.p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.p1 = (x, y)
            box = self._norm(self.p0, self.p1)
            if box[2] - box[0] > 5 and box[3] - box[1] > 5:
                self.boxes.append(box)
                tid = len(self.boxes)
                print(f"[SEED] track_id={tid} bbox={box}  ({len(self.boxes)}/{self.max_seeds})")
            else:
                print("[WARN] box too small, ignored")
            self.p0 = None
            self.p1 = None

    @staticmethod
    def _norm(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])
        return (int(xa), int(ya), int(xb), int(yb))

    def draw(self):
        vis = self.frame.copy()
        # arena
        if self.arena:
            ax0, ay0, ax1, ay1 = self.arena
            cv2.rectangle(vis, (ax0, ay0), (ax1, ay1), (255, 255, 0), 2)
            cv2.putText(vis, "arena", (ax0, ay0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # existing boxes
        for i, (x0, y0, x1, y1) in enumerate(self.boxes):
            tid = i + 1
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, f"id={tid}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # current drag
        if self.p0 is not None and self.p1 is not None:
            box = self._norm(self.p0, self.p1)
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 200, 255), 2)

        info = f"Seeds: {len(self.boxes)}/{self.max_seeds}  |  LMB=draw  U=undo  </>frame  Enter/S=save  Esc/Q=quit"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if self.frame_info:
            cv2.putText(vis, self.frame_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return vis

    def undo(self):
        if self.boxes:
            removed = self.boxes.pop()
            print(f"[UNDO] removed last box {removed}, remaining={len(self.boxes)}")
        else:
            print("[UNDO] no boxes to undo")


def save_seeds(out_path: str, boxes: List[Tuple[int, int, int, int]], frame_idx: int = 0):
    seeds = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        seeds.append({"track_id": i + 1, "bbox": [x0, y0, x1, y1]})
    payload = {"frame_idx": frame_idx, "seeds": seeds}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="Seed selection tool for insect tracking")
    ap.add_argument("--input", required=True, help="video file or frames folder")
    ap.add_argument("--arena", default="", help="arena yaml (optional, for visualization)")
    ap.add_argument("--out", default="seeds.yaml", help="output seeds yaml path")
    ap.add_argument("--max-seeds", type=int, default=10)
    ap.add_argument("--frame", type=int, default=0, help="start frame index (0-based)")
    args = ap.parse_args()

    loader = FrameLoader(args.input, start_frame=args.frame)
    arena = load_arena(args.arena) if args.arena else None

    cur_idx = loader.current_idx
    frame = loader.read(cur_idx)
    frame_info = f"Frame {cur_idx}/{loader.total_frames - 1}"
    sel = SeedSelector(frame, arena=arena, max_seeds=args.max_seeds, frame_info=frame_info)

    win = "Select Seeds (draw boxes) - CLICK WINDOW to focus"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, sel.mouse_cb)

    print("[INFO] Draw bounding boxes around each larva.")
    print("       LMB drag = draw box | U = undo last")
    print("       Left/Right arrow or ,/. = prev/next frame")
    print("       Enter/S = save | Esc/Q = quit")

    while True:
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        vis = sel.draw()
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key in (ord("u"), ord("U")):
            sel.undo()
            continue

        # Frame navigation: left arrow (81), right arrow (83), comma, period
        frame_changed = False
        if key in (81, ord(","), ord("<")):
            # previous frame
            if cur_idx > 0:
                cur_idx -= 1
                frame_changed = True
        elif key in (83, ord("."), ord(">")):
            # next frame
            if cur_idx < loader.total_frames - 1:
                cur_idx += 1
                frame_changed = True

        if frame_changed:
            frame = loader.read(cur_idx)
            sel.frame = frame
            sel.frame_info = f"Frame {cur_idx}/{loader.total_frames - 1}"
            continue

        is_enter = (key == 13 or key == 10)
        if is_enter or key in (ord("s"), ord("S")):
            if not sel.boxes:
                print("[WARN] draw at least one seed box")
                continue
            save_seeds(args.out, sel.boxes, frame_idx=cur_idx)
            print(f"[SAVED] {args.out} ({len(sel.boxes)} seeds, frame={cur_idx})")
            break

    loader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
