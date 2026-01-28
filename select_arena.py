# select_arena.py (rectangle selector, Enter to save)
import argparse
from pathlib import Path
import cv2
import yaml

state = {
    "dragging": False,
    "p0": None,   # (x,y)
    "p1": None,   # (x,y)
}

def load_first_frame(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        frames = [x for x in sorted(p.iterdir()) if x.suffix.lower() in exts]
        if not frames:
            raise FileNotFoundError(f"No images in folder: {p}")
        img = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read: {frames[0]}")
        return img

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {p}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read first frame from video")
    return frame

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["dragging"] = True
        state["p0"] = (x, y)
        state["p1"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
        state["p1"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state["dragging"] = False
        state["p1"] = (x, y)
        print(f"[RECT] p0={state['p0']} p1={state['p1']}  (Press Enter/S to save, R to reset, Esc/Q to quit)")

def rect_norm(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    xa, xb = sorted([x0, x1])
    ya, yb = sorted([y0, y1])
    return int(xa), int(ya), int(xb), int(yb)

def save_yaml(out_path: str, x0: int, y0: int, x1: int, y1: int):
    payload = {"arena_manual": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file or frames folder")
    ap.add_argument("--out", default="arena_manual.yaml", help="output yaml path")
    args = ap.parse_args()

    frame = load_first_frame(args.input)

    win = "Select Rect Arena (drag) - CLICK WINDOW to focus"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    print("[INFO] OpenCV window must be focused to receive key input.")
    print("       Drag to draw rectangle.")
    print("       Enter or S: save & exit | R: reset | Esc or Q: exit without saving")

    while True:
        # 창이 닫혔으면 종료
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        vis = frame.copy()

        if state["p0"] is not None and state["p1"] is not None:
            x0, y0, x1, y1 = rect_norm(state["p0"], state["p1"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"({x0},{y0})-({x1},{y1})  Enter/S=Save  R=Reset  Esc/Q=Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                vis,
                "Drag to draw rectangle. Enter/S=Save, R=Reset, Esc/Q=Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow(win, vis)

        # waitKey: 1ms라도 호출되어야 UI가 갱신되고 키 입력을 받음
        key = cv2.waitKey(20) & 0xFF

        # Enter: Windows에서 보통 13(\r)로 들어옴
        is_enter = (key == 13 or key == 10)

        if key in (27, ord("q"), ord("Q")):
            # quit without saving
            break

        if key in (ord("r"), ord("R")):
            state["dragging"] = False
            state["p0"] = None
            state["p1"] = None
            print("[RESET] drag again")
            continue

        if is_enter or key in (ord("s"), ord("S")):
            if state["p0"] is None or state["p1"] is None:
                print("[WARN] drag rectangle first")
                continue
            x0, y0, x1, y1 = rect_norm(state["p0"], state["p1"])
            save_yaml(args.out, x0, y0, x1, y1)
            print(f"[SAVED] {args.out}")
            print("Paste into config.yaml:")
            print(
                yaml.safe_dump(
                    {"detection": {"arena": {"manual_rect": {"enabled": True, "x0": x0, "y0": y0, "x1": x1, "y1": y1}}}},
                    allow_unicode=True,
                    sort_keys=False,
                )
            )
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
