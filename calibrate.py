"""
Interactive Scene Calibrator
Point-and-click tool to position counting lines and lane ROI zones
on any video, then save the result as a scene config file.

Usage:
    python calibrate.py ~/Downloads/traffic2.mp4 --scene traffic2
    python calibrate.py ~/Downloads/traffic.mp4  --scene traffic_india

Controls:
    LEFT CLICK + DRAG   — move the nearest line endpoint or zone corner
    S                   — save current positions to scenes/<scene>.py
    R                   — reload / reset to saved positions
    N / P               — next / previous frame
    Q / ESC             — quit without saving
    H                   — toggle this help overlay

What you see:
    Yellow lines   = counting lines  (4 lines: north, south, east, west)
    Colored boxes  = lane ROI zones  (up to 4 zones)
    White circles  = draggable handles on every endpoint / corner
"""

import os
import sys
import importlib
import argparse
import textwrap
import cv2
import numpy as np

HANDLE_RADIUS = 12   # px — how close the click must be to grab a handle
COLORS = {
    "north": (0,   255, 255),
    "south": (0,   255, 255),
    "east":  (0,   255, 255),
    "west":  (0,   255, 255),
}
ROI_COLORS = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
]


def load_scene(scene_name):
    """Import scene module and return mutable copies of lines and rois."""
    import config as base
    try:
        mod = importlib.import_module(f"scenes.{scene_name}")
    except ModuleNotFoundError:
        mod = None

    def _get(attr):
        if mod and hasattr(mod, attr):
            return getattr(mod, attr)
        return getattr(base, attr)

    lines = [
        {"name": cl["name"], "line": [list(cl["line"][0]), list(cl["line"][1])]}
        for cl in _get("COUNTING_LINES")
    ]
    rois = [
        {"name": r["name"], "polygon": [list(p) for p in r["polygon"]]}
        for r in _get("LANE_ROIS")
    ]
    return lines, rois


def save_scene(scene_name, lines, rois):
    path = os.path.join("scenes", f"{scene_name}.py")
    os.makedirs("scenes", exist_ok=True)

    with open(path, "w") as f:
        f.write(f'"""\nScene config: {scene_name} — calibrated interactively.\n\nRun with:\n')
        f.write(f'    python main.py <video> --scene {scene_name} --show\n"""\n\n')

        f.write("COUNTING_LINES = [\n")
        for cl in lines:
            p1, p2 = cl["line"]
            f.write(f'    {{"name": "{cl["name"]}", '
                    f'"line": [({p1[0]}, {p1[1]}), ({p2[0]}, {p2[1]})]}},\n')
        f.write("]\n\n")

        f.write("LANE_ROIS = [\n")
        for roi in rois:
            pts = ", ".join(f"({p[0]}, {p[1]})" for p in roi["polygon"])
            f.write(f'    {{"name": "{roi["name"]}", "polygon": [{pts}]}},\n')
        f.write("]\n\n")

        f.write("V_MIN   = 5.0\n")
        f.write("RHO_MAX = 0.000004\n")

    print(f"\n  Saved → {path}\n")


# Drag state 

class DragState:
    def __init__(self):
        self.active = False
        self.kind   = None   # "line" or "roi"
        self.idx    = None   # index into lines / rois list
        self.pt_idx = None   # 0 or 1 for lines; 0-3 for roi corners

    def reset(self):
        self.active = False
        self.kind = self.idx = self.pt_idx = None


drag = DragState()


def all_handles(lines, rois):
    """Yield (kind, list_idx, pt_idx, (x,y)) for every handle."""
    for li, cl in enumerate(lines):
        for pi, pt in enumerate(cl["line"]):
            yield ("line", li, pi, tuple(pt))
    for ri, roi in enumerate(rois):
        for pi, pt in enumerate(roi["polygon"]):
            yield ("roi", ri, pi, tuple(pt))


def nearest_handle(mx, my, lines, rois):
    best_dist = HANDLE_RADIUS * 3
    best = None
    for kind, li, pi, pt in all_handles(lines, rois):
        d = ((pt[0]-mx)**2 + (pt[1]-my)**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best = (kind, li, pi)
    return best


def mouse_cb(event, mx, my, flags, param):
    lines, rois = param
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = nearest_handle(mx, my, lines, rois)
        if hit:
            drag.active  = True
            drag.kind, drag.idx, drag.pt_idx = hit
    elif event == cv2.EVENT_MOUSEMOVE and drag.active:
        if drag.kind == "line":
            lines[drag.idx]["line"][drag.pt_idx] = [mx, my]
        else:
            rois[drag.idx]["polygon"][drag.pt_idx] = [mx, my]
    elif event == cv2.EVENT_LBUTTONUP:
        drag.reset()


def draw_scene(frame, lines, rois, show_help):
    out = frame.copy()

    # ROI fills
    overlay = out.copy()
    for ri, roi in enumerate(rois):
        pts = np.array(roi["polygon"], dtype=np.int32)
        color = ROI_COLORS[ri % len(ROI_COLORS)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(out, roi["name"], (cx-40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        # draggable corners
        for pt in roi["polygon"]:
            cv2.circle(out, tuple(pt), HANDLE_RADIUS, color, -1)
            cv2.circle(out, tuple(pt), HANDLE_RADIUS, (255,255,255), 2)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    # Counting lines
    for cl in lines:
        p1 = tuple(cl["line"][0])
        p2 = tuple(cl["line"][1])
        cv2.line(out, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, cl["name"], (p1[0]+4, p1[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        for pt in [p1, p2]:
            cv2.circle(out, pt, HANDLE_RADIUS, (0,255,255), -1)
            cv2.circle(out, pt, HANDLE_RADIUS, (255,255,255), 2)

    # Help
    if show_help:
        help_lines = [
            "DRAG white circles to move lines / zones",
            "S = save    R = reload    H = hide help",
            "N / P = next / prev frame    Q = quit",
        ]
        y = out.shape[0] - 30 * len(help_lines) - 10
        for hl in help_lines:
            cv2.rectangle(out, (8, y-20), (8 + len(hl)*11, y+6), (0,0,0), -1)
            cv2.putText(out, hl, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,200), 1)
            y += 30

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video",  help="Path to video file")
    parser.add_argument("--scene", required=True,
                        help="Scene name (e.g. traffic2) — saves to scenes/<name>.py")
    args = parser.parse_args()

    # Load all frames (small videos only) or seek
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}×{h}, {total} frames")

    frames = []
    print("Loading frames (Ctrl-C to stop loading early and use what's available)...")
    try:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
            if len(frames) % 50 == 0:
                print(f"  {len(frames)}/{total}", end="\r")
    except KeyboardInterrupt:
        pass
    cap.release()
    print(f"\nLoaded {len(frames)} frames.")

    lines, rois = load_scene(args.scene)

    win = "TrafficNet Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1600), min(h, 900))
    cv2.setMouseCallback(win, mouse_cb, (lines, rois))

    fi = 0
    show_help = True

    while True:
        frame = frames[fi] if frames else np.zeros((h, w, 3), dtype=np.uint8)
        out   = draw_scene(frame, lines, rois, show_help)

        info = f"Frame {fi+1}/{len(frames)}  |  {args.scene}  |  {w}x{h}"
        cv2.rectangle(out, (0, 0), (len(info)*9+10, 24), (0,0,0), -1)
        cv2.putText(out, info, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow(win, out)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_scene(args.scene, lines, rois)
        elif key == ord('r'):
            lines, rois = load_scene(args.scene)
            cv2.setMouseCallback(win, mouse_cb, (lines, rois))
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('n'):
            fi = min(fi + 1, len(frames) - 1)
        elif key == ord('p'):
            fi = max(fi - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
