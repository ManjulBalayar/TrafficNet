"""
All OpenCV drawing helpers for the TrafficNet pipeline.

Functions
---------
draw_tracks        — bounding boxes + track ID + class label
draw_trails        — trajectory history as fading polylines
draw_counting_lines — virtual counting lines with live counts
draw_lane_rois     — lane polygons with utilization labels
draw_congestion    — top-left status overlay
"""

import cv2
import numpy as np

# Deterministic per-ID color so the same track always gets the same color
def _track_color(track_id):
    rng = np.random.default_rng(seed=track_id * 7919)
    return tuple(int(c) for c in rng.integers(80, 230, size=3))


def draw_tracks(frame, track_outputs):
    """
    Draw bounding boxes and labels for confirmed tracks.

    Parameters
    ----------
    frame        : BGR numpy array (modified in-place)
    track_outputs: list of dicts from track.get_output()
    """
    for t in track_outputs:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        color = _track_color(t["track_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID{t['track_id']} {t['class_name']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_trails(frame, trajectory_store, tail_length=30):
    """
    Draw the recent trajectory trail for each track as a fading polyline.

    Parameters
    ----------
    frame            : BGR numpy array (modified in-place)
    trajectory_store : TrajectoryStore instance
    tail_length      : how many past positions to draw
    """
    for track_id, history in trajectory_store.all_histories().items():
        positions = [(int(cx), int(cy)) for _, cx, cy, _, _ in history]
        tail = positions[-tail_length:]
        if len(tail) < 2:
            continue
        color = _track_color(track_id)
        for i in range(1, len(tail)):
            # fade older segments toward black
            alpha = i / len(tail)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(frame, tail[i - 1], tail[i], faded, 2, cv2.LINE_AA)
    return frame


def draw_counting_lines(frame, counting_lines, crossing_results=None):
    """
    Draw virtual counting lines and optionally their live crossing counts.

    Parameters
    ----------
    frame           : BGR numpy array (modified in-place)
    counting_lines  : list of {"name": str, "line": [(x1,y1),(x2,y2)]}
    crossing_results: output of count_crossings(), or None
    """
    for cl in counting_lines:
        (x1, y1), (x2, y2) = cl["line"]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        count = 0
        if crossing_results and cl["name"] in crossing_results:
            count = crossing_results[cl["name"]]["count"]
        label = f"{cl['name']}: {count}"
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_lane_rois(frame, lane_rois, utilization=None):
    """
    Draw lane ROI polygons with optional utilization percentages.

    Parameters
    ----------
    frame       : BGR numpy array (modified in-place)
    lane_rois   : list of {"name": str, "polygon": [(x,y), ...]}
    utilization : output of lane_utilization(), or None
    """
    overlay = frame.copy()
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
              (255, 255, 100), (100, 255, 255)]

    for idx, roi in enumerate(lane_rois):
        pts = np.array(roi["polygon"], dtype=np.int32)
        color = colors[idx % len(colors)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        pct = utilization.get(roi["name"], 0.0) * 100 if utilization else 0.0
        label = f"{roi['name']}: {pct:.0f}%"
        cv2.putText(frame, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


def draw_congestion(frame, congestion_status):
    """
    Draw a top-left status overlay with speed, density, and congestion flag.

    Parameters
    ----------
    frame             : BGR numpy array (modified in-place)
    congestion_status : output of CongestionDetector.status()
    """
    congested  = congestion_status["congested"]
    avg_speed  = congestion_status["avg_speed"]
    density    = congestion_status["density"]

    bg_color  = (0, 0, 180) if congested else (0, 120, 0)
    label     = "CONGESTED" if congested else "CLEAR"

    cv2.rectangle(frame, (8, 8), (260, 70), bg_color, -1)
    cv2.putText(frame, label,
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"speed: {avg_speed:.1f} px/fr  density: {density:.5f}",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return frame
