"""
I/O utilities for the TrafficNet pipeline.

Functions
---------
load_frames      — iterate over a folder of images or a video file
save_video       — write annotated frames to an mp4 file
save_metrics_csv — dump per-frame and summary metrics to CSV
"""

import os
import cv2
import pandas as pd


def load_frames(source):
    """
    Yield (frame_id, bgr_frame) tuples from either:
      - a directory of sorted image files (.jpg / .jpeg / .png)
      - a video file (.mp4, .avi, etc.)

    Parameters
    ----------
    source : str — path to image directory or video file

    Yields
    ------
    (frame_id: int, frame: np.ndarray)
    """
    if os.path.isdir(source):
        image_files = sorted(
            f for f in os.listdir(source)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for frame_id, filename in enumerate(image_files, start=1):
            frame = cv2.imread(os.path.join(source, filename))
            if frame is not None:
                yield frame_id, frame
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
        cap.release()


def save_video(frames, output_path, fps=25):
    """
    Write a list of BGR frames to a video file.

    Tries mp4v (H.264-compatible) first; if the encoder silently fails
    (zero-byte output — common with conda OpenCV on macOS), falls back
    to MJPG inside an .avi container which is universally supported.

    Parameters
    ----------
    frames      : list of np.ndarray (BGR)
    output_path : str — e.g. 'output/result.mp4'
    fps         : int — frames per second
    """
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]

    def _try_write(path, fourcc_str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return os.path.exists(path) and os.path.getsize(path) > 0

    # Try mp4v first
    if _try_write(output_path, "mp4v"):
        print(f"Saved video → {output_path}  ({len(frames)} frames, {fps} fps)")
        return

    # Fall back to MJPG + .avi
    avi_path = os.path.splitext(output_path)[0] + ".avi"
    if _try_write(avi_path, "MJPG"):
        print(f"Saved video → {avi_path}  ({len(frames)} frames, {fps} fps)  [fallback: MJPG/avi]")
        return

    print(f"Warning: could not write video to {output_path} — no working codec found.")


def save_metrics_csv(per_frame_metrics, summary_metrics, output_path):
    """
    Write per-frame and summary metrics to a CSV file.

    Parameters
    ----------
    per_frame_metrics : list of dicts, one per frame, e.g.:
        [{"frame_id": 1, "active_tracks": 3, "avg_speed": 7.2,
          "density": 0.00005, "congested": False}, ...]
    summary_metrics   : dict with aggregate results, e.g.:
        {"total_crossings": {"line_north": 12},
         "lane_utilization": {"lane_left": 0.6, "lane_right": 0.4},
         "flow_rate": {"line_north": 5.2}}
    output_path       : str — e.g. 'output/metrics.csv'
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.DataFrame(per_frame_metrics)
    df.to_csv(output_path, index=False)

    # Append a summary block after a blank line
    summary_path = output_path.replace(".csv", "_summary.csv")
    rows = []
    for key, value in summary_metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                rows.append({"metric": f"{key}.{sub_key}", "value": sub_val})
        else:
            rows.append({"metric": key, "value": value})

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved metrics  → {output_path}")
    print(f"Saved summary  → {summary_path}")
