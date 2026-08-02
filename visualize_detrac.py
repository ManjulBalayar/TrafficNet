"""
Generate annotated sample frames from UA-DETRAC sequences.

Saves side-by-side images showing:
  - Left panel:  raw frame with GT bounding boxes (green) + mid-frame counting line
  - Right panel: raw frame with predicted tracks (colored) + mid-frame line + left/right zones

Usage:
    python visualize_detrac.py --seq MVI_20011 --n 12 --output output/detrac_viz/
"""

import os
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np

from detection.detector import Detector
from tracking.tracker import Tracker
from utils.io import load_frames

SEQUENCES_DIR   = "data/ua-detrac/sequences"
ANNOTATIONS_DIR = "data/ua-detrac/annotations"


def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = {}
    for frame_elem in root.findall("frame"):
        num = int(frame_elem.get("num"))
        targets = []
        for target in frame_elem.find("target_list").findall("target"):
            gt_id = int(target.get("id"))
            box   = target.find("box")
            l, t  = float(box.get("left")), float(box.get("top"))
            w, h  = float(box.get("width")), float(box.get("height"))
            targets.append((gt_id, int(l), int(t), int(l+w), int(t+h)))
        frames[num] = targets
    return frames


def draw_gt_panel(frame, gt_boxes, mid_y, frame_w):
    """Left panel: GT boxes in green + counting line."""
    out = frame.copy()

    # GT bounding boxes
    for gt_id, x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"GT{gt_id}", (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # Mid-frame counting line
    cv2.line(out, (0, mid_y), (frame_w, mid_y), (0, 255, 255), 2)
    cv2.putText(out, "counting line (y=mid)", (8, mid_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Label
    cv2.rectangle(out, (0, 0), (160, 22), (30, 30, 30), -1)
    cv2.putText(out, "GROUND TRUTH", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    return out


def draw_pred_panel(frame, track_outputs, mid_y, frame_w, frame_h):
    """Right panel: predicted tracks (colored) + counting line + L/R zone overlay."""
    out = frame.copy()
    overlay = out.copy()

    # Left/right zone fill
    cv2.rectangle(overlay, (0, 0), (frame_w//2, frame_h), (255, 180, 80), -1)   # left = orange
    cv2.rectangle(overlay, (frame_w//2, 0), (frame_w, frame_h), (80, 180, 255), -1)  # right = blue
    cv2.addWeighted(overlay, 0.10, out, 0.90, 0, out)

    # Zone divider
    cv2.line(out, (frame_w//2, 0), (frame_w//2, frame_h), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, "LEFT zone", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 80), 1)
    cv2.putText(out, "RIGHT zone", (frame_w//2 + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1)

    # Predicted track boxes
    for t in track_outputs:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        rng = np.random.default_rng(seed=t["track_id"] * 7919)
        color = tuple(int(c) for c in rng.integers(80, 230, size=3))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID{t['track_id']}", (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Mid-frame counting line
    cv2.line(out, (0, mid_y), (frame_w, mid_y), (0, 255, 255), 2)
    cv2.putText(out, "counting line (y=mid)", (8, mid_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Label
    cv2.rectangle(out, (0, frame_h-22), (170, frame_h), (30, 30, 30), -1)
    cv2.putText(out, "PREDICTED (our tracker)", (4, frame_h-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1)
    return out


def run(seq_name, n_frames, output_dir):
    seq_dir  = os.path.join(SEQUENCES_DIR, seq_name)
    xml_path = os.path.join(ANNOTATIONS_DIR, f"{seq_name}.xml")

    if not os.path.isdir(seq_dir) or not os.path.isfile(xml_path):
        print(f"[SKIP] {seq_name} — missing images or annotations")
        return

    gt_frames = parse_annotation(xml_path)
    os.makedirs(output_dir, exist_ok=True)

    detector = Detector()
    tracker  = Tracker()

    all_frames = list(load_frames(seq_dir))
    total = len(all_frames)

    # Sample evenly across the sequence
    if n_frames >= total:
        indices = list(range(total))
    else:
        step = total // n_frames
        indices = [i * step for i in range(n_frames)]

    print(f"Processing {seq_name} ({total} frames), saving {len(indices)} samples...")

    # Run full tracker pass first so tracks are established at the sampled frames
    track_at = {}
    for frame_id, frame in all_frames:
        dets = detector.detect_frame_bgr(frame)
        outputs = tracker.update(frame_id, dets)
        track_at[frame_id] = (frame, outputs)

    # Now render the sampled frames
    saved = 0
    for idx in indices:
        frame_id, frame = all_frames[idx]
        frame_h, frame_w = frame.shape[:2]
        mid_y = frame_h // 2

        gt_boxes = gt_frames.get(frame_id, [])
        _, track_outputs = track_at[frame_id]

        gt_panel   = draw_gt_panel(frame, gt_boxes, mid_y, frame_w)
        pred_panel = draw_pred_panel(frame, track_outputs, mid_y, frame_w, frame_h)

        combined = np.hstack([gt_panel, pred_panel])

        # Header bar
        header = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
        header[:] = (40, 40, 40)
        cv2.putText(header,
                    f"{seq_name}  |  frame {frame_id:04d}/{total}  |  "
                    f"GT vehicles: {len(gt_boxes)}  |  "
                    f"Predicted tracks: {len(track_outputs)}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        out_img = np.vstack([header, combined])
        out_path = os.path.join(output_dir, f"{seq_name}_frame{frame_id:04d}.jpg")
        cv2.imwrite(out_path, out_img)
        saved += 1
        print(f"  Saved: {out_path}")

    print(f"\nDone — {saved} images in {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",    default=None, help="Sequence name (default: all)")
    parser.add_argument("--n",      type=int, default=12, help="Number of frames to sample")
    parser.add_argument("--output", default="output/detrac_viz", help="Output directory")
    args = parser.parse_args()

    if args.seq:
        sequences = [args.seq]
    else:
        sequences = sorted(
            d for d in os.listdir(SEQUENCES_DIR)
            if os.path.isdir(os.path.join(SEQUENCES_DIR, d))
            and os.path.isfile(os.path.join(ANNOTATIONS_DIR, f"{d}.xml"))
        )

    for seq in sequences:
        run(seq, args.n, os.path.join(args.output, seq))


if __name__ == "__main__":
    main()
