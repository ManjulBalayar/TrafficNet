"""
Analytics Evaluation Script
============================

Evaluates the ANALYTICS layer (counting + lane assignment) of the pipeline
against UA-DETRAC ground-truth annotations.

Metrics computed
----------------
1. Vehicle Counting Accuracy
   - A horizontal mid-frame counting line is placed at y = frame_height / 2.
     The SAME line is applied to both ground-truth and predicted trajectories,
     making the comparison fair and independent of scene-specific configuration.
   - Counting Accuracy = 1 - |pred_count - gt_count| / max(1, gt_count)
     (reported as a percentage; proposal target: ≥ 85 %)

2. Lane Assignment Accuracy (qualitative proxy)
   - Splits the frame into Left / Right halves (x < frame_width/2 vs x ≥ width/2).
   - For every GT track computes its majority half from GT positions.
   - For every matched predicted track computes its majority half.
   - Agreement rate = fraction of matched tracks where pred half == gt half.
   - (Real lane ROIs vary by scene; this gives a scene-agnostic estimate.)

Usage
-----
    python evaluate_analytics.py                    # all annotated sequences
    python evaluate_analytics.py --seq MVI_20011    # single sequence
"""

import os
import argparse
import xml.etree.ElementTree as ET

from detection.detector import Detector
from tracking.tracker import Tracker
from utils.io import load_frames

SEQUENCES_DIR   = "data/ua-detrac/sequences"
ANNOTATIONS_DIR = "data/ua-detrac/annotations"


# ── GT parser (same as evaluate.py) ──────────────────────────────────────────

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
            cx = l + w / 2
            cy = t + h / 2
            targets.append((gt_id, cx, cy, l, t, l + w, t + h))
        frames[num] = targets
    return frames


# ── Build GT trajectory store (dict matching TrajectoryStore._histories) ──────

def build_gt_store(gt_frames):
    """
    Returns {track_id: [(frame_id, cx, cy, 0.0, 0.0), ...]}
    """
    store = {}
    for frame_id, targets in sorted(gt_frames.items()):
        for (gt_id, cx, cy, *_) in targets:
            if gt_id not in store:
                store[gt_id] = []
            store[gt_id].append((frame_id, cx, cy, 0.0, 0.0))
    return store


# ── Counting logic (duplicated from analytics/counting.py for portability) ───

def _side(point, p1, p2):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    px = point[0] - p1[0]; py = point[1] - p1[1]
    cross = dx * py - dy * px
    return 1 if cross > 0 else (-1 if cross < 0 else 0)


def count_crossings_from_store(histories, line):
    """Count unique tracks that cross `line` = [(x1,y1),(x2,y2)]."""
    p1, p2  = line
    crossed = set()
    for tid, history in histories.items():
        positions = [(cx, cy) for _, cx, cy, _, _ in history]
        if len(positions) < 2:
            continue
        prev = _side(positions[0], p1, p2)
        for pos in positions[1:]:
            curr = _side(pos, p1, p2)
            if prev != 0 and curr != 0 and prev != curr:
                crossed.add(tid)
                break
            prev = curr
    return len(crossed)


# ── Lane assignment logic (left / right half of frame) ───────────────────────

def majority_half(positions, frame_width):
    """Return 'left' or 'right' based on where the track spent most time."""
    if not positions:
        return None
    left  = sum(1 for cx, _ in positions if cx < frame_width / 2)
    right = len(positions) - left
    return "left" if left >= right else "right"


def build_pred_store(tracker):
    """Extract the same dict shape from a Tracker's TrajectoryStore."""
    return tracker.trajectory_store.all_histories()


# ── IoU helper for GT–pred track matching ────────────────────────────────────

def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter)


# ── Per-sequence evaluator ────────────────────────────────────────────────────

def evaluate_sequence_analytics(seq_name):
    seq_dir  = os.path.join(SEQUENCES_DIR, seq_name)
    xml_path = os.path.join(ANNOTATIONS_DIR, f"{seq_name}.xml")

    if not os.path.isdir(seq_dir) or not os.path.isfile(xml_path):
        print(f"  [SKIP] {seq_name} — missing images or annotations")
        return None

    gt_frames = parse_annotation(xml_path)

    # Get frame dimensions from the first available image
    frame_w, frame_h = None, None
    for _, frame in load_frames(seq_dir):
        frame_h, frame_w = frame.shape[:2]
        break
    if frame_w is None:
        print(f"  [SKIP] {seq_name} — could not read frames")
        return None

    mid_line = [(0, frame_h // 2), (frame_w, frame_h // 2)]

    # 1. Ground-truth counts using GT trajectories
    gt_store    = build_gt_store(gt_frames)
    gt_count    = count_crossings_from_store(gt_store, mid_line)

    # 2. Run full pipeline to get predicted trajectories
    print(f"  Running pipeline on {seq_name} ({frame_w}×{frame_h}, "
          f"{len(gt_frames)} annotated frames)...")
    detector = Detector()
    tracker  = Tracker()
    frame_count = 0
    for frame_id, frame in load_frames(seq_dir):
        dets = detector.detect_frame_bgr(frame)
        tracker.update(frame_id, dets)
        frame_count += 1

    pred_store  = build_pred_store(tracker)
    pred_count  = count_crossings_from_store(pred_store, mid_line)

    # 3. Counting accuracy
    count_acc = max(0.0, 1.0 - abs(pred_count - gt_count) / max(1, gt_count)) * 100

    # 4. Lane assignment accuracy (left / right half of frame)
    #    Build GT per-track majority-half lookup
    gt_halves = {}
    for tid, history in gt_store.items():
        positions = [(cx, cy) for _, cx, cy, _, _ in history]
        gt_halves[tid] = majority_half(positions, frame_w)

    #    For each GT track, find a predicted track that overlaps it enough to match.
    #    Simple approach: find pred track whose bounding box (approximated as a circle
    #    around its trajectory centroid) is closest to the GT centroid, then compare halves.
    pred_centroids = {}
    for tid, history in pred_store.items():
        positions = [(cx, cy) for _, cx, cy, _, _ in history]
        if positions:
            mcx = sum(p[0] for p in positions) / len(positions)
            mcy = sum(p[1] for p in positions) / len(positions)
            pred_centroids[tid] = (mcx, mcy, majority_half(positions, frame_w))

    matched_agreements = 0
    matched_total      = 0
    for gt_tid, gt_half in gt_halves.items():
        gt_positions = [(cx, cy) for _, cx, cy, _, _ in gt_store[gt_tid]]
        if not gt_positions:
            continue
        gt_mcx = sum(p[0] for p in gt_positions) / len(gt_positions)
        gt_mcy = sum(p[1] for p in gt_positions) / len(gt_positions)

        # nearest predicted centroid
        best_dist  = float("inf")
        best_half  = None
        for _, (pcx, pcy, phalf) in pred_centroids.items():
            d = ((pcx - gt_mcx)**2 + (pcy - gt_mcy)**2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_half = phalf

        if best_dist < frame_w * 0.15 and best_half is not None:
            matched_total += 1
            if best_half == gt_half:
                matched_agreements += 1

    lane_acc = (matched_agreements / matched_total * 100) if matched_total > 0 else 0.0

    return {
        "sequence":      seq_name,
        "frame_w":       frame_w,
        "frame_h":       frame_h,
        "gt_count":      gt_count,
        "pred_count":    pred_count,
        "count_acc":     round(count_acc, 1),
        "lane_matched":  matched_total,
        "lane_agreed":   matched_agreements,
        "lane_acc":      round(lane_acc, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default=None, help="Single sequence, e.g. MVI_20011")
    args = parser.parse_args()

    if args.seq:
        sequences = [args.seq]
    else:
        sequences = sorted(
            d for d in os.listdir(SEQUENCES_DIR)
            if os.path.isdir(os.path.join(SEQUENCES_DIR, d))
            and os.path.isfile(os.path.join(ANNOTATIONS_DIR, f"{d}.xml"))
        )

    if not sequences:
        print("No sequences found.")
        return

    print(f"\nAnalytics Evaluation on {len(sequences)} sequence(s): {sequences}")
    print("(Counting line: horizontal mid-frame — same for GT and predictions)\n")

    all_results = []
    for seq in sequences:
        r = evaluate_sequence_analytics(seq)
        if r:
            all_results.append(r)

    if not all_results:
        return

    print()
    print("=" * 70)
    print("ANALYTICS EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Sequence':<14} {'GT':>5} {'Pred':>5} {'Count Acc':>10} {'Lane Acc':>10} {'Lane N':>7}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['sequence']:<14} {r['gt_count']:>5} {r['pred_count']:>5} "
              f"{r['count_acc']:>9.1f}% {r['lane_acc']:>9.1f}% {r['lane_matched']:>7}")

    if len(all_results) > 1:
        avg_count = sum(r["count_acc"] for r in all_results) / len(all_results)
        avg_lane  = sum(r["lane_acc"]  for r in all_results) / len(all_results)
        print("-" * 70)
        print(f"{'AVERAGE':<14} {'':>5} {'':>5} {avg_count:>9.1f}% {avg_lane:>9.1f}%")
    print("=" * 70)
    print()
    print("Targets from proposal:")
    print("  Vehicle Counting Accuracy ≥ 85% (Good)  |  ≥ 92% (Stretch)")
    print("  Lane Assignment Accuracy  ≥ 80% (Good)  |  ≥ 90% (Stretch)")
    print()


if __name__ == "__main__":
    main()
