"""
UA-DETRAC Evaluation Script
Runs the TrafficNet pipeline on UA-DETRAC sequences and computes:

  MOTA  = 1 - (FN + FP + IDSW) / GT       (higher is better, can be negative)
  IDSW  = identity switches                 (lower is better)
  Frag  = track fragmentations              (lower is better)
  Recall    = TP / GT
  Precision = TP / (TP + FP)

Usage:
    python evaluate.py                          # full pipeline, all sequences
    python evaluate.py --seq MVI_20011          # single sequence
    python evaluate.py --baseline no_kalman     # position-only tracker
    python evaluate.py --baseline greedy        # greedy association
    python evaluate.py --baseline detection_only
    python evaluate.py --compare                # run all 4 conditions, print summary table
"""

import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.detector import Detector
from tracking.tracker import Tracker
from tracking.baselines import BaselineTracker, DetectionOnlyTracker
from utils.io import load_frames

# IoU threshold to accept a prediction–GT match (standard MOT threshold)
MATCH_IOU = 0.5

SEQUENCES_DIR   = "data/ua-detrac/sequences"
ANNOTATIONS_DIR = "data/ua-detrac/annotations"


# GT Parser 

def parse_annotation(xml_path):
    """
    Parse a UA-DETRAC XML file.

    Returns
    -------
    frames       : {frame_num (int): [(gt_id, [x1,y1,x2,y2]), ...]}
    ignored_boxes: [[x1,y1,x2,y2], ...]   regions excluded from scoring
    metadata     : {"weather": str, "camera_state": str}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Sequence metadata
    seq_attr = root.find("sequence_attribute")
    metadata = {
        "weather":      seq_attr.get("sence_weather", "unknown"),
        "camera_state": seq_attr.get("camera_state",  "unknown"),
    }

    # Ignored regions → convert ltwh → xyxy
    ignored_boxes = []
    ignored_region = root.find("ignored_region")
    if ignored_region is not None:
        for box in ignored_region.findall("box"):
            l, t = float(box.get("left")), float(box.get("top"))
            w, h = float(box.get("width")), float(box.get("height"))
            ignored_boxes.append([l, t, l + w, t + h])

    # Per-frame annotations
    frames = {}
    for frame_elem in root.findall("frame"):
        num = int(frame_elem.get("num"))
        targets = []
        for target in frame_elem.find("target_list").findall("target"):
            gt_id = int(target.get("id"))
            box   = target.find("box")
            l, t  = float(box.get("left")), float(box.get("top"))
            w, h  = float(box.get("width")), float(box.get("height"))
            targets.append((gt_id, [l, t, l + w, t + h]))
        frames[num] = targets

    return frames, ignored_boxes, metadata


# IoU helper 

def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _in_ignored(bbox, ignored_boxes, threshold=0.5):
    return any(_iou(bbox, ig) >= threshold for ig in ignored_boxes)


# Per-frame matching

def match_frame(gt_boxes, pred_boxes, iou_thresh=MATCH_IOU):
    """
    Hungarian matching between GT and predictions.
    Returns list of (gt_idx, pred_idx) matches with IoU >= iou_thresh.
    """
    if not gt_boxes or not pred_boxes:
        return []
    cost = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            cost[i, j] = 1.0 - _iou(g, p)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= 1.0 - iou_thresh]


# MOTA accumulator 

class MOTAAccumulator:
    def __init__(self):
        self.total_gt   = 0
        self.total_tp   = 0
        self.total_fp   = 0
        self.total_fn   = 0
        self.total_idsw = 0
        self.total_frag = 0
        # gt_id → last matched pred_id
        self._gt_to_pred = {}
        # gt_id → was matched last frame (for fragmentation)
        self._gt_matched_prev = {}

    def update(self, gt_entries, pred_entries, ignored_boxes):
        """
        gt_entries   : [(gt_id, bbox), ...]
        pred_entries : [(pred_id, bbox), ...]
        ignored_boxes: [[x1,y1,x2,y2], ...]
        """
        # Filter out GT boxes that fall inside ignored regions
        active_gt = [(gid, b) for gid, b in gt_entries
                     if not _in_ignored(b, ignored_boxes)]

        # Filter out predictions inside ignored regions (don't penalise as FP)
        active_pred = [(pid, b) for pid, b in pred_entries
                       if not _in_ignored(b, ignored_boxes)]

        gt_ids   = [g[0] for g in active_gt]
        gt_boxes = [g[1] for g in active_gt]
        pred_ids = [p[0] for p in active_pred]
        pred_boxes = [p[1] for p in active_pred]

        matches = match_frame(gt_boxes, pred_boxes)
        matched_gt   = {r for r, _ in matches}
        matched_pred = {c for _, c in matches}

        n_gt   = len(active_gt)
        n_fp   = len(active_pred) - len(matches)
        n_fn   = n_gt - len(matches)

        # Identity switches
        n_idsw = 0
        for r, c in matches:
            gid = gt_ids[r]
            pid = pred_ids[c]
            if gid in self._gt_to_pred and self._gt_to_pred[gid] != pid:
                n_idsw += 1
            self._gt_to_pred[gid] = pid

        # Fragmentation: GT was matched before, then missed, now matched again
        n_frag = 0
        gt_matched_now = {gt_ids[r] for r, _ in matches}
        for gid in gt_matched_now:
            if gid in self._gt_matched_prev and not self._gt_matched_prev[gid]:
                n_frag += 1
        for gid in gt_ids:
            self._gt_matched_prev[gid] = (gid in gt_matched_now)

        self.total_gt   += n_gt
        self.total_tp   += len(matches)
        self.total_fp   += n_fp
        self.total_fn   += n_fn
        self.total_idsw += n_idsw
        self.total_frag += n_frag

    def results(self):
        errors = self.total_fn + self.total_fp + self.total_idsw
        mota   = 1.0 - errors / self.total_gt if self.total_gt > 0 else 0.0
        recall = self.total_tp / self.total_gt if self.total_gt > 0 else 0.0
        prec_denom = self.total_tp + self.total_fp
        precision  = self.total_tp / prec_denom if prec_denom > 0 else 0.0
        return {
            "MOTA":      round(mota * 100, 2),
            "Recall":    round(recall * 100, 2),
            "Precision": round(precision * 100, 2),
            "IDSW":      self.total_idsw,
            "Frag":      self.total_frag,
            "FP":        self.total_fp,
            "FN":        self.total_fn,
            "GT":        self.total_gt,
        }


# Per-sequence runner 

def make_tracker(baseline):
    """Return a fresh tracker instance for the given baseline mode."""
    if baseline == "full":
        return Tracker()
    elif baseline == "no_kalman":
        return BaselineTracker(use_kalman=False, use_hungarian=True)
    elif baseline == "greedy":
        return BaselineTracker(use_kalman=True, use_hungarian=False)
    elif baseline == "detection_only":
        return DetectionOnlyTracker()
    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def evaluate_sequence(seq_name, baseline="full"):
    seq_dir  = os.path.join(SEQUENCES_DIR, seq_name)
    xml_path = os.path.join(ANNOTATIONS_DIR, f"{seq_name}.xml")

    if not os.path.isdir(seq_dir):
        print(f"  [SKIP] No images found: {seq_dir}")
        return None
    if not os.path.isfile(xml_path):
        print(f"  [SKIP] No annotation found: {xml_path}")
        return None

    gt_frames, ignored_boxes, metadata = parse_annotation(xml_path)

    detector = Detector()
    tracker  = make_tracker(baseline)
    acc      = MOTAAccumulator()

    print(f"  Running {seq_name} [{baseline}] ({len(gt_frames)} annotated frames, "
          f"weather={metadata['weather']})...")

    for frame_id, frame in load_frames(seq_dir):
        detections    = detector.detect_frame_bgr(frame)
        track_outputs = tracker.update(frame_id, detections)

        pred_entries = [(t["track_id"], t["bbox"]) for t in track_outputs]
        gt_entries   = gt_frames.get(frame_id, [])
        acc.update(gt_entries, pred_entries, ignored_boxes)

    results = acc.results()
    results["sequence"] = seq_name
    results["weather"]  = metadata["weather"]
    results["baseline"] = baseline
    return results


# Output helpers 

BASELINE_LABELS = {
    "full":           "Full pipeline (Kalman + Hungarian)",
    "no_kalman":      "No-motion model  (position-only)",
    "greedy":         "Greedy association (no Hungarian)",
    "detection_only": "Detection-only   (no tracker)",
}


def print_table(all_results, label=None):
    if label:
        print(f"\n{'─'*80}")
        print(f"  {label}")
    print()
    print(f"{'Sequence':<14} {'Weather':<8} {'MOTA':>6} {'Recall':>7} {'Prec':>6} "
          f"{'IDSW':>5} {'Frag':>5} {'FP':>7} {'FN':>7} {'GT':>7}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['sequence']:<14} {r['weather']:<8} {r['MOTA']:>6.1f} "
              f"{r['Recall']:>7.1f} {r['Precision']:>6.1f} "
              f"{r['IDSW']:>5} {r['Frag']:>5} {r['FP']:>7} {r['FN']:>7} {r['GT']:>7}")
    if len(all_results) > 1:
        avg_mota   = sum(r["MOTA"]   for r in all_results) / len(all_results)
        avg_recall = sum(r["Recall"] for r in all_results) / len(all_results)
        avg_prec   = sum(r["Precision"] for r in all_results) / len(all_results)
        tot_idsw   = sum(r["IDSW"]   for r in all_results)
        tot_frag   = sum(r["Frag"]   for r in all_results)
        print("-" * 80)
        print(f"{'AVERAGE':<14} {'':8} {avg_mota:>6.1f} "
              f"{avg_recall:>7.1f} {avg_prec:>6.1f} "
              f"{tot_idsw:>5} {tot_frag:>5}")
    print("=" * 80)


def print_comparison_table(results_by_baseline):
    """
    Print a side-by-side comparison of all baselines averaged across sequences.
    columns: Baseline | MOTA | Recall | Precision | IDSW | Frag | FP | FN
    """
    print()
    print("=" * 90)
    print("ABLATION STUDY — averaged across all evaluated sequences")
    print("=" * 90)
    print(f"{'Condition':<38} {'MOTA':>6} {'Recall':>7} {'Prec':>6} "
          f"{'IDSW':>6} {'Frag':>6} {'FP':>7} {'FN':>7}")
    print("-" * 90)

    order = ["full", "no_kalman", "greedy", "detection_only"]
    for bl in order:
        if bl not in results_by_baseline:
            continue
        rs = results_by_baseline[bl]
        n  = len(rs)
        avg_mota  = sum(r["MOTA"]      for r in rs) / n
        avg_rec   = sum(r["Recall"]    for r in rs) / n
        avg_prec  = sum(r["Precision"] for r in rs) / n
        tot_idsw  = sum(r["IDSW"]      for r in rs)
        tot_frag  = sum(r["Frag"]      for r in rs)
        tot_fp    = sum(r["FP"]        for r in rs)
        tot_fn    = sum(r["FN"]        for r in rs)
        label = BASELINE_LABELS.get(bl, bl)
        print(f"{label:<38} {avg_mota:>6.1f} {avg_rec:>7.1f} {avg_prec:>6.1f} "
              f"{tot_idsw:>6} {tot_frag:>6} {tot_fp:>7} {tot_fn:>7}")

    print("=" * 90)
    print()
    print("Targets from proposal:")
    print("  MOTA ≥ 50% (Good)   |  MOTA ≥ 65% (Stretch)")
    print("  IDSW ≤ 200/seq      |  IDSW ≤ 100/seq (Stretch)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",      default=None,   help="Single sequence, e.g. MVI_20011")
    parser.add_argument("--baseline", default="full",
                        choices=["full", "no_kalman", "greedy", "detection_only"],
                        help="Which tracker variant to run")
    parser.add_argument("--compare",  action="store_true",
                        help="Run all 4 baseline conditions and print a comparison table")
    args = parser.parse_args()

    # Resolve sequences
    if args.seq:
        sequences = [args.seq]
    else:
        sequences = sorted(
            d for d in os.listdir(SEQUENCES_DIR)
            if os.path.isdir(os.path.join(SEQUENCES_DIR, d))
            and os.path.isfile(os.path.join(ANNOTATIONS_DIR, f"{d}.xml"))
        )

    if not sequences:
        print("No sequences found. Check data/ua-detrac/ directories.")
        return

    if args.compare:
        baselines = ["full", "no_kalman", "greedy", "detection_only"]
    else:
        baselines = [args.baseline]

    results_by_baseline = {}

    for bl in baselines:
        print(f"\n{'='*60}")
        print(f"  Condition: {BASELINE_LABELS[bl]}")
        print(f"  Sequences: {sequences}")
        print(f"{'='*60}")

        bl_results = []
        for seq in sequences:
            result = evaluate_sequence(seq, baseline=bl)
            if result:
                bl_results.append(result)
                r = result
                print(f"  → MOTA={r['MOTA']}%  IDSW={r['IDSW']}  "
                      f"Frag={r['Frag']}  Recall={r['Recall']}%  "
                      f"Precision={r['Precision']}%\n")

        if bl_results:
            print_table(bl_results, label=BASELINE_LABELS[bl])
        results_by_baseline[bl] = bl_results

    if args.compare:
        print_comparison_table(results_by_baseline)


if __name__ == "__main__":
    main()

