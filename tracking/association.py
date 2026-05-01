import numpy as np
from scipy.optimize import linear_sum_assignment
from config import IOU_MATCH_THRESHOLD


def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def build_cost_matrix(tracks, detections):
    """
    Build an (N_tracks x N_detections) cost matrix where cost = 1 - IoU.
    """
    n_tracks = len(tracks)
    n_dets   = len(detections)
    cost = np.ones((n_tracks, n_dets), dtype=float)

    for i, track in enumerate(tracks):
        pred_bbox = track.get_bbox()
        for j, det in enumerate(detections):
            cost[i, j] = 1.0 - compute_iou(pred_bbox, det["bbox"])

    return cost


def associate_tracks_and_detections(tracks, detections):
    """
    Match predicted tracks to current detections using the Hungarian algorithm.

    Returns
    -------
    matches            : list of (track_idx, detection_idx) pairs
    unmatched_tracks   : list of track indices with no match
    unmatched_dets     : list of detection indices with no match
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = build_cost_matrix(tracks, detections)

    row_inds, col_inds = linear_sum_assignment(cost)

    matches, unmatched_tracks, unmatched_dets = [], [], []

    matched_track_inds = set()
    matched_det_inds   = set()

    for r, c in zip(row_inds, col_inds):
        if cost[r, c] <= (1.0 - IOU_MATCH_THRESHOLD):
            matches.append((r, c))
            matched_track_inds.add(r)
            matched_det_inds.add(c)

    unmatched_tracks = [i for i in range(len(tracks))     if i not in matched_track_inds]
    unmatched_dets   = [j for j in range(len(detections)) if j not in matched_det_inds]

    return matches, unmatched_tracks, unmatched_dets
