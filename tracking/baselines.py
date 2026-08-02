"""
Baseline trackers for ablation study.

Baseline 1 — No Motion Model (no_kalman)
    Replaces the Kalman filter with a position-only tracker.
    Each track simply stores its last known bbox with no velocity
    and no prediction step. The "predicted" position is just the
    previous frame's bbox.

Baseline 2 — Greedy Association (greedy)
    Replaces the Hungarian algorithm with greedy nearest-neighbor.
    For each track (sorted by track_id), pick the closest unmatched
    detection by IoU. No global optimum — first-come, first-served.

Baseline 3 — Detection-Only Counting (detection_only)
    No tracking at all. Counts how many unique bounding boxes appear
    across all frames using IoU-based deduplication per frame.
    Returns a simple per-frame detection list (no IDs, no history).
"""

from config import IOU_MATCH_THRESHOLD, MAX_MISSED_FRAMES, MIN_INIT_HITS


# Baseline 1: Position-only track (no Kalman) 

class PositionOnlyTrack:
    """Stores last bbox only — no velocity, no prediction."""

    def __init__(self, detection, track_id):
        self.track_id     = track_id
        self.bbox         = list(detection["bbox"])
        self.conf         = detection["conf"]
        self.class_id     = detection["class_id"]
        self.class_name   = detection["class_name"]
        self.age          = 1
        self.missed_frames = 0

    def predict(self):
        """Return last known bbox unchanged (no motion model)."""
        self.age += 1
        return self.bbox

    def update(self, detection):
        self.bbox          = list(detection["bbox"])
        self.conf          = detection["conf"]
        self.class_id      = detection["class_id"]
        self.class_name    = detection["class_name"]
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1

    def is_dead(self):
        return self.missed_frames >= MAX_MISSED_FRAMES

    def get_bbox(self):
        return self.bbox

    def get_output(self):
        return {
            "track_id":   self.track_id,
            "bbox":       self.bbox,
            "class_name": self.class_name,
            "class_id":   self.class_id,
            "conf":       self.conf,
        }


# Baseline 2: Greedy association 

def greedy_associate(tracks, detections):
    """
    Greedy nearest-neighbour association by IoU.
    Iterates tracks in order; each track grabs the best available detection.
    Returns (matches, unmatched_tracks, unmatched_dets) — same interface
    as associate_tracks_and_detections().
    """
    from tracking.association import compute_iou

    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    available_dets = set(range(len(detections)))
    matches        = []

    for ti, track in enumerate(tracks):
        pred_bbox = track.get_bbox()
        best_iou  = IOU_MATCH_THRESHOLD
        best_di   = None

        for di in available_dets:
            iou = compute_iou(pred_bbox, detections[di]["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_di  = di

        if best_di is not None:
            matches.append((ti, best_di))
            available_dets.remove(best_di)

    matched_tracks = {ti for ti, _ in matches}
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets   = list(available_dets)

    return matches, unmatched_tracks, unmatched_dets


# Baseline 1+2 combined tracker 

class BaselineTracker:
    """
    Drop-in replacement for Tracker that supports:
      use_kalman=False  → position-only tracks
      use_hungarian=False → greedy association
    """

    def __init__(self, use_kalman=True, use_hungarian=True):
        from tracking.tracker import Tracker
        from analytics.trajectories import TrajectoryStore

        self.use_kalman    = use_kalman
        self.use_hungarian = use_hungarian
        self.tracks        = []
        self.next_track_id = 1
        self.trajectory_store = TrajectoryStore()

    def update(self, frame_id, detections):
        from tracking.track import Track
        from tracking.association import associate_tracks_and_detections

        # Step 1: predict
        for track in self.tracks:
            track.predict()

        # Step 2: associate
        if self.use_hungarian:
            matches, unmatched_tracks, unmatched_dets = \
                associate_tracks_and_detections(self.tracks, detections)
        else:
            matches, unmatched_tracks, unmatched_dets = \
                greedy_associate(self.tracks, detections)

        # Step 3-5: update / mark missed / create new
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()
        for di in unmatched_dets:
            TrackClass = Track if self.use_kalman else PositionOnlyTrack
            self.tracks.append(TrackClass(detections[di], self.next_track_id))
            self.next_track_id += 1

        # Step 6: remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]

        # Step 6.5: record trajectories
        self.trajectory_store.update(frame_id, self.tracks)

        # Step 7: output confirmed tracks
        return [
            t.get_output()
            for t in self.tracks
            if t.age >= MIN_INIT_HITS and t.missed_frames == 0
        ]


# Baseline 3: Detection-only (no tracker) 

class DetectionOnlyTracker:
    """
    No tracking. Each detection in each frame is treated as an independent
    observation. Returns detections directly as track-like dicts with
    synthetic IDs so the evaluation pipeline stays unchanged.
    """

    def __init__(self):
        from analytics.trajectories import TrajectoryStore
        self.tracks           = []   # always empty, no persistent state
        self.trajectory_store = TrajectoryStore()
        self._next_id         = 1

    def update(self, frame_id, detections):
        outputs = []
        for det in detections:
            outputs.append({
                "track_id":   self._next_id,
                "bbox":       det["bbox"],
                "class_name": det["class_name"],
                "class_id":   det["class_id"],
                "conf":       det["conf"],
            })
            self._next_id += 1
        return outputs
