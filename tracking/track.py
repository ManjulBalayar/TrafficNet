"""
Creates a track object that stores the following:
- track_id
- class info
- current bbox
- missed frames / age
- Kalman filter instance
"""

from tracking.kalman_filter import KalmanFilter
from config import MAX_MISSED_FRAMES


def _xyxy_to_cxcywh(bbox):
    """Convert [x1, y1, x2, y2] → [cx, cy, w, h]."""
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def _cxcywh_to_xyxy(state):
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = state
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class Track:

    def __init__(self, detection, track_id):
        self.track_id = track_id

        self.bbox = detection["bbox"]  # stored as [x1, y1, x2, y2]
        self.conf = detection["conf"]
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]

        self.age = 1
        self.missed_frames = 0

        initial_measurement = _xyxy_to_cxcywh(self.bbox)
        self.kalman_filter = KalmanFilter(initial_measurement)

    def predict(self):
        """Advance the Kalman filter one step and return the predicted bbox [x1,y1,x2,y2]."""
        predicted_cxcywh = self.kalman_filter.predict()
        self.bbox = _cxcywh_to_xyxy(predicted_cxcywh)
        self.age += 1
        return self.bbox

    def update(self, detection):
        """Correct the filter with a matched detection."""
        measurement = _xyxy_to_cxcywh(detection["bbox"])
        updated_cxcywh = self.kalman_filter.update(measurement)
        self.bbox = _cxcywh_to_xyxy(updated_cxcywh)

        self.missed_frames = 0
        self.conf = detection["conf"]
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]

    def mark_missed(self):
        """Called when no detection is matched this frame."""
        self.missed_frames += 1

    def is_dead(self):
        """Return True once the track has been unmatched too many times."""
        return self.missed_frames >= MAX_MISSED_FRAMES

    def get_bbox(self):
        """Return current bbox [x1, y1, x2, y2] from Kalman state."""
        state = self.kalman_filter.get_state()
        return _cxcywh_to_xyxy(state[:4])

    def get_output(self):
        """Return a minimal dict for downstream consumers."""
        return {
            "track_id": self.track_id,
            "bbox": self.get_bbox(),
            "class_name": self.class_name,
            "class_id": self.class_id,
            "conf": self.conf,
        }
