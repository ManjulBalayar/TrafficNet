"""
Congestion detection from average speed and spatial density.

A congestion event is flagged when:
  - average speed of active tracks < V_MIN  (px/frame)
  - spatial density of active tracks > RHO_MAX (vehicles / px²)

A sliding window smoother prevents single-frame false alarms.
"""

import math
from collections import deque
from config import V_MIN, RHO_MAX, SMOOTH_WINDOW


def _avg_speed(tracks):
    """Mean pixel-space speed across all active tracks."""
    if not tracks:
        return 0.0
    speeds = []
    for track in tracks:
        state = track.kalman_filter.get_state()
        vx, vy = state[4], state[5]
        speeds.append(math.sqrt(vx**2 + vy**2))
    return sum(speeds) / len(speeds)


def _density(tracks, frame_area):
    """Vehicles per pixel² (scaled for readability)."""
    if frame_area <= 0:
        return 0.0
    return len(tracks) / frame_area


class CongestionDetector:
    """
    Stateful detector: call update() once per frame and read is_congested.
    """

    def __init__(self, frame_width, frame_height):
        self._frame_area = frame_width * frame_height
        self._window = deque(maxlen=SMOOTH_WINDOW)  # stores per-frame boolean flags
        self.is_congested = False
        self.last_avg_speed = 0.0
        self.last_density = 0.0

    def update(self, active_tracks):
        """
        Parameters
        ----------
        active_tracks : list of Track objects currently alive in the tracker

        Returns
        -------
        bool: congestion status for this frame (after smoothing)
        """
        self.last_avg_speed = _avg_speed(active_tracks)
        self.last_density   = _density(active_tracks, self._frame_area)

        frame_congested = (
            self.last_avg_speed < V_MIN and
            self.last_density   > RHO_MAX
        )
        self._window.append(frame_congested)

        # flag congestion only if majority of the smoothing window is congested
        self.is_congested = (sum(self._window) / len(self._window)) > 0.5
        return self.is_congested

    def status(self):
        return {
            "congested":  self.is_congested,
            "avg_speed":  round(self.last_avg_speed, 3),
            "density":    round(self.last_density, 6),
        }
