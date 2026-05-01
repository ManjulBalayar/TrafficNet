class TrajectoryStore:
    """
    Records the Kalman-corrected state history for every active track.

    Stored per track:
        [(frame_id, cx, cy, vx, vy), ...]

    The analytics modules (counting, lanes, congestion) read from here.
    """

    def __init__(self):
        # track_id → list of (frame_id, cx, cy, vx, vy)
        self._histories = {}

    def update(self, frame_id, tracks):
        """
        Call once per frame, after the tracker has updated all tracks.
        `tracks` is the list of active Track objects (not output dicts).

        Works with both Kalman-backed tracks (Track) and position-only
        tracks (PositionOnlyTrack) by falling back to bbox centre when
        no Kalman filter is present.
        """
        for track in tracks:
            if hasattr(track, "kalman_filter"):
                state = track.kalman_filter.get_state()  # [cx, cy, w, h, vx, vy]
                cx, cy, vx, vy = state[0], state[1], state[4], state[5]
            else:
                # Position-only baseline: derive centre from stored bbox
                b = track.get_bbox()
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0
                vx, vy = 0.0, 0.0

            if track.track_id not in self._histories:
                self._histories[track.track_id] = []

            self._histories[track.track_id].append((frame_id, cx, cy, vx, vy))

    def get(self, track_id):
        """Return the full history for a single track, or [] if unknown."""
        return self._histories.get(track_id, [])

    def all_histories(self):
        """Return the full dict: {track_id: [(frame_id, cx, cy, vx, vy), ...]}."""
        return self._histories

    def get_positions(self, track_id):
        """Return just the (cx, cy) positions for a track."""
        return [(cx, cy) for _, cx, cy, _, _ in self.get(track_id)]

    def get_velocities(self, track_id):
        """Return just the (vx, vy) pairs for a track."""
        return [(vx, vy) for _, _, _, vx, vy in self.get(track_id)]
