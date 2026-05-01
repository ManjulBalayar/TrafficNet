"""
Takes in the `frame_detections` dictionary from our detector.py and processes one frame at a time.

For each frame, it roughly does:

Frame 1
- no existing tracks
- creates a new `Track` object for each detection

Frame 2
- tell every existing track to predict
- collect predicted boxes from all tracks
- call association.py to match predictions with current detections
- update matched tracks
- mark unmatched tracks as missed
- create new tracks from an unmatched detection
- delete old dead tracks
- save output for that frame

This file is the orchestrator.
"""

from tracking.track import Track
from tracking.association import associate_tracks_and_detections
from analytics.trajectories import TrajectoryStore
from config import MIN_INIT_HITS


class Tracker:

    def __init__(self):
        self.tracks = []
        self.next_track_id = 1
        self.trajectory_store = TrajectoryStore()

    def update(self, frame_id, detections):
        # Step 1: predict all tracks
        for track in self.tracks:
            track.predict()

        # Step 2: associate tracks with detections
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(
            self.tracks, detections
        )

        # Step 3: update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Step 4: mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 5: create new tracks from unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(detections[det_idx], track_id=self.next_track_id)
            self.next_track_id += 1
            self.tracks.append(new_track)

        # Step 6: remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]

        # Step 6.5: record trajectory history for all surviving tracks
        self.trajectory_store.update(frame_id, self.tracks)

        # Step 7: collect outputs — only return tracks confirmed over MIN_INIT_HITS
        outputs = [
            t.get_output()
            for t in self.tracks
            if t.age >= MIN_INIT_HITS and t.missed_frames == 0
        ]

        return outputs
