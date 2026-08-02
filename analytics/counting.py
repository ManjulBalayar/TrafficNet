"""
Flow rate estimation via virtual counting lines.

A counting line is defined by two endpoints [(x1,y1), (x2,y2)].

Two implementations are provided:

1. ``CountingLineMonitor`` (preferred) — stateful, real-time detector.
   Call ``monitor.update(tracks)`` every frame.  Crossings are caught the
   instant they happen (O(T × L) per frame, constant memory).  Also checks
   the bounding-box leading edge so large vehicles are counted before their
   centre crosses the line.

2. ``count_crossings`` (legacy) — replays the full trajectory store each call
   (O(total_positions × L)).  Still used for the final summary.
"""


def _side(point, line_p1, line_p2):
    """Signed cross-product tells which side of the line the point is on.

    Returns +1, -1, or 0 (exactly on the line).
    The line is treated as infinite in both directions.
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    px = point[0] - line_p1[0]
    py = point[1] - line_p1[1]
    cross = dx * py - dy * px
    if cross > 0:
        return 1
    elif cross < 0:
        return -1
    return 0


def _leading_point(bbox, line_p1, line_p2):
    """
    Return the bounding-box corner/edge midpoint that is closest to the line,
    so large vehicles are counted as soon as their front reaches the line.

    For a roughly horizontal line (|dy| < |dx|) we use the top or bottom edge
    midpoint; for a roughly vertical line we use the left or right edge.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(line_p2[0] - line_p1[0])
    dy = abs(line_p2[1] - line_p1[1])
    if dx >= dy:
        # Horizontal-ish line — check which vertical edge (top or bottom) of
        # the box is closer to the line, and use that edge's midpoint.
        side_top    = _side((cx, y1), line_p1, line_p2)
        side_bottom = _side((cx, y2), line_p1, line_p2)
        # The "leading" edge is the one on the opposite side from the centre,
        # i.e. the edge that is crossing first.
        side_centre = _side((cx, cy), line_p1, line_p2)
        if side_centre != 0 and side_top != side_centre:
            return (cx, y1)
        if side_centre != 0 and side_bottom != side_centre:
            return (cx, y2)
    else:
        # Vertical-ish line — use left or right edge midpoint.
        side_left  = _side((x1, cy), line_p1, line_p2)
        side_right = _side((x2, cy), line_p1, line_p2)
        side_centre = _side((cx, cy), line_p1, line_p2)
        if side_centre != 0 and side_left != side_centre:
            return (x1, cy)
        if side_centre != 0 and side_right != side_centre:
            return (x2, cy)
    return (cx, cy)


class CountingLineMonitor:
    """
    Stateful, real-time crossing detector.

    Maintains the last-seen side of the counting line for every active track.
    A crossing is recorded the frame the side flips.  Each track is counted
    at most once per line even if it crosses back and forth.

    Improvements over the legacy ``count_crossings`` function:
      - O(T × L) per frame instead of O(total_trajectory_length × L).
      - Checks the bounding-box leading edge, so large vehicles are counted
        before their centre crosses the line.
      - Counts are incremented live, so the on-screen number updates the
        moment a crossing happens.
    """

    def __init__(self, counting_lines):
        self.counting_lines = counting_lines
        self._track_side   = {}          # track_id -> {line_name: last_side}
        self._counted_ids  = {cl["name"]: set() for cl in counting_lines}
        self.counts        = {cl["name"]: 0     for cl in counting_lines}

    def update(self, tracks):
        """
        Call once per frame with the list of all active Track objects.

        Parameters
        ----------
        tracks : list of Track  (must expose .track_id and .get_bbox())
        """
        active_ids = set()
        for track in tracks:
            tid  = track.track_id
            active_ids.add(tid)
            bbox = track.get_bbox()           # [x1, y1, x2, y2]

            if tid not in self._track_side:
                self._track_side[tid] = {}

            for cl in self.counting_lines:
                name   = cl["name"]
                p1, p2 = cl["line"]

                # Use the leading bbox edge so large vehicles count earlier
                pt         = _leading_point(bbox, p1, p2)
                curr_side  = _side(pt, p1, p2)

                if curr_side == 0:
                    continue    # exactly on the line — skip to avoid false trigger

                prev_side = self._track_side[tid].get(name)
                if prev_side is not None and prev_side != curr_side:
                    # Side flipped → vehicle just crossed
                    if tid not in self._counted_ids[name]:
                        self.counts[name]          += 1
                        self._counted_ids[name].add(tid)

                self._track_side[tid][name] = curr_side

        # Purge state for tracks that are no longer alive
        for tid in list(self._track_side.keys()):
            if tid not in active_ids:
                del self._track_side[tid]

    def get_crossing_results(self):
        """Return results in the same dict format as ``count_crossings``."""
        return {
            name: {"count": self.counts[name], "track_ids": self._counted_ids[name]}
            for name in self.counts
        }


def count_crossings(trajectory_store, counting_lines):
    """
    Legacy trajectory-replay counter.  Used for the final summary only.

    Parameters
    ----------
    trajectory_store : TrajectoryStore
    counting_lines   : list of {"name": str, "line": [(x1,y1), (x2,y2)]}

    Returns
    -------
    dict: {line_name: {"count": int, "track_ids": set}}
    """
    results = {cl["name"]: {"count": 0, "track_ids": set()} for cl in counting_lines}

    for track_id, history in trajectory_store.all_histories().items():
        positions = [(cx, cy) for _, cx, cy, _, _ in history]
        if len(positions) < 2:
            continue

        for cl in counting_lines:
            p1, p2 = cl["line"]
            name = cl["name"]

            prev_side = _side(positions[0], p1, p2)
            for pos in positions[1:]:
                curr_side = _side(pos, p1, p2)
                if prev_side != 0 and curr_side != 0 and prev_side != curr_side:
                    if track_id not in results[name]["track_ids"]:
                        results[name]["count"] += 1
                        results[name]["track_ids"].add(track_id)
                    break
                prev_side = curr_side

    return results


def flow_rate(crossing_results, num_frames, fps=25):
    """
    Convert crossing counts to vehicles-per-minute.

    Parameters
    ----------
    crossing_results : output of count_crossings()
    num_frames       : total frames in the sequence
    fps              : frames per second of the video

    Returns
    -------
    dict: {line_name: flow_veh_per_min}
    """
    duration_min = (num_frames / fps) / 60.0
    return {
        name: round(data["count"] / duration_min, 2) if duration_min > 0 else 0
        for name, data in crossing_results.items()
    }
