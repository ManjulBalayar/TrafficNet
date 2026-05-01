"""
Lane utilization via region-of-interest (ROI) polygon assignment.

Each lane is a convex polygon. A track is assigned to the lane whose ROI
contains the majority of its trajectory positions.
"""


def _point_in_polygon(point, polygon):
    """
    Ray-casting algorithm: returns True if point is inside the polygon.
    polygon is a list of (x, y) vertices.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def assign_lanes(trajectory_store, lane_rois):
    """
    Assign each track to a lane based on majority position vote.

    Parameters
    ----------
    trajectory_store : TrajectoryStore
    lane_rois        : list of {"name": str, "polygon": [(x,y), ...]}

    Returns
    -------
    dict: {track_id: lane_name or None}
    """
    assignments = {}

    for track_id, history in trajectory_store.all_histories().items():
        positions = [(cx, cy) for _, cx, cy, _, _ in history]

        # count how many positions fall in each lane
        lane_votes = {roi["name"]: 0 for roi in lane_rois}
        for pos in positions:
            for roi in lane_rois:
                if _point_in_polygon(pos, roi["polygon"]):
                    lane_votes[roi["name"]] += 1

        best_lane = max(lane_votes, key=lane_votes.get)
        assignments[track_id] = best_lane if lane_votes[best_lane] > 0 else None

    return assignments


def lane_utilization(lane_assignments):
    """
    Compute fraction of tracked vehicles in each lane.

    Parameters
    ----------
    lane_assignments : output of assign_lanes()

    Returns
    -------
    dict: {lane_name: fraction (0.0–1.0)}
    """
    total = len([v for v in lane_assignments.values() if v is not None])
    if total == 0:
        return {}

    counts = {}
    for lane in lane_assignments.values():
        if lane is not None:
            counts[lane] = counts.get(lane, 0) + 1

    return {lane: round(count / total, 3) for lane, count in counts.items()}
