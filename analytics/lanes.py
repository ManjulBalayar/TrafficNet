"""
Lane utilization via region-of-interest (ROI) polygon assignment.

Each lane is a convex polygon.  Utilization is measured by *zone presence*:
a vehicle is counted toward a zone if its trajectory passes through that zone
at least once.  This is more meaningful than majority-vote assignment for
intersection arms, where vehicles transit quickly through one arm and then
slow/stop in another — majority vote would unfairly exclude the arm they
entered from.
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
    For each track, record every zone it ever touched (presence-based).

    Returns
    -------
    dict: {track_id: set of lane_names the track was ever seen in}
    """
    assignments = {}

    for track_id, history in trajectory_store.all_histories().items():
        positions = [(cx, cy) for _, cx, cy, _, _ in history]
        touched = set()
        for pos in positions:
            for roi in lane_rois:
                if _point_in_polygon(pos, roi["polygon"]):
                    touched.add(roi["name"])
        assignments[track_id] = touched

    return assignments


def lane_utilization(occupancy_history):
    """
    Compute average instantaneous occupancy fraction for each zone.

    This is the most intuitive utilization metric: on average, what fraction
    of the total vehicles visible at any given moment were in each zone?

    If east_arm consistently holds 8-10 cars and north_arm holds 2-3, this
    metric reflects that directly — unlike presence or majority-vote approaches
    which are distorted by identity switches and track lifetimes.

    Parameters
    ----------
    occupancy_history : list of dicts — one per frame, each is the output of
                        current_zone_occupancy() for that frame.

    Returns
    -------
    dict: {zone_name: average_fraction (0.0–1.0)}
           Fractions are relative to total vehicles in frame, so they can sum
           to > 1.0 when zones overlap.
    """
    if not occupancy_history:
        return {}

    zone_names = set()
    for snap in occupancy_history:
        zone_names.update(snap.keys())

    zone_totals   = {z: 0.0 for z in zone_names}
    frame_totals  = []

    for snap in occupancy_history:
        total_in_frame = sum(snap.values())
        frame_totals.append(total_in_frame)
        for z in zone_names:
            zone_totals[z] += snap.get(z, 0)

    total_vehicle_frames = sum(frame_totals)
    if total_vehicle_frames == 0:
        return {z: 0.0 for z in zone_names}

    return {z: round(zone_totals[z] / total_vehicle_frames, 3)
            for z in zone_names}


def current_zone_occupancy(tracks, lane_rois):
    """
    Count how many active tracks are currently inside each zone.

    This is a real-time snapshot metric (per-frame), complementing the
    cumulative ``lane_utilization`` which covers the full video history.

    Parameters
    ----------
    tracks    : list of Track objects (must expose .get_bbox())
    lane_rois : list of {"name": str, "polygon": [(x,y), ...]}

    Returns
    -------
    dict: {zone_name: int}  — number of tracks currently in each zone
    """
    occupancy = {roi["name"]: 0 for roi in lane_rois}
    for track in tracks:
        x1, y1, x2, y2 = track.get_bbox()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        for roi in lane_rois:
            if _point_in_polygon((cx, cy), roi["polygon"]):
                occupancy[roi["name"]] += 1
    return occupancy
