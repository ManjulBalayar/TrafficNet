"""
Flow rate estimation via virtual counting lines.

A counting line is defined by two endpoints [(x1,y1), (x2,y2)].
A vehicle is counted when consecutive trajectory positions straddle the line
(i.e. the sign of the cross-product flips side-to-side).
"""


def _side(point, line_p1, line_p2):
    """Return the sign of the cross product (line_p2-line_p1) × (point-line_p1)."""
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


def count_crossings(trajectory_store, counting_lines):
    """
    Count how many unique tracks crossed each counting line.

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
                    # crossed the line — count once per track per line
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
