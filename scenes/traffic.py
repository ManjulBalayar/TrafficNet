"""
Scene config: traffic — calibrated interactively.

Run with:
    python main.py <video> --scene traffic --show
"""

COUNTING_LINES = [
    {"name": "north", "line": [(399, 196), (765, 250)]},
    {"name": "south", "line": [(210, 482), (653, 582)]},
    {"name": "east", "line": [(864, 316), (754, 571)]},
    {"name": "west", "line": [(369, 195), (155, 404)]},
]

LANE_ROIS = [
    {"name": "north_arm", "polygon": [(621, 6), (835, 12), (788, 226), (402, 165)]},
    {"name": "south_arm", "polygon": [(188, 506), (655, 628), (628, 719), (29, 716)]},
    {"name": "east_arm", "polygon": [(898, 315), (1269, 369), (1280, 720), (773, 592)]},
    {"name": "west_arm", "polygon": [(0, 140), (375, 157), (120, 395), (9, 348)]},
]

V_MIN   = 5.0
RHO_MAX = 0.000004
