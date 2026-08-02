"""
Scene config: traffic2 — calibrated interactively.

Run with:
    python main.py <video> --scene traffic2 --show
"""

COUNTING_LINES = [
    {"name": "north", "line": [(540, 363), (1041, 407)]},
    {"name": "south", "line": [(121, 900), (685, 1055)]},
    {"name": "east", "line": [(1128, 467), (885, 1035)]},
    {"name": "west", "line": [(460, 390), (77, 827)]},
]

LANE_ROIS = [
    {"name": "north_arm", "polygon": [(837, 9), (1233, 10), (1020, 330), (555, 330)]},
    {"name": "east_arm", "polygon": [(1154, 513), (1903, 620), (1862, 1062), (940, 1051)]},
    {"name": "west_arm", "polygon": [(10, 347), (425, 399), (53, 800), (0, 690)]},
]

V_MIN   = 5.0
RHO_MAX = 0.000004
