"""
Scene config: Indian 4-way intersection (~/Downloads/traffic.mp4)
Frame: 1280×720

Run with:
    python main.py ~/Downloads/traffic.mp4 --scene traffic_india --show
"""

# Counting lines
COUNTING_LINES = [
    {"name": "north", "line": [(390, 290), (710, 290)]},   # horizontal, lower N arm
    {"name": "south", "line": [(370, 560), (680, 560)]},   # horizontal, S arm
    {"name": "east",  "line": [(1050, 160), (1050, 470)]}, # vertical, right of E arm
    {"name": "west",  "line": [(70,   220), (70,   520)]}, # vertical, left of W arm
]

# Lane ROIs 
#   Intersection centre ≈ (640, 330)
#   East/West extend full height to cover bottom-right / bottom-left approach roads.
LANE_ROIS = [
    {"name": "north_arm", "polygon": [(390,   0), (710,   0), (710, 330), (390, 330)]},
    {"name": "south_arm", "polygon": [(300, 330), (680, 330), (680, 720), (300, 720)]},
    {"name": "east_arm",  "polygon": [(640, 140), (1280, 140), (1280, 720), (640, 720)]},
    {"name": "west_arm",  "polygon": [(0,   140), (640,  140), (640,  720), (0,  720)]},
]

# Congestion overrides (optional — falls back to config.py defaults if absent) 
V_MIN   = 5.0       # px/frame
RHO_MAX = 0.000004  # about 4 vehicles in 1280×720
