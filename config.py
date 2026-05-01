# Detection 
CONF_THRESHOLD  = 0.5   # discard detections below this confidence
NMS_THRESHOLD   = 0.45  # IoU threshold for Non-Maximum Suppression
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# Tracking
IOU_MATCH_THRESHOLD = 0.3   # max (1 - IoU) cost to accept a track↔detection match
MAX_MISSED_FRAMES   = 5     # frames unmatched before a track is killed
MIN_INIT_HITS       = 2     # consecutive matches before a tentative track goes active

# Congestion
V_MIN           = 5.0      # px/frame — speeds below this contribute to congestion
# RHO_MAX: with CONF_THRESHOLD=0.5, a busy intersection yields ~5-12 confirmed
# tracks at once in 1280×720 (921,600 px²).  4 tracks / 921600 ≈ 4.3e-6.
RHO_MAX         = 0.000004 # vehicles per pixel² — triggers at ~4+ simultaneous tracks
SMOOTH_WINDOW   = 10       # frames for sliding-window temporal smoothing

# Scene-specific spatial config — 4-way intersection (1280×720)
#
#  Road geometry (approximate pixel coordinates):
#
#        [N arm: x 390-680, y 0-190]
#  [W]                                    [E]
#  x 0-160      INTERSECTION BOX        x 960-1280
#  y 230-500       x 160-960            y 140-430
# 
#        [S arm: x 300-640, y 510-720]

# Counting lines — one per approach arm
COUNTING_LINES = [
    {"name": "north", "line": [(390, 220), (710, 220)]},  # horizontal, mid N arm
    {"name": "south", "line": [(370, 560), (680, 560)]},  # horizontal, bottom of S arm
    {"name": "east",  "line": [(1050, 160), (1050, 470)]}, # vertical, right of E arm
    {"name": "west",  "line": [(70,  220), (70,  520)]},  # vertical, left of W arm
]

# Lane ROIs — expanded to cover each approach quadrant through the intersection.
# Each zone extends from the frame edge through the intersection centre so that
# vehicles detected mid-intersection still get a lane assignment.
#
#   Frame is 1280×720.  Intersection centre ≈ (640, 330).
#
#   North zone:  x=[390..710],  y=[0..330]   — vehicles coming from the north
#   South zone:  x=[300..680],  y=[330..720] — vehicles coming from the south
#   East zone:   x=[640..1280], y=[140..470] — vehicles coming from the east
#   West zone:   x=[0..640],    y=[140..470] — vehicles coming from the west
#
# The N/S zones and E/W zones overlap in the intersection box; assign_lanes
# picks the lane with the MOST trajectory points, so vehicles that transit
# through both zones are assigned by majority direction.
LANE_ROIS = [
    {"name": "north_arm", "polygon": [(390,   0), (710,   0), (710, 330), (390, 330)]},
    {"name": "south_arm", "polygon": [(300, 330), (680, 330), (680, 720), (300, 720)]},
    {"name": "east_arm",  "polygon": [(640, 140), (1280, 140), (1280, 470), (640, 470)]},
    {"name": "west_arm",  "polygon": [(0,   140), (640,  140), (640,  470), (0,  470)]},
]
