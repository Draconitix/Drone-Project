"""Settings for the tire scan (tune these)."""
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- flight
HOVER_TIME = 3.0          # seconds to hover and settle after take-off before scanning
YAW_RATE = 0.06           # yaw command while scanning (+ = turn right). ~15.8 deg/s
YAW_DEG_PER_UNIT = 263.0  # measured from run 20261106_091113 (yaw 0.3 -> one turn every ~4.56 s)
SCAN_DEGREES = 360.0      # how far to rotate before landing
CAMERA_ANGLE = 0          # camera tilt sent with every command (0 = forward)
LOOP_HZ = 30

# heading is measured from the camera (horizontal image shift between frames),
# so the scan stops after one real turn even if YAW_DEG_PER_UNIT is off
CAMERA_HFOV = 69.0        # effective horizontal FOV (deg), calibrated so 1 turn reads 360 on run 20261106_091113
VO_MAX_STEP_DEG = 20.0    # ignore per-frame shifts larger than this (bad match)
VO_MIN_RESPONSE = 0.05    # ignore phase-correlation matches weaker than this
LOOP_CLOSE_MIN_DEG = 300.0  # start checking for "back at the start view" after this many degrees
LOOP_CLOSE_SIM = 0.90     # similarity to the start view that counts as a full turn
# safety: land if the scan takes this much longer than one turn should
SCAN_TIMEOUT = 1.5 * SCAN_DEGREES / (abs(YAW_RATE) * YAW_DEG_PER_UNIT)

# ---------------------------------------------------------------- detector
# tires are dark, low-saturation rubber
V_MAX = 60                # max brightness (HSV V) counted as "tire black"
CHROMA_MAX = 25           # max colourfulness (max-min of B,G,R) counted as "tire black".
                          # (HSV saturation is unreliable on near-black pixels: it rejected ~40% of tire pixels)
MORPH_SIZE = 7            # kernel size for cleaning the mask
# Values below were tightened using run 20261106_094754:
#   real tires:      area 3500-4500, aspect 0.88-0.99, circ 0.57-0.70, solidity 0.91-0.94, hole ~0.5
#   false positives: area 226-551,   aspect 0.31-0.74, no hole (desk shadow, stool legs, ladder, bottle)
MIN_AREA = 800            # px, smallest blob considered (360x240 image)
MAX_AREA_FRAC = 0.6       # ignore blobs bigger than this fraction of the image (e.g. dark walls)
MIN_ASPECT = 0.6          # ellipse minor/major axis ratio (lower = tire seen more edge-on)
MIN_CIRCULARITY = 0.5     # 4*pi*area/perimeter^2 of the outer contour (1 = perfect circle)
MIN_SOLIDITY = 0.85       # outer area / convex hull area
REQUIRE_HOLE = True       # True: blob must have an inner hole (rim/hub) to count as a tire
HOLE_MIN_FRAC = 0.15      # smallest hole area as a fraction of the outer area
HOLE_MAX_OFFSET = 0.2     # max distance hole centre <-> blob centre, as a fraction of blob diameter
HOLE_BONUS = 1.5          # score multiplier for blobs with a hole
MIN_SCORE = 0.6           # detections scoring below this are ignored

# hole-first detection, for tires that touch other dark objects (shelf, ladder) and merge with them.
# Measured on runs 20261106_094754 / _101503: every tire hole has circ 0.80-0.84, aspect >= 0.81,
# and the ellipse 1.3x the hole size is 99-100 % dark; no non-tire hole >= 100 px exists.
RING_DETECT = True
RING_HOLE_MIN_AREA = 300  # px, smallest hole considered (tire holes are ~1000-1500 px)
RING_HOLE_MIN_CIRC = 0.7  # hole roundness
RING_HOLE_MIN_ASPECT = 0.65
RING_SCALE = 1.3          # where to test for the dark ring, relative to the hole size
RING_MIN_DARK = 0.95      # fraction of that ring that must be "tire black"
TIRE_TO_HOLE = 1.6        # tire outline size relative to the hole (for the bounding box)

# ---------------------------------------------------------------- capture logic
CONFIRM_FRAMES = 3        # consecutive frames with a tire before taking a screenshot
LOST_FRAMES = 5           # frames without a tire before the view counts as "empty" again
REARM_DEGREES = 30.0      # min rotation between screenshots (avoids photographing the same tire twice)
REARM_FORCE_DEGREES = 60.0  # after this much rotation re-arm even if a tire is still in view (next tire)

# ---------------------------------------------------------------- output
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
RECORD_DIR = os.path.join(BASE_DIR, "runs")
RECORD_FRAMES = True      # save every raw frame of the flight for offline tuning
WINDOW_NAME = "Tire scan"

KERNEL = np.ones((MORPH_SIZE, MORPH_SIZE), np.uint8)
