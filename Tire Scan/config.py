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
S_MAX = 90                # max saturation counted as "tire black"
MORPH_SIZE = 5            # kernel size for cleaning the mask
MIN_AREA = 250            # px, smallest blob considered (360x240 image)
MAX_AREA_FRAC = 0.6       # ignore blobs bigger than this fraction of the image (e.g. dark walls)
MIN_ASPECT = 0.35         # ellipse minor/major axis ratio (lower = tire seen more edge-on)
MIN_CIRCULARITY = 0.35    # 4*pi*area/perimeter^2 of the outer contour (1 = perfect circle)
MIN_SOLIDITY = 0.75       # outer area / convex hull area
REQUIRE_HOLE = False      # True: blob must have an inner hole (rim/hub) to count as a tire
HOLE_MIN_FRAC = 0.03      # smallest hole area as a fraction of the outer area
HOLE_BONUS = 1.5          # score multiplier for blobs with a hole
MIN_SCORE = 0.35          # detections scoring below this are ignored

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
