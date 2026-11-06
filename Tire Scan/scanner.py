"""Scan state machine: HOVER -> SCAN (rotate 360 deg, watch for tires) -> DONE."""
import cv2
import numpy as np

import config
from detector import detect_tires

HOVER, SCAN, DONE = "HOVER", "SCAN", "DONE"

_SMALL = (90, 60)  # size used for the "back at the start view" check


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _signature(gray):
    """Normalised low-res image, for comparing whole views."""
    s = cv2.resize(gray, _SMALL)
    s = s - s.mean()
    n = np.linalg.norm(s)
    return s / n if n > 0 else s


class HeadingTracker:
    """Measures how far the drone has turned from the camera image.

    The horizontal shift between consecutive frames (phase correlation) is
    converted to degrees with the camera's field of view. This does not depend
    on the yaw-command scale, which differs between levels.
    """

    def __init__(self):
        self.prev = None
        self.window = None
        self.start_sig = None
        self.angle = 0.0

    def reset(self, frame):
        self.angle = 0.0
        self.prev = None
        self.start_sig = None
        if frame is not None:
            self.update(frame)

    def update(self, frame):
        """Add one frame. Returns the measured rotation since reset (deg)."""
        g = _gray(frame)
        if self.window is None or self.window.shape != g.shape:
            self.window = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
            self.prev = None
        if self.start_sig is None:
            self.start_sig = _signature(g)
        if self.prev is not None:
            (dx, _dy), response = cv2.phaseCorrelate(self.prev, g, self.window)
            # turning right moves the scene left in the image (negative dx)
            step = -dx * config.CAMERA_HFOV / g.shape[1]
            if response >= config.VO_MIN_RESPONSE and abs(step) <= config.VO_MAX_STEP_DEG:
                self.angle += step
        self.prev = g
        return self.angle

    def similarity_to_start(self, frame):
        if self.start_sig is None:
            return 0.0
        return float((_signature(_gray(frame)) * self.start_sig).sum())


class TireScanner:
    def __init__(self, start_time, params=None):
        self.params = params
        self.state = HOVER
        self.state_start = start_time
        self.last_time = start_time
        self.yaw = 0.0            # yaw command currently being flown
        self.angle = 0.0          # measured rotation since the scan started (deg)
        self.dr_angle = 0.0       # dead-reckoned rotation from the yaw commands (backup)
        self.heading = HeadingTracker()
        self.seen_far_view = False  # the view has left the start view at least once

        self.seen_count = 0       # consecutive frames with a tire
        self.lost_count = 0       # consecutive frames without a tire
        self.armed = True
        self.last_capture_angle = -1e9
        self.captures = []        # headings (deg) of saved screenshots
        self.last_mask = None     # detector mask of the latest frame (for display)

    def _set_state(self, state, now):
        self.state = state
        self.state_start = now

    def _update_capture(self, detections):
        """Return True when a screenshot should be taken for this frame."""
        if detections:
            self.seen_count += 1
            self.lost_count = 0
        else:
            self.seen_count = 0
            self.lost_count += 1

        turned = self.angle - self.last_capture_angle
        if not self.armed:
            view_cleared = self.lost_count >= config.LOST_FRAMES and turned >= config.REARM_DEGREES
            if view_cleared or turned >= config.REARM_FORCE_DEGREES:
                self.armed = True
                self.seen_count = min(self.seen_count, 1)  # need fresh confirmation

        if self.armed and self.seen_count >= config.CONFIRM_FRAMES:
            self.armed = False
            self.last_capture_angle = self.angle
            self.captures.append(self.angle)
            return True
        return False

    def step(self, frame, now):
        """Process one frame. Returns (yaw command, detections, capture_now)."""
        dt = now - self.last_time
        self.last_time = now
        # backup estimate: integrate the rotation produced by the command flown since the last step
        self.dr_angle += self.yaw * config.YAW_DEG_PER_UNIT * dt

        detections = []
        capture = False

        if self.state == HOVER:
            self.yaw = 0.0
            if now - self.state_start >= config.HOVER_TIME and frame is not None:
                print("Hover done - starting 360 deg scan.")
                self.angle = 0.0
                self.dr_angle = 0.0
                self.heading.reset(frame)
                self._set_state(SCAN, now)
                frame = None  # already used as the start view

        if self.state == SCAN:
            closed_loop = False
            if frame is not None:
                self.angle = abs(self.heading.update(frame))
                sim = self.heading.similarity_to_start(frame)
                if sim < 0.5:
                    self.seen_far_view = True
                closed_loop = (self.seen_far_view and self.angle >= config.LOOP_CLOSE_MIN_DEG
                               and sim >= config.LOOP_CLOSE_SIM)
                detections, self.last_mask = detect_tires(frame, self.params)
                capture = self._update_capture(detections)
            if self.angle >= config.SCAN_DEGREES or closed_loop:
                why = "back at start view" if closed_loop and self.angle < config.SCAN_DEGREES else "measured"
                print(f"Scan complete ({self.angle:.0f} deg, {why}).")
                self._set_state(DONE, now)
            elif abs(self.dr_angle) >= config.SCAN_DEGREES * 1.5:
                print(f"Yaw-command estimate passed {abs(self.dr_angle):.0f} deg - stopping.")
                self._set_state(DONE, now)
            elif now - self.state_start >= config.SCAN_TIMEOUT:
                print("Scan timeout - stopping.")
                self._set_state(DONE, now)
            else:
                self.yaw = config.YAW_RATE

        if self.state == DONE:
            self.yaw = 0.0

        return self.yaw, detections, capture
