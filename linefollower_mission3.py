from pysimverse import Drone
import math
import time
import cv2
import numpy as np

# ---------------- settings (tune these) ----------------
# The pysimverse camera looks straight down after take-off and the drone sits at the
# centre of the image. Angles below are measured in the image: 0 = straight ahead (up),
# positive = to the right.

# red colour range in HSV (red wraps around hue 0, so two ranges)
RED_LOW_1, RED_HIGH_1 = (0, 100, 70), (10, 255, 255)
RED_LOW_2, RED_HIGH_2 = (170, 100, 70), (180, 255, 255)

# line tracking geometry (pixels, for the 360x240 camera image)
LOOK_RADIUS = 60        # look-ahead circle around the drone; the line's exits from it give the directions
CENTER_RADIUS = 25      # line pixels this close to the drone are used to stay centred on the line
MIN_ARC_SAMPLES = 2     # ignore red crossings of the circle narrower than this (noise)
CIRCLE_SAMPLES = 180

# end of the line
END_TIP_DIST = 26       # land when the line's tip is this close ahead (end crossbar is then under the drone)
DESCENT_RATIO = 1.35    # line looks this much wider than normal -> drone is descending (simulator auto-lands at the end)

# control gains / limits (axis directions verified in the simulator)
FWD_SPEED = 18          # speed along the line on straight sections
MIN_FWD_FRAC = 0.3      # slowest speed on sharp curves (fraction of FWD_SPEED)
SLOW_ANGLE = 1.0        # turn angle (rad) at which speed reaches its minimum
APPROACH_SPEED = 8      # speed while closing in on the end of the line (or returning to it)
K_YAW = 1.2             # yaw per radian of turn angle (+yaw = turn right)
YAW_MAX = 1.0
K_CENTER = 0.4          # speed per pixel of offset from the line's centre
ALPHA = 0.35            # command smoothing (0..1, higher = more responsive)

# timing
LOOP_HZ = 30
START_TIMEOUT = 10.0    # seconds to find the line after take-off before giving up
END_CONFIRM = 1.5       # seconds the line must stay lost before we decide it has ended

WINDOW_NAME = "Line follower"
KERNEL = np.ones((5, 5), np.uint8)
CIRCLE_ANGLES = np.linspace(-math.pi, math.pi, CIRCLE_SAMPLES, endpoint=False)
# -------------------------------------------------------


def clamp(value, limit):
    return max(-limit, min(limit, value))


def wrap(angle):
    """Wrap an angle to -pi..pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angle_of(dx, dy):
    """Image-space angle of a vector: 0 = up (ahead), positive = right."""
    return math.atan2(dx, -dy)


def red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1) | cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def find_arcs(local, cx, cy):
    """Angles where the local line crosses the look-ahead circle (one angle per crossing)."""
    h, w = local.shape
    xs = np.clip((cx + LOOK_RADIUS * np.sin(CIRCLE_ANGLES)).astype(int), 0, w - 1)
    ys = np.clip((cy - LOOK_RADIUS * np.cos(CIRCLE_ANGLES)).astype(int), 0, h - 1)
    on = local[ys, xs]
    if on.all() or not on.any():
        return []
    shift = int(np.argmin(on))           # start the scan on an "off" sample so no run wraps around
    rolled = np.roll(on, -shift)
    arcs, start = [], None
    for i, value in enumerate(np.append(rolled, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= MIN_ARC_SAMPLES:
                mid = (shift + (start + i - 1) / 2) % CIRCLE_SAMPLES
                arcs.append(wrap(-math.pi + 2 * math.pi * mid / CIRCLE_SAMPLES))
            start = None
    return arcs


def analyse(mask):
    """Find the piece of line under the drone and where it leaves the look-ahead circle."""
    h, w = mask.shape
    cx, cy = w / 2, h / 2
    disc = np.zeros_like(mask)
    cv2.circle(disc, (int(cx), int(cy)), LOOK_RADIUS + 4, 255, cv2.FILLED)
    near_mask = cv2.bitwise_and(mask, disc)
    ys, xs = np.nonzero(near_mask)
    if len(xs) == 0:
        return None

    # keep only the piece of line connected to the red pixel closest to the drone,
    # so neighbouring parts of the track (e.g. the other side of a hairpin) are ignored
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    _, labels = cv2.connectedComponents(near_mask)
    local = labels == labels[ys[i], xs[i]]

    # centre of the line close to the drone (for staying centred)
    ly, lx = np.nonzero(local)
    close = (lx - cx) ** 2 + (ly - cy) ** 2 <= CENTER_RADIUS ** 2
    if close.any():
        center = (lx[close].mean(), ly[close].mean())
    else:
        center = (float(xs[i]), float(ys[i]))

    # apparent line width (grows when the drone descends)
    dist = cv2.distanceTransform(local.astype(np.uint8) * 255, cv2.DIST_L2, 3)
    width = 2 * float(dist.max())

    return {
        "local": local,
        "points": (lx, ly),
        "center": center,
        "arcs": find_arcs(local, cx, cy),
        "width": width,
    }


def tip_distance(points, cx, cy, direction):
    """How far the line reaches ahead of the drone along `direction` (pixels)."""
    lx, ly = points
    proj = (lx - cx) * math.sin(direction) - (ly - cy) * math.cos(direction)
    return float(proj.max())


def draw_debug(frame, info, target, cmd, status):
    view = frame.copy()
    h, w = view.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.circle(view, (cx, cy), LOOK_RADIUS, (255, 255, 0), 1)
    cv2.drawMarker(view, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
    if info is not None:
        contours, _ = cv2.findContours(info["local"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 255, 0), 2)
        for a in info["arcs"]:
            p = (int(cx + LOOK_RADIUS * math.sin(a)), int(cy - LOOK_RADIUS * math.cos(a)))
            cv2.circle(view, p, 4, (255, 0, 255), -1)
        cv2.circle(view, (int(info["center"][0]), int(info["center"][1])), 4, (0, 128, 255), -1)
    if target is not None:
        p = (int(cx + LOOK_RADIUS * math.sin(target)), int(cy - LOOK_RADIUS * math.cos(target)))
        cv2.line(view, (cx, cy), p, (0, 255, 255), 2)
    lr, fb, yaw = cmd
    cv2.putText(view, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(view, f"lr {lr:+.1f} fb {fb:+.1f} yaw {yaw:+.2f}",
                (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imshow(WINDOW_NAME, view)


# initialization
drone = Drone()
drone.connect()
time.sleep(0.4)
drone.streamon()
drone.take_off()

lr = fb = yaw = 0.0
period = 1.0 / LOOP_HZ
start_time = time.time()
line_seen_once = False
lost_since = None
heading = 0.0            # last chosen direction along the line (image angle)
approaching_end = False  # set once the line only continues behind the drone
behind = math.pi         # direction the line comes from while approaching the end
normal_width = None      # typical line width at cruise altitude (learned while following)

try:
    while True:
        loop_start = time.time()

        frame, is_success = drone.get_frame()
        if is_success:
            mask = red_mask(frame)
            h, w = mask.shape
            cx, cy = w / 2, h / 2
            info = analyse(mask)
            target = None
            t_lr = t_fb = t_yaw = 0.0

            if info is None:
                ys, xs = np.nonzero(mask)
                if line_seen_once and not approaching_end and len(xs):
                    # drifted off the line but it is still in view: head back to the closest red pixel
                    lost_since = None
                    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
                    target = angle_of(xs[i] - cx, ys[i] - cy)
                    t_fb = APPROACH_SPEED * math.cos(target)
                    t_lr = APPROACH_SPEED * math.sin(target)
                    status = "Returning to line"
                elif not line_seen_once:
                    status = "Searching for line..."
                    if loop_start - start_time > START_TIMEOUT:
                        print("No red line found - landing.")
                        break
                else:
                    if lost_since is None:
                        lost_since = loop_start
                    status = f"Line lost ({loop_start - lost_since:.1f}s)"
                    if approaching_end or loop_start - lost_since > END_CONFIRM:
                        print("Line ended - landing.")
                        break
            else:
                line_seen_once = True
                lost_since = None
                arcs = info["arcs"]
                center = info["center"]

                if not approaching_end:
                    # drop the crossing we came from, then continue in the direction closest to the last one
                    ahead = list(arcs)
                    if len(ahead) >= 2:
                        ahead.remove(min(ahead, key=lambda a: abs(wrap(a - (heading + math.pi)))))
                    elif len(ahead) == 1 and abs(wrap(ahead[0] - heading)) > math.pi / 2:
                        behind = ahead[0]       # only the way we came is left: the line ends ahead
                        ahead = []
                    if ahead:
                        heading = min(ahead, key=lambda a: abs(wrap(a - heading)))
                    else:
                        approaching_end = True
                        if not arcs:
                            behind = wrap(heading + math.pi)

                if approaching_end:
                    # the line ends ahead: creep along it and land when its tip is under the drone
                    status = "Approaching end"
                    behind = min(arcs, key=lambda a: abs(wrap(a - behind))) if arcs else behind
                    forward = wrap(behind + math.pi)
                    if tip_distance(info["points"], cx, cy, forward) <= END_TIP_DIST:
                        print("Reached the end of the line - landing.")
                        break
                    if normal_width and info["width"] > DESCENT_RATIO * normal_width:
                        # the simulator registered the end and is landing the drone: stop moving
                        print("End of the line reached (simulator is landing) - landing.")
                        break
                    target = forward
                    speed = APPROACH_SPEED
                else:
                    status = "Following"
                    normal_width = info["width"] if normal_width is None else 0.9 * normal_width + 0.1 * info["width"]
                    target = heading
                    speed = FWD_SPEED * max(MIN_FWD_FRAC, 1 - abs(heading) / SLOW_ANGLE)
                    t_yaw = clamp(K_YAW * heading, YAW_MAX)

                # move toward the look-ahead point, plus a pull back to the middle of the line
                t_fb = speed * math.cos(target) - K_CENTER * (center[1] - cy)
                t_lr = speed * math.sin(target) + K_CENTER * (center[0] - cx)
                t_fb, t_lr = clamp(t_fb, FWD_SPEED), clamp(t_lr, FWD_SPEED)

            # smooth the commands
            lr += ALPHA * (t_lr - lr)
            fb += ALPHA * (t_fb - fb)
            yaw += ALPHA * (t_yaw - yaw)

            draw_debug(frame, info, target, (lr, fb, yaw), status)

        drone.send_rc_control(lr, fb, 0, yaw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Aborted by user - landing.")
            break

        # keep a steady loop rate
        remaining = period - (time.time() - loop_start)
        if remaining > 0:
            time.sleep(remaining)

except KeyboardInterrupt:
    print("Interrupted - landing.")

finally:
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()
    time.sleep(1)
    drone.shutdown()
