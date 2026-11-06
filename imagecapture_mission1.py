from pysimverse import Drone
import ctypes
import os
import time
import cv2

# settings
SPEED = 50          # max movement speed (left/right, forward/back, up/down)
YAW_SPEED = 0.5     # max rotation speed (yaw uses a small scale in pysimverse)
ACCEL = 150         # how fast movement ramps up (units per second)
DECEL = 200         # how fast movement ramps down (units per second)
YAW_ACCEL = 1.5     # how fast rotation ramps up/down (units per second)
LOOP_HZ = 30        # control loop rate
EPSILON = 1e-3      # values closer to 0 than this snap to exactly 0
WINDOW_NAME = "Drone feed"
PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")
MESSAGE_TIME = 1.0  # seconds to show the "Photo saved" message
VK_SPACE = 0x20

# Win32 keyboard access (reads the real held/released state of every key)
user32 = ctypes.windll.user32
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.FindWindowW.restype = ctypes.c_void_p


def feed_window_focused():
    hwnd = user32.FindWindowW(None, WINDOW_NAME)
    return bool(hwnd) and hwnd == user32.GetForegroundWindow()


def held(ch):
    return bool(user32.GetAsyncKeyState(ord(ch.upper())) & 0x8000)


def space_held():
    return bool(user32.GetAsyncKeyState(VK_SPACE) & 0x8000)


def take_photo(frame, count):
    """Save the frame as a .jpg in PHOTO_DIR. Returns True on success."""
    if frame is None:
        print("No frame available yet - photo not taken.")
        return False
    filename = f"photo_{time.strftime('%Y%m%d_%H%M%S')}_{count}.jpg"
    path = os.path.join(PHOTO_DIR, filename)
    if cv2.imwrite(path, frame):
        print(f"Photo saved: {path}")
        return True
    print(f"Failed to save photo: {path}")
    return False


def approach(current, target, accel, decel, dt):
    """Move current toward target, limited by accel (speeding up) or decel (slowing down)."""
    rate = accel if abs(target) > abs(current) else decel
    step = rate * dt
    if current < target:
        current = min(current + step, target)
    else:
        current = max(current - step, target)
    return 0 if abs(current) < EPSILON else current


print("Keyboard controls (click the video window first; keys can be combined):")
print("  W/S = forward/back   A/D = left/right")
print("  I/K = up/down        J/L = rotate left/right")
print("  SPACE = take photo   Q = land and quit")

os.makedirs(PHOTO_DIR, exist_ok=True)

# initialization
drone = Drone()
drone.connect()
time.sleep(0.4)
drone.streamon()
drone.take_off()

lr = fb = ud = yaw = 0
period = 1.0 / LOOP_HZ
last_time = time.time()

last_frame = None
photo_count = 0
space_was_down = False
message_until = 0.0

try:
    while True:
        loop_start = time.time()
        dt = loop_start - last_time
        last_time = loop_start

        frame, is_success = drone.get_frame()
        if is_success:
            last_frame = frame

        # show the feed (overlay is drawn on a copy so saved photos stay clean)
        if last_frame is not None:
            display = last_frame
            if loop_start < message_until:
                display = last_frame.copy()
                cv2.putText(display, f"Photo saved ({photo_count})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)  # needed so the window refreshes

        # read keys (only while the feed window is focused)
        if feed_window_focused():
            if held('q'):
                break

            # take one photo per space bar press
            space_down = space_held()
            if space_down and not space_was_down:
                if take_photo(last_frame, photo_count + 1):
                    photo_count += 1
                    message_until = loop_start + MESSAGE_TIME
            space_was_down = space_down

            target_lr = SPEED * (held('d') - held('a'))
            target_fb = SPEED * (held('w') - held('s'))
            target_ud = SPEED * (held('i') - held('k'))
            target_yaw = YAW_SPEED * (held('l') - held('j'))
        else:
            space_was_down = False
            target_lr = target_fb = target_ud = target_yaw = 0

        # smoothly ramp toward the targets
        lr = approach(lr, target_lr, ACCEL, DECEL, dt)
        fb = approach(fb, target_fb, ACCEL, DECEL, dt)
        ud = approach(ud, target_ud, ACCEL, DECEL, dt)
        yaw = approach(yaw, target_yaw, YAW_ACCEL, YAW_ACCEL, dt)

        drone.send_rc_control(lr, fb, ud, yaw)

        # keep a steady loop rate
        remaining = period - (time.time() - loop_start)
        if remaining > 0:
            time.sleep(remaining)

except KeyboardInterrupt:
    pass

finally:
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()
    time.sleep(1)
    drone.shutdown()
