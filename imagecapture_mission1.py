from pysimverse import Drone
import ctypes
import os
import threading
import time
import cv2
import numpy as np
import zmq

# settings
SPEED = 50          # max movement speed (left/right, forward/back, up/down)
YAW_SPEED = 0.5     # max rotation speed (yaw uses a small scale in pysimverse)
ACCEL = 150         # how fast movement ramps up (units per second)
DECEL = 200         # how fast movement ramps down (units per second)
YAW_ACCEL = 1.5     # how fast rotation ramps up/down (units per second)
LOOP_HZ = 30        # minimum control loop rate (the loop also wakes on every new frame)
EPSILON = 1e-3      # values closer to 0 than this snap to exactly 0
WINDOW_NAME = "Drone feed"
PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")
MESSAGE_TIME = 1.0  # seconds to show the "Photo saved" message
FPS_REPORT_EVERY = 5.0  # seconds between fps prints (0 = off)
VK_SPACE = 0x20

# Win32 keyboard access (reads the real held/released state of every key)
user32 = ctypes.windll.user32
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.FindWindowW.restype = ctypes.c_void_p


class FrameReader:
    """Receives and decodes video frames on a background thread.

    This thread is the only user of drone.video_socket; the main thread only
    uses the command socket, so no ZMQ socket is shared between threads.
    """

    def __init__(self, video_socket):
        self.socket = video_socket
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.stop_event = threading.Event()
        self.frame = None
        self.count = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def latest(self):
        with self.lock:
            return self.frame, self.count

    def _run(self):
        while not self.stop_event.is_set():
            try:
                if not self.socket.poll(100):
                    continue
                data = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break  # socket closed
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            with self.lock:
                self.frame = frame
                self.count += 1
            self.new_frame.set()


def send_rc(drone, lr, fb, ud, yaw):
    """Same message as drone.send_rc_control, but without printing every call."""
    drone.command_socket.send_json({
        "left_right": lr,
        "forward_backward": fb,
        "up_down": ud,
        "yaw": yaw,
        "cameraangle": 0,
        "streamon": None,
        "color": {"red": None, "green": None, "blue": None},
    })


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
# keep only the newest frame in the socket (must be set before connect)
drone.video_socket.setsockopt(zmq.CONFLATE, 1)
drone.connect()
time.sleep(0.4)
drone.streamon()

reader = FrameReader(drone.video_socket)
reader.start()

drone.take_off()

lr = fb = ud = yaw = 0
period = 1.0 / LOOP_HZ
last_time = time.time()

shown_count = 0
photo_count = 0
space_was_down = False
message_until = 0.0
overlay_shown = False
fps_start = last_time
fps_recv_start = reader.latest()[1]
fps_shown = 0

try:
    while True:
        # wake as soon as a new frame arrives, or after `period` at the latest
        reader.new_frame.wait(timeout=period)
        reader.new_frame.clear()

        loop_start = time.time()
        dt = loop_start - last_time
        last_time = loop_start

        last_frame, frame_count = reader.latest()

        # show the feed (overlay is drawn on a copy so saved photos stay clean)
        if last_frame is not None:
            overlay_on = loop_start < message_until
            if frame_count != shown_count or overlay_on != overlay_shown:
                display = last_frame
                if overlay_on:
                    display = last_frame.copy()
                    cv2.putText(display, f"Photo saved ({photo_count})", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, display)
                if frame_count != shown_count:
                    fps_shown += 1
                shown_count = frame_count
                overlay_shown = overlay_on
        cv2.waitKey(1)  # needed so the window refreshes

        if FPS_REPORT_EVERY and loop_start - fps_start >= FPS_REPORT_EVERY:
            elapsed = loop_start - fps_start
            print(f"camera {(frame_count - fps_recv_start) / elapsed:.1f} fps, "
                  f"shown {fps_shown / elapsed:.1f} fps")
            fps_start = loop_start
            fps_recv_start = frame_count
            fps_shown = 0

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

        send_rc(drone, lr, fb, ud, yaw)

except KeyboardInterrupt:
    pass

finally:
    send_rc(drone, 0, 0, 0, 0)
    drone.land()
    time.sleep(1)
    reader.stop()
    drone.shutdown()
