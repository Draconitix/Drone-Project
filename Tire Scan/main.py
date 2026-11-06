"""Entry point: take off, hover, rotate 360 deg, screenshot every tire seen, land.

Run with Pysimverse open:  .venv\\Scripts\\python "Tire Scan\\main.py"
"""
import contextlib
import io
import os
import time

import cv2
from pysimverse import Drone

import config
from detector import draw_detections
from scanner import DONE, TireScanner


def quiet(fn, *args, **kwargs):
    """Call a pysimverse method without its per-call console spam."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def save_screenshot(frame, detections, heading, count, stamp):
    """Save the clean frame and an annotated copy. Returns the clean path."""
    base = os.path.join(config.SCREENSHOT_DIR, f"tire_{stamp}_{count}")
    cv2.imwrite(base + ".jpg", frame)
    cv2.imwrite(base + "_annotated.jpg",
                draw_detections(frame, detections, f"heading {heading:.0f} deg"))
    return base + ".jpg"


def build_display(frame, detections, scanner, n_shots, flash):
    display = draw_detections(frame, detections)
    # small mask inset in the top-right corner
    mask = scanner.last_mask
    if mask is not None:
        h, w = frame.shape[:2]
        inset = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
        display[0:h // 4, w - w // 4:w] = inset
    cv2.putText(display, f"{scanner.state}  {scanner.angle:5.1f} deg  shots: {n_shots}",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if flash:
        cv2.putText(display, "Screenshot saved", (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return display


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    record_dir = os.path.join(config.RECORD_DIR, stamp)
    if config.RECORD_FRAMES:
        os.makedirs(record_dir, exist_ok=True)

    # initialization
    drone = Drone()
    drone.connect()
    time.sleep(0.4)
    quiet(drone.streamon)
    print("Taking off...")
    quiet(drone.take_off)

    period = 1.0 / config.LOOP_HZ
    scanner = TireScanner(start_time=time.time())
    shots = []           # (path, heading)
    flash_until = 0.0
    frame_no = 0

    try:
        while scanner.state != DONE:
            loop_start = time.time()

            frame, ok = quiet(drone.get_frame)
            if not ok:
                frame = None
            yaw, detections, capture = scanner.step(frame, loop_start)

            if frame is not None:
                if config.RECORD_FRAMES:
                    frame_no += 1
                    cv2.imwrite(os.path.join(record_dir, f"frame_{frame_no:05d}.jpg"), frame)
                if capture:
                    path = save_screenshot(frame, detections, scanner.angle, len(shots) + 1, stamp)
                    shots.append((path, scanner.angle))
                    flash_until = loop_start + 1.0
                    print(f"Tire detected at {scanner.angle:.0f} deg -> {path}")
                cv2.imshow(config.WINDOW_NAME,
                           build_display(frame, detections, scanner, len(shots), loop_start < flash_until))

            quiet(drone.send_rc_control, 0, 0, 0, yaw, config.CAMERA_ANGLE)

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
        quiet(drone.send_rc_control, 0, 0, 0, 0, config.CAMERA_ANGLE)
        print("Landing...")
        quiet(drone.land)
        time.sleep(1)
        drone.shutdown()

        print(f"\nScan finished: {len(shots)} tire screenshot(s).")
        for path, heading in shots:
            print(f"  {heading:6.1f} deg  {path}")
        if config.RECORD_FRAMES:
            print(f"Raw frames recorded to {record_dir}")


if __name__ == "__main__":
    main()
