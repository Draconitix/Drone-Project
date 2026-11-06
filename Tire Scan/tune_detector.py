"""Tune the tire detector offline on saved images / recorded runs (no flying needed).

Usage:
    python tune_detector.py <image | folder | runs/<timestamp>>      # interactive trackbars
    python tune_detector.py <image | folder> --report                 # just print detections

Keys (interactive):  n / space = next image,  p = previous,  q / Esc = quit and print settings.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

from detector import Params, detect_tires, draw_detections

WINDOW = "Tire detector tuner"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# trackbar name -> (Params field, max value, scale)  value = trackbar / scale
TRACKBARS = {
    "V max": ("v_max", 255, 1),
    "S max": ("s_max", 255, 1),
    "Morph": ("morph_size", 15, 1),
    "Min area": ("min_area", 5000, 1),
    "Min aspect x100": ("min_aspect", 100, 100),
    "Min circ x100": ("min_circularity", 100, 100),
    "Min solid x100": ("min_solidity", 100, 100),
    "Min score x100": ("min_score", 150, 100),
    "Need hole": ("require_hole", 1, 1),
}


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    files = [f for f in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
             if f.lower().endswith(EXTS)]
    return files


def read_params():
    p = Params()
    for name, (attr, _, scale) in TRACKBARS.items():
        v = cv2.getTrackbarPos(name, WINDOW)
        if attr == "require_hole":
            v = bool(v)
        elif scale != 1:
            v = v / scale
        setattr(p, attr, v)
    return p


def print_settings(p):
    print("\n# paste into config.py")
    for attr, cfg in [("v_max", "V_MAX"), ("s_max", "S_MAX"), ("morph_size", "MORPH_SIZE"),
                      ("min_area", "MIN_AREA"), ("min_aspect", "MIN_ASPECT"),
                      ("min_circularity", "MIN_CIRCULARITY"), ("min_solidity", "MIN_SOLIDITY"),
                      ("min_score", "MIN_SCORE"), ("require_hole", "REQUIRE_HOLE")]:
        print(f"{cfg} = {getattr(p, attr)}")


def report(files):
    total = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        dets, _ = detect_tires(img)
        total += bool(dets)
        desc = ", ".join(f"score={d.score:.2f} hole={d.has_hole} bbox={d.bbox}" for d in dets)
        print(f"{os.path.basename(f)}: {len(dets)} tire(s) {desc}")
    print(f"\n{total}/{len(files)} images with at least one tire")


def interactive(files):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    defaults = Params()
    for name, (attr, maxv, scale) in TRACKBARS.items():
        cv2.createTrackbar(name, WINDOW, int(round(float(getattr(defaults, attr)) * scale)), maxv,
                           lambda _: None)

    idx = 0
    while True:
        img = cv2.imread(files[idx])
        if img is None:
            idx = (idx + 1) % len(files)
            continue
        p = read_params()
        dets, mask = detect_tires(img, p)
        view = draw_detections(img, dets, f"{idx + 1}/{len(files)}  {len(dets)} tire(s)")
        cv2.imshow(WINDOW, np.hstack([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('n'), ord(' ')):
            idx = (idx + 1) % len(files)
        elif key == ord('p'):
            idx = (idx - 1) % len(files)
    print_settings(read_params())
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="image file, folder, or runs/<timestamp> recording")
    ap.add_argument("--report", action="store_true", help="print detections instead of opening the tuner")
    args = ap.parse_args()

    files = collect_images(args.path)
    if not files:
        sys.exit(f"No images found at {args.path}")
    if args.report:
        report(files)
    else:
        interactive(files)


if __name__ == "__main__":
    main()
