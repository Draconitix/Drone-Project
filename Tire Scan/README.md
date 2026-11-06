# Tire Scan

Flies the Pysimverse drone through a fixed sequence: take off, hover for 3 s, turn slowly through 360°, save a screenshot each time a tire comes into view, then land.

## Run
1. Open Pysimverse in the level that has the tires.
2. Run this from the `VT1` folder:
   ```
   .venv\Scripts\python "Tire Scan\main.py"
   ```
3. To stop early, press `q` in the video window or Ctrl-C in the terminal. The drone lands either way.

## Output
- `screenshots/tire_<time>_<n>.jpg`: the clean camera frame.
- `screenshots/tire_<time>_<n>_annotated.jpg`: the same frame with the detection box and the heading.
- `runs/<time>/frame_#####.jpg`: every raw frame from the flight, kept for tuning. Set `RECORD_FRAMES = False` in `config.py` to turn this off.

## How detection works (`detector.py`)
- Tires are black rubber, so the detector first keeps only dark pixels with little colour (`V_MAX`, `CHROMA_MAX`). Chroma (max − min of B, G, R) is used instead of HSV saturation. Saturation is unreliable on near-black pixels and was cutting the tire ring open.
- A second pass finds tires that touch other dark objects (a shelf, the ladder) and merge into one odd-shaped blob. It looks for a round hole with a complete dark ring around it (`RING_*` settings).
- It then keeps blobs that are large enough, roughly round or elliptical, and solid.
- The blob must have a hole (the rim). The hole must be at least `HOLE_MIN_FRAC` of the blob's area and near its centre (`HOLE_MAX_OFFSET`). Dark objects without a hole are rejected, such as shadows, legs, ladders and bottles. So are dark frames whose opening is off to one side.
- A screenshot is taken once a tire shows up in `CONFIRM_FRAMES` frames in a row.
- No second screenshot is taken until the drone has turned at least `REARM_DEGREES` and the view has cleared, or until it has turned `REARM_FORCE_DEGREES`.

## Tuning without flying
After a flight, tune the detector on the recorded frames:
```
.venv\Scripts\python "Tire Scan\tune_detector.py" "Tire Scan\runs\<time>"
.venv\Scripts\python "Tire Scan\tune_detector.py" "Tire Scan\runs\<time>" --report
```
- Adjust the trackbars until the tires are boxed and nothing else is.
- Use `n`/`p` to step through the frames.
- Press `q` to quit. The tuner prints the values to paste into `config.py`.

## Flight settings (`config.py`)
- `YAW_RATE = 0.06` turns at about 15.8 °/s, so a full turn takes about 23 s. In this level one yaw unit is about 263 °/s (measured from the recorded frames; the older 52.6 figure made the drone spin about 5 times).
- The heading is measured from the camera: the sideways image shift between frames, converted to degrees with `CAMERA_HFOV`. The scan stops after 360°, or earlier if the view matches the starting view again after at least `LOOP_CLOSE_MIN_DEG`. Because of this, a wrong yaw scale can no longer cause extra turns.
- If the drone stops short of a full turn or turns past it, adjust `CAMERA_HFOV`. A lower value reads fewer degrees, so the drone turns further.
- As backups, the drone also lands if the yaw-command estimate passes 540° or if `SCAN_TIMEOUT` runs out (1.5 times the expected time for one turn).
