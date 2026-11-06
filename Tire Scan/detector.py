"""Tire detection with plain OpenCV: dark, round(ish) blobs, ideally with a hole in the middle."""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import config


@dataclass
class Params:
    v_max: int = config.V_MAX
    s_max: int = config.S_MAX
    morph_size: int = config.MORPH_SIZE
    min_area: int = config.MIN_AREA
    max_area_frac: float = config.MAX_AREA_FRAC
    min_aspect: float = config.MIN_ASPECT
    min_circularity: float = config.MIN_CIRCULARITY
    min_solidity: float = config.MIN_SOLIDITY
    require_hole: bool = config.REQUIRE_HOLE
    hole_min_frac: float = config.HOLE_MIN_FRAC
    hole_bonus: float = config.HOLE_BONUS
    min_score: float = config.MIN_SCORE


@dataclass
class Detection:
    bbox: tuple            # x, y, w, h
    score: float
    contour: np.ndarray = field(repr=False)
    has_hole: bool = False
    circularity: float = 0.0
    solidity: float = 0.0
    aspect: float = 0.0

    @property
    def center(self):
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2


def tire_mask(frame, p):
    """Binary mask of 'tire black' pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, p.s_max, p.v_max))
    k = np.ones((max(1, p.morph_size), max(1, p.morph_size)), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _hole_area(contours, hierarchy, idx):
    """Largest child contour (hole) area of contour idx, 0 if none."""
    best = 0.0
    child = hierarchy[idx][2]
    while child != -1:
        best = max(best, cv2.contourArea(contours[child]))
        child = hierarchy[child][0]
    return best


def detect_tires(frame, params=None):
    """Return (detections sorted by score, mask)."""
    p = params or Params()
    mask = tire_mask(frame, p)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], mask
    hierarchy = hierarchy[0]
    max_area = p.max_area_frac * frame.shape[0] * frame.shape[1]

    detections = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:      # skip holes themselves; only outer contours
            continue
        area = cv2.contourArea(cnt)
        if area < p.min_area or area > max_area or len(cnt) < 5:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area else 0.0
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        aspect = min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0

        has_hole = _hole_area(contours, hierarchy, i) >= p.hole_min_frac * area
        if p.require_hole and not has_hole:
            continue
        if circularity < p.min_circularity or solidity < p.min_solidity or aspect < p.min_aspect:
            continue

        score = circularity * solidity * (p.hole_bonus if has_hole else 1.0)
        if score < p.min_score:
            continue
        detections.append(Detection(cv2.boundingRect(cnt), score, cnt, has_hole,
                                    circularity, solidity, aspect))

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, mask


def draw_detections(frame, detections, label_extra=""):
    """Return an annotated copy of frame."""
    out = frame.copy()
    for d in detections:
        x, y, w, h = d.bbox
        color = (0, 255, 0) if d.has_hole else (0, 200, 255)
        cv2.drawContours(out, [d.contour], -1, color, 1)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"tire {d.score:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if label_extra:
        cv2.putText(out, label_extra, (6, out.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
