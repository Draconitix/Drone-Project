"""Tire detection with plain OpenCV: dark, round(ish) blobs, ideally with a hole in the middle."""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import config


@dataclass
class Params:
    v_max: int = config.V_MAX
    chroma_max: int = config.CHROMA_MAX
    morph_size: int = config.MORPH_SIZE
    min_area: int = config.MIN_AREA
    max_area_frac: float = config.MAX_AREA_FRAC
    min_aspect: float = config.MIN_ASPECT
    min_circularity: float = config.MIN_CIRCULARITY
    min_solidity: float = config.MIN_SOLIDITY
    require_hole: bool = config.REQUIRE_HOLE
    hole_min_frac: float = config.HOLE_MIN_FRAC
    hole_max_offset: float = config.HOLE_MAX_OFFSET
    hole_bonus: float = config.HOLE_BONUS
    min_score: float = config.MIN_SCORE
    ring_detect: bool = config.RING_DETECT
    ring_hole_min_area: int = config.RING_HOLE_MIN_AREA
    ring_hole_min_circ: float = config.RING_HOLE_MIN_CIRC
    ring_hole_min_aspect: float = config.RING_HOLE_MIN_ASPECT
    ring_scale: float = config.RING_SCALE
    ring_min_dark: float = config.RING_MIN_DARK
    tire_to_hole: float = config.TIRE_TO_HOLE


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
    value = frame.max(axis=2)                       # HSV "V" (brightness)
    chroma = value - frame.min(axis=2)              # colourfulness, ~0 for black/grey
    mask = ((value <= p.v_max) & (chroma <= p.chroma_max)).astype(np.uint8) * 255
    k = np.ones((max(1, p.morph_size), max(1, p.morph_size)), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _largest_hole(contours, hierarchy, idx):
    """(area, contour) of the largest child contour (hole) of contour idx; (0, None) if none."""
    best, best_cnt = 0.0, None
    child = hierarchy[idx][2]
    while child != -1:
        a = cv2.contourArea(contours[child])
        if a > best:
            best, best_cnt = a, contours[child]
        child = hierarchy[child][0]
    return best, best_cnt


def _centroid(cnt):
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def hole_offset(outer, hole, outer_area):
    """Distance between hole centre and blob centre, as a fraction of the blob's diameter."""
    c1, c2 = _centroid(outer), _centroid(hole)
    if c1 is None or c2 is None:
        return float("inf")
    diameter = 2 * math.sqrt(outer_area / math.pi)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / diameter


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
    found_outer = set()
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

        # a tire's hole is big and sits in the middle (unlike gaps in shelves or frames)
        hole_area, hole_cnt = _largest_hole(contours, hierarchy, i)
        has_hole = (hole_area >= p.hole_min_frac * area
                    and hole_offset(cnt, hole_cnt, area) <= p.hole_max_offset)
        if p.require_hole and not has_hole:
            continue
        if circularity < p.min_circularity or solidity < p.min_solidity or aspect < p.min_aspect:
            continue

        score = circularity * solidity * (p.hole_bonus if has_hole else 1.0)
        if score < p.min_score:
            continue
        detections.append(Detection(cv2.boundingRect(cnt), score, cnt, has_hole,
                                    circularity, solidity, aspect))
        found_outer.add(i)

    # second pass: a tire that touches other dark things (shelf, ladder) merges into one
    # odd-shaped blob and fails the checks above, but its hole is still a clean, round
    # hole with a complete dark ring around it.
    if p.ring_detect:
        detections += _detect_by_hole(frame, mask, contours, hierarchy, found_outer, p)

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, mask


def _detect_by_hole(frame, mask, contours, hierarchy, found_outer, p):
    found = []
    img_h, img_w = mask.shape
    for i, hole in enumerate(contours):
        parent = hierarchy[i][3]
        if parent == -1 or parent in found_outer or len(hole) < 5:
            continue                       # only holes, of blobs not already detected
        area = cv2.contourArea(hole)
        if area < p.ring_hole_min_area:
            continue
        perimeter = cv2.arcLength(hole, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        ellipse = cv2.fitEllipse(hole)
        (cx, cy), (e1, e2), angle = ellipse
        aspect = min(e1, e2) / max(e1, e2) if max(e1, e2) > 0 else 0.0
        if circularity < p.ring_hole_min_circ or aspect < p.ring_hole_min_aspect:
            continue

        # fraction of "tire black" pixels on an ellipse just outside the hole
        ring = np.zeros_like(mask)
        cv2.ellipse(ring, ((cx, cy), (e1 * p.ring_scale, e2 * p.ring_scale), angle), 255, 2)
        on = ring > 0
        dark = float((mask[on] > 0).mean()) if on.any() else 0.0
        if dark < p.ring_min_dark:
            continue

        # tire outline ~ hole ellipse scaled up
        outer = ((cx, cy), (e1 * p.tire_to_hole, e2 * p.tire_to_hole), angle)
        poly = cv2.ellipse2Poly((int(cx), int(cy)),
                                (int(e1 * p.tire_to_hole / 2), int(e2 * p.tire_to_hole / 2)),
                                int(angle), 0, 360, 10).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(poly)
        x0, y0 = max(0, x), max(0, y)
        bbox = (x0, y0, min(img_w, x + w) - x0, min(img_h, y + h) - y0)
        score = circularity * dark * p.hole_bonus
        if score < p.min_score:
            continue
        found.append(Detection(bbox, score, poly, True, circularity, dark, aspect))
    return found


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
