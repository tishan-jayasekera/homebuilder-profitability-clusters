import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def segment_region(image_bgr, target_rgb, tolerance=25, min_area=600):
    if image_bgr is None:
        return []
    image = image_bgr.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target = np.uint8([[target_rgb[::-1]]])
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0][0]
    tol = int(tolerance)
    lower = np.array([max(target_hsv[0] - tol, 0), max(target_hsv[1] - tol, 0), max(target_hsv[2] - tol, 0)])
    upper = np.array([min(target_hsv[0] + tol, 179), min(target_hsv[1] + tol, 255), min(target_hsv[2] + tol, 255)])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        if len(coords) >= 3:
            poly = Polygon(coords)
            if poly.is_valid:
                polygons.append(poly)
    return polygons
