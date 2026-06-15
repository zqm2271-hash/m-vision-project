from __future__ import annotations

import cv2
import numpy as np

from src.keypoint import Keypoint


def draw_keypoints(image: np.ndarray, keypoints: list[Keypoint]) -> np.ndarray:
    output = image.copy()
    for kp in keypoints:
        center = (int(round(kp.x)), int(round(kp.y)))
        cv2.circle(output, center, 4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(output, center, 1, (0, 255, 255), -1, cv2.LINE_AA)
    return output


def draw_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    keypoints1: list[Keypoint],
    keypoints2: list[Keypoint],
    matches: list[tuple[int, int, float]],
) -> np.ndarray:
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    canvas = np.full((max(h1, h2), w1 + w2, 3), 245, dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1 : w1 + w2] = image2

    rng = np.random.default_rng(7)
    for idx1, idx2, _ in matches:
        p1 = (int(round(keypoints1[idx1].x)), int(round(keypoints1[idx1].y)))
        p2 = (int(round(keypoints2[idx2].x)) + w1, int(round(keypoints2[idx2].y)))
        color = tuple(int(v) for v in rng.integers(40, 235, size=3))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, color, -1, cv2.LINE_AA)
    return canvas
