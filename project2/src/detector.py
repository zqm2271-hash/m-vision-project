from __future__ import annotations

import cv2
import numpy as np

from src.image_utils import to_gray_float
from src.keypoint import Keypoint


def harris_interest_points(
    image: np.ndarray,
    max_points: int = 350,
    quality: float = 0.01,
    min_distance: int = 8,
    border: int = 20,
    k: float = 0.04,
) -> list[Keypoint]:
    gray = to_gray_float(image)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)

    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ixx = cv2.GaussianBlur(ix * ix, (7, 7), 1.5)
    iyy = cv2.GaussianBlur(iy * iy, (7, 7), 1.5)
    ixy = cv2.GaussianBlur(ix * iy, (7, 7), 1.5)

    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    response = det - k * trace * trace

    response[:border, :] = 0
    response[-border:, :] = 0
    response[:, :border] = 0
    response[:, -border:] = 0

    threshold = quality * float(response.max())
    window = np.ones((2 * min_distance + 1, 2 * min_distance + 1), np.uint8)
    local_max = cv2.dilate(response, window)
    mask = (response == local_max) & (response > threshold)

    ys, xs = np.nonzero(mask)
    candidates = sorted(
        (Keypoint(float(x), float(y), float(response[y, x])) for y, x in zip(ys, xs)),
        key=lambda p: p.response,
        reverse=True,
    )
    return candidates[:max_points]
