from __future__ import annotations

import cv2
import numpy as np

from src.image_utils import to_gray_float
from src.keypoint import Keypoint


def dominant_orientation(gray: np.ndarray, x: float, y: float, radius: int = 8) -> float:
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(1, xi - radius), min(w - 1, xi + radius + 1)
    y0, y1 = max(1, yi - radius), min(h - 1, yi + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag, ori = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ori, bins=36, range=(0, 360), weights=mag)
    return float(np.argmax(hist) * 10.0)


def sift_like_descriptors(
    image: np.ndarray,
    keypoints: list[Keypoint],
    patch_size: int = 16,
    cells: int = 4,
    bins: int = 8,
) -> tuple[list[Keypoint], np.ndarray]:
    gray = to_gray_float(image)
    h, w = gray.shape
    half = patch_size // 2
    cell_size = patch_size // cells

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ori = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    valid_points: list[Keypoint] = []
    descs: list[np.ndarray] = []

    for kp in keypoints:
        x, y = int(round(kp.x)), int(round(kp.y))
        if x - half < 1 or x + half >= w - 1 or y - half < 1 or y + half >= h - 1:
            continue

        angle = dominant_orientation(gray, kp.x, kp.y, half)
        local_mag = mag[y - half : y + half, x - half : x + half]
        local_ori = (ori[y - half : y + half, x - half : x + half] - angle) % 360.0

        descriptor = np.zeros((cells, cells, bins), dtype=np.float32)
        weight = cv2.getGaussianKernel(patch_size, patch_size / 2)
        weight = weight @ weight.T
        local_mag = local_mag * weight

        for row in range(patch_size):
            for col in range(patch_size):
                cell_y = min(row // cell_size, cells - 1)
                cell_x = min(col // cell_size, cells - 1)
                bin_idx = int(local_ori[row, col] / (360.0 / bins)) % bins
                descriptor[cell_y, cell_x, bin_idx] += local_mag[row, col]

        vector = descriptor.ravel()
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            continue
        vector = vector / norm
        vector = np.clip(vector, 0.0, 0.2)
        vector = vector / (np.linalg.norm(vector) + 1e-8)

        valid_points.append(Keypoint(kp.x, kp.y, kp.response, angle))
        descs.append(vector.astype(np.float32))

    if not descs:
        return [], np.empty((0, cells * cells * bins), dtype=np.float32)
    return valid_points, np.vstack(descs)
