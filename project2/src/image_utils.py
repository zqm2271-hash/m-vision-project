from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.keypoint import Keypoint


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.0


def create_demo_image(size: int = 520) -> np.ndarray:
    image = np.full((size, size, 3), 245, dtype=np.uint8)

    cv2.rectangle(image, (70, 80), (220, 230), (30, 80, 180), -1)
    cv2.rectangle(image, (95, 105), (195, 205), (245, 245, 245), 5)
    cv2.circle(image, (370, 150), 72, (40, 150, 90), -1)
    cv2.line(image, (315, 95), (425, 205), (250, 250, 250), 7)
    cv2.line(image, (425, 95), (315, 205), (250, 250, 250), 7)

    pts = np.array([[130, 360], [245, 300], [290, 410], [180, 455]], np.int32)
    cv2.fillConvexPoly(image, pts, (185, 60, 130))
    cv2.polylines(image, [pts], True, (35, 35, 35), 4)

    for x in range(330, 470, 35):
        for y in range(325, 455, 35):
            cv2.circle(image, (x, y), 7, (40, 40, 40), -1)

    cv2.putText(
        image,
        "CV",
        (64, 438),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (25, 25, 25),
        4,
        cv2.LINE_AA,
    )
    return image


def load_or_create_image(path: str | None, output_dir: Path) -> np.ndarray:
    if path is None:
        image = create_demo_image()
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
    cv2.imwrite(str(output_dir / "input.png"), image)
    return image


def rotate_image(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )
    return rotated, matrix


def transform_keypoints(
    keypoints: list[Keypoint],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> list[Keypoint]:
    projected: list[Keypoint] = []
    for kp in keypoints:
        x_new = matrix[0, 0] * kp.x + matrix[0, 1] * kp.y + matrix[0, 2]
        y_new = matrix[1, 0] * kp.x + matrix[1, 1] * kp.y + matrix[1, 2]
        if 0 <= x_new < width and 0 <= y_new < height:
            projected.append(Keypoint(float(x_new), float(y_new), kp.response, kp.angle))
    return projected
