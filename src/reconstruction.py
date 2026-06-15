from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def colorize_tracks(image_paths: list[Path], tracks: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    colors = np.full((len(track_ids), 3), 220, dtype=np.uint8)
    loaded_images: dict[int, np.ndarray] = {}
    for out_idx, track_id in enumerate(track_ids):
        for view in range(tracks.shape[1] // 2):
            x, y = tracks[int(track_id), 2 * view : 2 * view + 2]
            if x < 0 or y < 0:
                continue
            if view not in loaded_images:
                image = cv2.imread(str(image_paths[view]), cv2.IMREAD_COLOR)
                if image is None:
                    break
                loaded_images[view] = image
            image = loaded_images[view]
            px = int(round(x))
            py = int(round(y))
            if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                bgr = image[py, px]
                colors[out_idx] = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)
                break
    return colors


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = np.full((len(points), 3), 230, dtype=np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def camera_centers(poses: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    centers = []
    for view in sorted(poses):
        R, t = poses[view]
        centers.append(-R.T @ t.reshape(3))
    return np.array(centers, dtype=np.float64)


def camera_centers_from_projection_matrices(camera_matrices: list[np.ndarray]) -> np.ndarray:
    centers = []
    for P in camera_matrices:
        _, _, vt = np.linalg.svd(P)
        center = vt[-1]
        centers.append(center[:3] / center[3])
    return np.array(centers, dtype=np.float64)
