from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_epipolar_plot(
    img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = min(20, len(pts1))
    pts1_sample = pts1[:sample_count]
    pts2_sample = pts2[:sample_count]
    lines2 = cv2.computeCorrespondEpilines(pts1_sample.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    canvas = img2.copy()
    height, width = canvas.shape[:2]
    rng = np.random.default_rng(3)
    for line, point in zip(lines2, pts2_sample):
        color = tuple(int(x) for x in rng.integers(40, 255, size=3))
        a, b, c = line
        x0, x1 = 0, width - 1
        y0 = int((-c - a * x0) / b) if abs(b) > 1e-9 else 0
        y1 = int((-c - a * x1) / b) if abs(b) > 1e-9 else height - 1
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.round(point).astype(int)), 5, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def save_sfm_overview(points3d: np.ndarray, camera_centers: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    if len(points3d) > 0:
        centered = points3d - np.median(points3d, axis=0)
        distances = np.linalg.norm(centered, axis=1)
        keep = distances < np.percentile(distances, 96)
        p = centered[keep]
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], s=2, c=p[:, 2], cmap="viridis", alpha=0.75)
    if len(camera_centers) > 0:
        c = camera_centers - np.median(camera_centers, axis=0)
        ax.plot(c[:, 0], c[:, 2], -c[:, 1], "r-o", linewidth=1.5, markersize=4, label="cameras")
        ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.set_title("Dinosaur sparse SfM reconstruction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_dinosaur_views(points3d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points3d) == 0:
        return
    p = points3d - np.median(points3d, axis=0)
    distances = np.linalg.norm(p, axis=1)
    p = p[distances < np.percentile(distances, 99.0)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    views = [
        (0, 2, "front / side", "X", "Z"),
        (0, 1, "top", "X", "Y"),
        (1, 2, "profile", "Y", "Z"),
    ]
    for ax, (a, b, title, xlabel, ylabel) in zip(axes, views):
        ax.scatter(p[:, a], p[:, b], s=1.2, c=p[:, 2], cmap="viridis", alpha=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
