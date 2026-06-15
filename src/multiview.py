from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OfficialReconstruction:
    points3d: np.ndarray
    track_ids: np.ndarray
    reprojection_errors: np.ndarray
    observations_per_point: np.ndarray


def reconstruct_from_official_cameras(
    tracks: np.ndarray,
    camera_matrices: list[np.ndarray],
    min_views: int = 4,
    max_reprojection_error: float = 2.5,
) -> OfficialReconstruction:
    points: list[np.ndarray] = []
    track_ids: list[int] = []
    errors: list[float] = []
    observation_counts: list[int] = []
    view_count = min(len(camera_matrices), tracks.shape[1] // 2)

    for track_id, row in enumerate(tracks):
        observations = []
        projections = []
        for view in range(view_count):
            x, y = row[2 * view : 2 * view + 2]
            if x < 0 or y < 0:
                continue
            observations.append((x, y))
            projections.append(camera_matrices[view])
        if len(observations) < min_views:
            continue

        point = triangulate_n_view(projections, observations)
        error = mean_reprojection_error(point, projections, observations)
        if np.isfinite(error) and error <= max_reprojection_error:
            points.append(point)
            track_ids.append(track_id)
            errors.append(error)
            observation_counts.append(len(observations))

    if not points:
        return OfficialReconstruction(
            points3d=np.empty((0, 3)),
            track_ids=np.empty((0,), dtype=int),
            reprojection_errors=np.empty((0,)),
            observations_per_point=np.empty((0,), dtype=int),
        )

    ids = np.array(track_ids, dtype=int)
    pts = np.array(points, dtype=np.float64)
    errs = np.array(errors, dtype=np.float64)
    obs = np.array(observation_counts, dtype=int)
    keep = _robust_shape_filter(pts)
    return OfficialReconstruction(pts[keep], ids[keep], errs[keep], obs[keep])


def triangulate_n_view(projections: list[np.ndarray], observations: list[tuple[float, float]]) -> np.ndarray:
    rows = []
    for P, (x, y) in zip(projections, observations):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    homog = vt[-1]
    if abs(homog[3]) < 1e-12:
        return np.full(3, np.nan)
    return homog[:3] / homog[3]


def mean_reprojection_error(
    point: np.ndarray, projections: list[np.ndarray], observations: list[tuple[float, float]]
) -> float:
    if not np.isfinite(point).all():
        return float("inf")
    homog = np.append(point, 1.0)
    residuals = []
    for P, observed in zip(projections, observations):
        projected = P @ homog
        if abs(projected[2]) < 1e-12:
            return float("inf")
        xy = projected[:2] / projected[2]
        residuals.append(np.linalg.norm(xy - np.asarray(observed)))
    return float(np.mean(residuals))


def _robust_shape_filter(points: np.ndarray) -> np.ndarray:
    centered = points - np.median(points, axis=0)
    scale = np.median(np.linalg.norm(centered, axis=1))
    if scale <= 1e-12:
        return np.ones(len(points), dtype=bool)
    normalized = centered / scale
    axis_limits = np.percentile(np.abs(normalized), 98.5, axis=0)
    return (np.abs(normalized) <= axis_limits).all(axis=1)
