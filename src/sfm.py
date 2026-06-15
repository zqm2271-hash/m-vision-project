from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dinosaur import observations_for_pair, observations_for_view
from src.geometry import (
    essential_from_fundamental,
    estimate_fundamental,
    estimate_pose_pnp,
    recover_pose_from_essential,
    triangulate_with_poses,
)


@dataclass
class SfmResult:
    K: np.ndarray
    F: np.ndarray
    E: np.ndarray
    poses: dict[int, tuple[np.ndarray, np.ndarray]]
    points3d: np.ndarray
    track_ids: np.ndarray
    initial_pair: tuple[int, int]
    initial_inliers: int
    registered_views: list[int]


def reconstruct_dinosaur(
    tracks: np.ndarray,
    K: np.ndarray,
    max_views: int = 12,
    ransac_threshold: float = 1.0,
    min_pnp_points: int = 30,
) -> SfmResult:
    view_count = tracks.shape[1] // 2
    first_view, second_view = _choose_initial_pair(tracks, view_count)
    initial_ids, pts1, pts2 = observations_for_pair(tracks, first_view, second_view)
    F, f_mask = estimate_fundamental(pts1, pts2, ransac_threshold)
    E = essential_from_fundamental(F, K)
    R2, t2, pose_mask = recover_pose_from_essential(E, K, pts1[f_mask], pts2[f_mask])

    usable_ids = initial_ids[f_mask][pose_mask]
    usable_pts1 = pts1[f_mask][pose_mask]
    usable_pts2 = pts2[f_mask][pose_mask]

    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {
        first_view: (np.eye(3), np.zeros(3)),
        second_view: (R2, t2.reshape(3)),
    }
    points = triangulate_with_poses(K, poses[first_view], poses[second_view], usable_pts1, usable_pts2)
    point_by_track = {int(track_id): point for track_id, point in zip(usable_ids, points)}
    registered = [first_view, second_view]

    for view in _view_order({first_view, second_view}, second_view, view_count):
        if len(registered) >= max_views:
            break
        obs_ids, obs_pts = observations_for_view(tracks, view)
        common_ids = [int(track_id) for track_id in obs_ids if int(track_id) in point_by_track]
        if len(common_ids) < min_pnp_points:
            continue

        object_points = np.array([point_by_track[track_id] for track_id in common_ids], dtype=np.float64)
        image_lookup = {int(track_id): point for track_id, point in zip(obs_ids, obs_pts)}
        image_points = np.array([image_lookup[track_id] for track_id in common_ids], dtype=np.float64)
        try:
            R, t, _ = estimate_pose_pnp(object_points, image_points, K)
        except RuntimeError:
            continue
        poses[view] = (R, t)
        registered.append(view)

        for ref_view in reversed(registered[:-1]):
            new_count = _triangulate_new_tracks(tracks, K, poses, ref_view, view, point_by_track, ransac_threshold)
            if new_count > 40:
                break

    ordered_ids = np.array(sorted(point_by_track), dtype=int)
    ordered_points = np.array([point_by_track[int(track_id)] for track_id in ordered_ids], dtype=np.float64)
    filtered_ids, filtered_points = _filter_points(ordered_ids, ordered_points)
    return SfmResult(
        K=K,
        F=F,
        E=E,
        poses=poses,
        points3d=filtered_points,
        track_ids=filtered_ids,
        initial_pair=(first_view, second_view),
        initial_inliers=int(pose_mask.sum()),
        registered_views=registered,
    )


def _choose_initial_pair(tracks: np.ndarray, view_count: int) -> tuple[int, int]:
    best_pair = (0, 1)
    best_score = -1
    for i in range(view_count):
        for j in range(i + 2, view_count):
            ids, pts1, pts2 = observations_for_pair(tracks, i, j)
            if len(ids) < 80:
                continue
            parallax = np.median(np.linalg.norm(pts1 - pts2, axis=1))
            score = len(ids) * min(parallax, 80.0)
            if score > best_score:
                best_pair = (i, j)
                best_score = score
    return best_pair


def _view_order(registered: set[int], anchor: int, view_count: int) -> list[int]:
    candidates = list(range(view_count))
    for view in registered:
        if view in candidates:
            candidates.remove(view)
    return sorted(candidates, key=lambda v: (abs(v - anchor), v))


def _triangulate_new_tracks(
    tracks: np.ndarray,
    K: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    ref_view: int,
    view: int,
    point_by_track: dict[int, np.ndarray],
    ransac_threshold: float,
) -> int:
    ids, pts_ref, pts_cur = observations_for_pair(tracks, ref_view, view)
    keep = np.array([int(track_id) not in point_by_track for track_id in ids], dtype=bool)
    if keep.sum() < 20:
        return 0
    ids = ids[keep]
    pts_ref = pts_ref[keep]
    pts_cur = pts_cur[keep]
    try:
        _, mask = estimate_fundamental(pts_ref, pts_cur, ransac_threshold)
    except RuntimeError:
        return 0
    ids = ids[mask]
    pts_ref = pts_ref[mask]
    pts_cur = pts_cur[mask]
    new_points = triangulate_with_poses(K, poses[ref_view], poses[view], pts_ref, pts_cur)
    added = 0
    for track_id, point in zip(ids, new_points):
        if point[2] > -1000 and np.linalg.norm(point) < 1e4:
            point_by_track[int(track_id)] = point
            added += 1
    return added


def _filter_points(track_ids: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return track_ids, points
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    track_ids = track_ids[finite]
    centered = points - np.median(points, axis=0)
    distances = np.linalg.norm(centered, axis=1)
    keep = distances < np.percentile(distances, 97.0)
    return track_ids[keep], points[keep]
