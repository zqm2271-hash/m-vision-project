from __future__ import annotations

import cv2
import numpy as np


def estimate_fundamental(
    pts1: np.ndarray, pts2: np.ndarray, ransac_threshold: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=0.999,
        maxIters=8000,
    )
    if F is None or F.shape != (3, 3):
        raise RuntimeError("Fundamental matrix estimation failed.")
    if mask is not None and int(mask.sum()) >= 8:
        inlier_mask = mask.ravel().astype(bool)
        refined_F, _ = cv2.findFundamentalMat(pts1[inlier_mask], pts2[inlier_mask], method=cv2.FM_8POINT)
        if refined_F is not None and refined_F.shape == (3, 3):
            F = refined_F
    return F / np.linalg.norm(F), mask.ravel().astype(bool)


def choose_best_focal(
    F: np.ndarray,
    image_size: tuple[int, int],
    min_scale: float = 0.45,
    max_scale: float = 2.50,
    steps: int = 240,
) -> tuple[float, np.ndarray, float]:
    width, height = image_size
    center = (width / 2.0, height / 2.0)
    base = max(width, height)
    candidates = np.linspace(min_scale * base, max_scale * base, steps)

    best_focal = float(candidates[0])
    best_score = float("inf")
    best_K = _intrinsics(best_focal, center)
    for focal in candidates:
        K = _intrinsics(float(focal), center)
        E = K.T @ F @ K
        score = _essential_score(E)
        if score < best_score:
            best_focal, best_score, best_K = float(focal), float(score), K
    return best_focal, best_K, best_score


def recover_pose_from_fundamental(
    F: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = K.T @ F @ K
    E = _project_to_essential(E)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask.ravel().astype(bool)


def essential_from_fundamental(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    return _project_to_essential(K.T @ F @ K)


def recover_pose_from_essential(
    E: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask.ravel().astype(bool)


def triangulate_inlier_points(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
) -> np.ndarray:
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points = (homog[:3] / homog[3]).T
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def triangulate_with_poses(
    K: np.ndarray,
    pose1: tuple[np.ndarray, np.ndarray],
    pose2: tuple[np.ndarray, np.ndarray],
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    R1, t1 = pose1
    R2, t2 = pose2
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points = (homog[:3] / homog[3]).T
    return points[np.isfinite(points).all(axis=1)]


def estimate_pose_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    reprojection_error: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        K,
        None,
        iterationsCount=3000,
        reprojectionError=reprojection_error,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 8:
        raise RuntimeError("PnP pose estimation failed.")
    ok, rvec, tvec = cv2.solvePnP(
        object_points[inliers.ravel()].astype(np.float64),
        image_points[inliers.ravel()].astype(np.float64),
        K,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("PnP pose refinement failed.")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), inliers.ravel()


def _intrinsics(focal: float, center: tuple[float, float]) -> np.ndarray:
    cx, cy = center
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _essential_score(E: np.ndarray) -> float:
    singular_values = np.linalg.svd(E, compute_uv=False)
    s1, s2, s3 = singular_values
    if s1 <= 1e-12 or s2 <= 1e-12:
        return float("inf")
    equal_large_singulars = ((s1 - s2) / s1) ** 2
    zero_small_singular = (s3 / s2) ** 2
    return float(equal_large_singulars + zero_small_singular)


def _project_to_essential(E: np.ndarray) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(E)
    average = 0.5 * (singular_values[0] + singular_values[1])
    return U @ np.diag([average, average, 0.0]) @ Vt
