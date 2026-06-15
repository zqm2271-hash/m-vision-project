from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.dinosaur import (
    download_dinosaur_dataset,
    load_camera_matrices,
    load_dinosaur_dataset,
    make_initial_intrinsics,
    observations_for_pair,
)
from src.geometry import estimate_fundamental
from src.multiview import reconstruct_from_official_cameras
from src.reconstruction import camera_centers, camera_centers_from_projection_matrices, colorize_tracks, save_ply
from src.sfm import reconstruct_dinosaur
from src.visualization import save_dinosaur_views, save_epipolar_plot, save_sfm_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 5: Dinosaur multi-view SfM reconstruction.")
    parser.add_argument(
        "--download-dinosaur",
        action="store_true",
        help="Download Oxford VGG Dinosaur images, tracked points, and camera matrices.",
    )
    parser.add_argument("--dinosaur-dir", type=Path, default=Path("data/dinosaur"), help="Dinosaur dataset directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--max-views", type=int, default=10, help="Maximum views for the self-estimated SfM chain.")
    parser.add_argument("--ransac-threshold", type=float, default=1.0, help="RANSAC threshold in pixels.")
    parser.add_argument(
        "--use-official-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use official camera matrices for the final clearer Dinosaur point cloud.",
    )
    return parser.parse_args()


def run_dinosaur(args: argparse.Namespace) -> None:
    if args.download_dinosaur:
        download_dinosaur_dataset(args.dinosaur_dir)

    dataset = load_dinosaur_dataset(args.dinosaur_dir)
    K = make_initial_intrinsics(dataset.image_size)
    result = reconstruct_dinosaur(
        dataset.tracks,
        K,
        max_views=args.max_views,
        ransac_threshold=args.ransac_threshold,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    colors = colorize_tracks(dataset.image_paths, dataset.tracks, result.track_ids)
    save_ply(args.output / "dinosaur_sparse.ply", result.points3d, colors)
    save_sfm_overview(result.points3d, camera_centers(result.poses), args.output / "dinosaur_reconstruction.png")

    official = None
    if args.use_official_cameras:
        camera_matrices = load_camera_matrices(args.dinosaur_dir)
        official = reconstruct_from_official_cameras(dataset.tracks, camera_matrices)
        official_colors = colorize_tracks(dataset.image_paths, dataset.tracks, official.track_ids)
        official_centers = camera_centers_from_projection_matrices(camera_matrices)
        save_ply(args.output / "dinosaur_official_dense.ply", official.points3d, official_colors)
        save_sfm_overview(
            official.points3d,
            official_centers,
            args.output / "dinosaur_official_reconstruction.png",
        )
        save_dinosaur_views(official.points3d, args.output / "dinosaur_three_views.png")

    first, second = result.initial_pair
    _, pts1, pts2 = observations_for_pair(dataset.tracks, first, second)
    _, mask = estimate_fundamental(pts1, pts2, args.ransac_threshold)
    img1 = cv2.imread(str(dataset.image_paths[first]), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(dataset.image_paths[second]), cv2.IMREAD_COLOR)
    save_epipolar_plot(img1, img2, pts1[mask], pts2[mask], result.F, args.output / "dinosaur_epipolar_lines.png")

    print_summary(args, dataset, result, official)


def print_summary(args, dataset, result, official) -> None:
    np.set_printoptions(precision=4, suppress=True)
    first, second = result.initial_pair
    print("\nMode: Oxford VGG Dinosaur multi-view SfM")
    print(f"Dataset images: {len(dataset.image_paths)}")
    print(f"Tracked feature points: {dataset.tracks.shape[0]}")
    print(f"Initial pair: frame {first + 1} and frame {second + 1}")
    print(f"Initial pose inliers: {result.initial_inliers}")
    print(f"Registered views: {[view + 1 for view in result.registered_views]}")
    print(f"Self-estimated sparse 3D points: {len(result.points3d)}")
    if official is not None:
        print(f"Official-camera filtered 3D points: {len(official.points3d)}")
        print(f"Mean reprojection error: {official.reprojection_errors.mean():.3f} px")

    print("\nApproximate intrinsic matrix K used for self-estimated F/E/R/t chain:")
    print(result.K)
    print("\nFundamental matrix F from initial pair:")
    print(result.F)
    print("\nEssential matrix E = K^T F K:")
    print(result.E)
    R, t = result.poses[second]
    print("\nRecovered relative pose for initial pair:")
    print("R =")
    print(R)
    print("t direction =")
    print(t.reshape(-1))

    print("\nSaved:")
    print(f"  {args.output / 'dinosaur_sparse.ply'}")
    print(f"  {args.output / 'dinosaur_reconstruction.png'}")
    print(f"  {args.output / 'dinosaur_epipolar_lines.png'}")
    if official is not None:
        print(f"  {args.output / 'dinosaur_official_dense.ply'}")
        print(f"  {args.output / 'dinosaur_official_reconstruction.png'}")
        print(f"  {args.output / 'dinosaur_three_views.png'}")


def main() -> None:
    run_dinosaur(parse_args())


if __name__ == "__main__":
    main()
