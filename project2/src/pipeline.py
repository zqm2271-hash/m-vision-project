from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.descriptor import sift_like_descriptors
from src.detector import harris_interest_points
from src.image_utils import ensure_dir, load_or_create_image, rotate_image, transform_keypoints
from src.matcher import match_descriptors
from src.visualize import draw_keypoints, draw_matches


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    image = load_or_create_image(args.image, output_dir)
    rotated, matrix = rotate_image(image, args.angle)
    cv2.imwrite(str(output_dir / "rotated.png"), rotated)

    keypoints1 = harris_interest_points(image, max_points=args.max_points)
    keypoints2 = harris_interest_points(rotated, max_points=args.max_points)
    projected_keypoints = transform_keypoints(keypoints1, matrix, rotated.shape[1], rotated.shape[0])

    keypoints1, desc1 = sift_like_descriptors(image, keypoints1)
    keypoints2, desc2 = sift_like_descriptors(rotated, keypoints2)
    matches = match_descriptors(desc1, desc2, ratio=args.ratio, max_matches=args.max_matches)

    cv2.imwrite(str(output_dir / "input_keypoints.png"), draw_keypoints(image, keypoints1))
    cv2.imwrite(str(output_dir / "rotated_keypoints.png"), draw_keypoints(rotated, keypoints2))
    cv2.imwrite(str(output_dir / "rotated_projected_keypoints.png"), draw_keypoints(rotated, projected_keypoints))
    cv2.imwrite(str(output_dir / "matches.png"), draw_matches(image, rotated, keypoints1, keypoints2, matches))

    print(f"Input keypoints:   {len(keypoints1)}")
    print(f"Rotated keypoints: {len(keypoints2)}")
    print(f"Projected points:  {len(projected_keypoints)}")
    print(f"Matches:           {len(matches)}")
    print(f"Output directory:  {output_dir.resolve()}")
