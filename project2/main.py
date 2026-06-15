from __future__ import annotations

import argparse

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local feature detection, description, and matching demo.")
    parser.add_argument("--image", default="picture/ren.jpg", help="Path to the input image.")
    parser.add_argument("--angle", type=float, default=25.0, help="Rotation angle in degrees.")
    parser.add_argument("--output", default="outputs_ren_split", help="Directory for output images.")
    parser.add_argument("--max-points", type=int, default=600, help="Maximum Harris corners per image.")
    parser.add_argument("--max-matches", type=int, default=80, help="Maximum matches to draw.")
    parser.add_argument("--ratio", type=float, default=0.9, help="Lowe ratio-test threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
