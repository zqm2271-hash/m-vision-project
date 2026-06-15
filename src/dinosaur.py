from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


DINO_BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/dino"
DINO_IMAGES_URL = f"{DINO_BASE_URL}/images.tar.gz"
DINO_TRACKS_URL = f"{DINO_BASE_URL}/viff.xy"
DINO_README_URL = f"{DINO_BASE_URL}/README.txt"
DINO_CAMERAS_URL = f"{DINO_BASE_URL}/dino_Ps.mat"


@dataclass(frozen=True)
class DinosaurDataset:
    root: Path
    image_paths: list[Path]
    tracks: np.ndarray
    image_size: tuple[int, int]


def download_dinosaur_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _download(DINO_TRACKS_URL, root / "viff.xy")
    _download(DINO_README_URL, root / "README.txt")
    _download(DINO_CAMERAS_URL, root / "dino_Ps.mat")
    archive = root / "images.tar.gz"
    _download(DINO_IMAGES_URL, archive)
    image_dir = root / "images"
    if not image_dir.exists() or not list(image_dir.glob("*")):
        image_dir.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(image_dir)


def load_dinosaur_dataset(root: Path) -> DinosaurDataset:
    tracks_path = root / "viff.xy"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing Dinosaur tracks file: {tracks_path}")

    tracks = np.loadtxt(tracks_path, dtype=np.float64)
    if tracks.ndim != 2 or tracks.shape[1] % 2 != 0:
        raise ValueError("Dinosaur viff.xy must be an N x (2 * views) matrix.")
    view_count = tracks.shape[1] // 2

    image_paths = _find_images(root)[:view_count]
    if len(image_paths) < view_count:
        raise FileNotFoundError(f"Expected at least {view_count} Dinosaur images under: {root}")

    sample = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(f"Cannot read first Dinosaur image: {image_paths[0]}")
    height, width = sample.shape[:2]
    return DinosaurDataset(root=root, image_paths=image_paths, tracks=tracks, image_size=(width, height))


def load_camera_matrices(root: Path) -> list[np.ndarray]:
    mat_path = root / "dino_Ps.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing official Dinosaur camera matrix file: {mat_path}")
    data = loadmat(mat_path)
    matrices: list[np.ndarray] = []
    for key in sorted(data):
        value = data[key]
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.shape == (3, 4):
            matrices.append(value.astype(np.float64))
    if not matrices:
        for value in data.values():
            if isinstance(value, np.ndarray) and value.dtype == object:
                for item in value.ravel():
                    arr = np.asarray(item, dtype=np.float64)
                    if arr.shape == (3, 4):
                        matrices.append(arr)
    if not matrices:
        raise ValueError(f"No 3x4 camera matrices found in {mat_path}")
    return matrices


def observations_for_pair(tracks: np.ndarray, view1: int, view2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts1 = tracks[:, 2 * view1 : 2 * view1 + 2]
    pts2 = tracks[:, 2 * view2 : 2 * view2 + 2]
    valid = (pts1[:, 0] >= 0) & (pts1[:, 1] >= 0) & (pts2[:, 0] >= 0) & (pts2[:, 1] >= 0)
    return np.flatnonzero(valid), pts1[valid], pts2[valid]


def observations_for_view(tracks: np.ndarray, view: int) -> tuple[np.ndarray, np.ndarray]:
    pts = tracks[:, 2 * view : 2 * view + 2]
    valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
    return np.flatnonzero(valid), pts[valid]


def make_initial_intrinsics(image_size: tuple[int, int], focal_scale: float = 1.2) -> np.ndarray:
    width, height = image_size
    focal = focal_scale * max(width, height)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def _find_images(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".tif", ".tiff"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in extensions]
    return sorted(paths, key=lambda p: p.name)
