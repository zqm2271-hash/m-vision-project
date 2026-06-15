from __future__ import annotations

import numpy as np


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.78,
    max_matches: int = 80,
) -> list[tuple[int, int, float]]:
    if len(desc1) == 0 or len(desc2) == 0:
        return []

    distances = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
    matches: list[tuple[int, int, float]] = []

    for i in range(distances.shape[0]):
        order = np.argsort(distances[i])
        if len(order) < 2:
            continue
        best, second = int(order[0]), int(order[1])
        best_dist = float(distances[i, best])
        second_dist = float(distances[i, second])
        if best_dist < ratio * second_dist:
            matches.append((i, best, best_dist))

    matches.sort(key=lambda item: item[2])
    return matches[:max_matches]
