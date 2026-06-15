from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    response: float
    angle: float = 0.0
