import math
from typing import Union

import numpy as np


def wrap180(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap degrees into [-180, 180)."""
    arr = np.asarray(deg)
    wrapped = ((arr + 180.0) % 360.0) - 180.0
    return float(wrapped) if arr.shape == () else wrapped


def ang_diff_deg(a: float, b: float) -> float:
    """Absolute wrapped difference in degrees."""
    return float(abs(wrap180(a - b)))


def heading_deg_from_vec(v: np.ndarray) -> float:
    """Compute heading in degrees from a 2D vector."""
    return float(math.degrees(math.atan2(v[1], v[0])))


def unit_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    """Convert yaw in degrees to unit vector."""
    r = math.radians(float(wrap180(yaw_deg)))
    return np.array([math.cos(r), math.sin(r)], dtype=float)


def cumulative_dist(points_xy: np.ndarray) -> np.ndarray:
    """Compute cumulative arc-length along polyline."""
    if len(points_xy) < 2:
        return np.array([0.0])
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def sanitize(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("|", "-")
    )
