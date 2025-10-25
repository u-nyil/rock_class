"""Stereographic geometry helpers for structural geology."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, radians, sin
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class DipDipDirection:
    """Dip/dip-direction representation."""

    dip_deg: float
    dip_direction_deg: float
    trend_deg: float
    plunge_deg: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized")
    return vector / norm


def to_lower_hemisphere(vector: np.ndarray) -> np.ndarray:
    """Ensure the vector is on the lower hemisphere."""

    unit = _normalize(vector)
    if unit[2] > 0:
        unit = -unit
    return unit


def pole_from_dip(dip_deg: float, dip_direction_deg: float) -> np.ndarray:
    """Convert dip/dip-direction (deg) to a pole unit vector."""

    dip_rad = radians(dip_deg)
    dipdir_rad = radians(dip_direction_deg)
    plunge_deg = 90.0 - dip_deg
    plunge_rad = radians(plunge_deg)

    u_x = sin(dipdir_rad) * sin(plunge_rad)
    u_y = cos(dipdir_rad) * sin(plunge_rad)
    u_z = -cos(plunge_rad)
    pole = np.array([u_x, u_y, u_z], dtype=float)
    return to_lower_hemisphere(pole)


def normal_to_dip(normal: Iterable[float]) -> DipDipDirection:
    """Compute dip/dip-direction from a normal vector."""

    n = np.asarray(list(normal), dtype=float)
    if n.shape != (3,):
        raise ValueError("normal must be a 3-vector")
    u = to_lower_hemisphere(n)
    trend_deg = (atan2(u[0], u[1]) * 180.0 / np.pi + 360.0) % 360.0
    plunge_deg = asin(-u[2]) * 180.0 / np.pi
    dip_deg = 90.0 - plunge_deg
    return DipDipDirection(
        dip_deg=dip_deg,
        dip_direction_deg=trend_deg,
        trend_deg=trend_deg,
        plunge_deg=plunge_deg,
    )


def normals_to_poles(normals: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3)`` array of normals to lower-hemisphere poles."""

    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals array must have shape (N, 3)")
    poles = np.apply_along_axis(to_lower_hemisphere, 1, normals)
    return poles


def normals_to_dip_table(normals: np.ndarray) -> np.ndarray:
    """Return dip table with columns dip, dipdir, trend, plunge."""

    poles = normals_to_poles(normals)
    trend = (np.degrees(np.arctan2(poles[:, 0], poles[:, 1])) + 360.0) % 360.0
    plunge = np.degrees(np.arcsin(-poles[:, 2]))
    dip = 90.0 - plunge
    table = np.column_stack([dip, trend, trend, plunge])
    return table
