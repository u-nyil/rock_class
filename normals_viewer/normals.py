"""Normal handling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np

try:  # pragma: no cover - import guard for optional dependency
    import open3d as o3d
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for docs/tests
    raise RuntimeError(
        "open3d is required for normals_viewer.normals; install it via requirements.txt"
    ) from exc


class NormalOrientation(str, Enum):
    """Orientation strategies for surface normals."""

    NONE = "none"
    Z_UP = "zup"
    VIEWPOINT = "viewpoint"
    CONSISTENT = "consistent"


@dataclass(slots=True)
class NormalComputationParams:
    """Parameters controlling normal estimation."""

    k_neighbors: int | None = 30
    radius: float | None = None

    def to_search_param(self) -> o3d.geometry.KDTreeSearchParam:
        if self.radius is not None:
            max_nn = self.k_neighbors or 30
            return o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=max_nn)
        if self.k_neighbors is None:
            raise ValueError("Either k_neighbors or radius must be provided for normal estimation")
        return o3d.geometry.KDTreeSearchParamKNN(self.k_neighbors)


@dataclass(slots=True)
class NormalGlyphParams:
    """Parameters for glyph construction."""

    scale: float = 0.05
    every: int = 10

    def validate(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.every <= 0:
            raise ValueError("every must be positive")


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def ensure_normals(cloud: o3d.geometry.PointCloud, params: NormalComputationParams) -> None:
    """Ensure that a point cloud contains normalized normals."""

    if not cloud.has_points():
        raise ValueError("Point cloud is empty")
    if not cloud.has_normals():
        cloud.estimate_normals(search_param=params.to_search_param())
    normals = np.asarray(cloud.normals)
    normals = _normalize_vectors(normals)
    cloud.normals = o3d.utility.Vector3dVector(normals)


def orient_normals(
    cloud: o3d.geometry.PointCloud,
    strategy: NormalOrientation,
    *,
    camera_location: Optional[Iterable[float]] = None,
    consistent_k: int = 30,
) -> None:
    """Orient normals according to the chosen strategy."""

    if not cloud.has_normals():
        raise ValueError("Point cloud has no normals to orient")

    if strategy is NormalOrientation.NONE:
        return
    if strategy is NormalOrientation.Z_UP:
        normals = np.asarray(cloud.normals)
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] *= -1.0
        cloud.normals = o3d.utility.Vector3dVector(normals)
        return
    if strategy is NormalOrientation.VIEWPOINT:
        if camera_location is None:
            raise ValueError("camera_location is required for viewpoint orientation")
        cloud.orient_normals_towards_camera_location(camera_location)
        return
    if strategy is NormalOrientation.CONSISTENT:
        if consistent_k <= 0:
            raise ValueError("consistent_k must be > 0")
        cloud.orient_normals_consistent_tangent_plane(consistent_k)
        return
    raise ValueError(f"Unsupported orientation strategy {strategy}")


def normals_to_numpy(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """Return a copy of the normals array as ``(N, 3)`` numpy array."""

    if not cloud.has_normals():
        raise ValueError("Point cloud has no normals")
    normals = np.asarray(cloud.normals)
    return normals.copy()


def build_normal_glyphs(cloud: o3d.geometry.PointCloud, params: NormalGlyphParams) -> o3d.geometry.LineSet:
    """Create a LineSet visualizing surface normals."""

    params.validate()
    if not cloud.has_normals():
        raise ValueError("Point cloud requires normals to create glyphs")

    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    n_points = len(points)
    if n_points == 0:
        raise ValueError("Point cloud has no points")

    indices = np.arange(0, n_points, params.every)
    selected_points = points[indices]
    selected_normals = normals[indices]
    endpoints = selected_points + selected_normals * params.scale

    line_points = np.vstack([selected_points, endpoints])
    n_lines = len(indices)
    lines = [[i, i + n_lines] for i in range(n_lines)]

    glyph = o3d.geometry.LineSet()
    glyph.points = o3d.utility.Vector3dVector(line_points)
    glyph.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array([[0.1, 0.6, 0.9]]), (n_lines, 1))
    glyph.colors = o3d.utility.Vector3dVector(colors)
    return glyph
