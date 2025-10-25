"""Helpers for rendering point clouds with Open3D."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - import guard for optional dependency
    import open3d as o3d
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for docs/tests
    raise RuntimeError(
        "open3d is required for normals_viewer.visualize; install it via requirements.txt"
    ) from exc

NOISE_COLOR = np.array([0.8, 0.8, 0.8], dtype=float)
_CLUSTER_PALETTE = np.array(
    [
        [0.894, 0.102, 0.110],
        [0.215, 0.494, 0.721],
        [0.302, 0.686, 0.290],
        [0.596, 0.306, 0.639],
        [1.000, 0.498, 0.000],
        [1.000, 1.000, 0.200],
        [0.651, 0.337, 0.157],
        [0.969, 0.506, 0.749],
    ],
    dtype=float,
)


def _cluster_color(cluster_id: int) -> np.ndarray:
    palette_idx = cluster_id % len(_CLUSTER_PALETTE)
    return _CLUSTER_PALETTE[palette_idx]


def labels_to_colors(labels: np.ndarray, show_noise: bool = True) -> np.ndarray:
    """Map cluster labels to RGB colors."""

    labels = np.asarray(labels, dtype=int)
    colors = np.empty((len(labels), 3), dtype=float)
    for idx, label in enumerate(labels):
        if label == -1:
            colors[idx] = NOISE_COLOR if show_noise else np.array([0.0, 0.0, 0.0])
        else:
            colors[idx] = _cluster_color(label)
    return colors


def apply_colors(cloud: o3d.geometry.PointCloud, colors: np.ndarray) -> None:
    """Assign colors to an Open3D point cloud."""

    if colors.shape[0] != np.asarray(cloud.points).shape[0]:
        raise ValueError("Color array must match number of points")
    cloud.colors = o3d.utility.Vector3dVector(colors)


def colorize_by_clusters(
    cloud: o3d.geometry.PointCloud, labels: np.ndarray, show_noise: bool = True
) -> None:
    colors = labels_to_colors(labels, show_noise=show_noise)
    apply_colors(cloud, colors)


def colorize_by_normals(cloud: o3d.geometry.PointCloud) -> None:
    """Color points by normalized normal vectors."""

    if not cloud.has_normals():
        raise ValueError("Point cloud has no normals for colorization")
    normals = np.asarray(cloud.normals)
    colors = 0.5 * (normals + 1.0)
    apply_colors(cloud, colors)


def open_viewer(
    cloud: o3d.geometry.PointCloud,
    *,
    glyphs: Optional[o3d.geometry.LineSet] = None,
    window_name: str = "Normals Viewer",
) -> None:
    """Open an Open3D viewer window with the supplied geometries."""

    geometries: list[o3d.geometry.Geometry] = [cloud]
    if glyphs is not None:
        geometries.append(glyphs)
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
