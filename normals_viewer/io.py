"""Input/Output utilities for point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:  # pragma: no cover - import guard for optional dependency
    import open3d as o3d
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for docs/tests
    raise RuntimeError(
        "open3d is required for normals_viewer.io; install it via requirements.txt"
    ) from exc


SUPPORTED_EXTENSIONS = {".ply", ".pcd", ".xyz", ".txt"}


@dataclass
class LoadedPointCloud:
    """Container bundling a point cloud and metadata."""

    cloud: o3d.geometry.PointCloud
    path: Path
    has_normals: bool


def _load_xyz_with_optional_normals(path: Path) -> o3d.geometry.PointCloud:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        msg = f"File {path} must contain at least three columns for x,y,z coordinates"
        raise ValueError(msg)
    points = data[:, :3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if data.shape[1] >= 6:
        normals = data[:, 3:6]
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def load_point_cloud(path: str | Path) -> LoadedPointCloud:
    """Load a point cloud from a file.

    Parameters
    ----------
    path:
        File path pointing to a supported point cloud format.

    Returns
    -------
    LoadedPointCloud
        The loaded cloud and metadata.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{p.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if p.suffix.lower() in {".xyz", ".txt"}:
        cloud = _load_xyz_with_optional_normals(p)
    else:
        cloud = o3d.io.read_point_cloud(str(p))
    has_normals = bool(cloud.has_normals())
    return LoadedPointCloud(cloud=cloud, path=p, has_normals=has_normals)


def save_point_cloud(cloud: o3d.geometry.PointCloud, path: str | Path) -> Path:
    """Save a point cloud to disk.

    Parameters
    ----------
    cloud:
        Point cloud geometry to save.
    path:
        Output file path.

    Returns
    -------
    Path
        The path where the file was written.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(p), cloud)
    if not success:
        raise IOError(f"Failed to save point cloud to {p}")
    return p
