"""Unit tests for orientation clustering."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
open3d = pytest.importorskip("open3d")

from normals_viewer.cluster_orient import DBSCANOrientationParams, cluster_orientations
from normals_viewer.stereogeom import normals_to_poles


def random_unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=3)
    vec /= np.linalg.norm(vec)
    return vec


def add_noise(normal: np.ndarray, degrees: float, rng: np.random.Generator) -> np.ndarray:
    axis = random_unit_vector(rng.integers(0, 10_000))
    angle = math.radians(degrees) * rng.uniform(-1.0, 1.0)
    axis = axis / np.linalg.norm(axis)
    normal = normal / np.linalg.norm(normal)
    cross = np.cross(axis, normal)
    normal = normal * math.cos(angle) + cross * math.sin(angle) + axis * np.dot(axis, normal) * (1 - math.cos(angle))
    return normal / np.linalg.norm(normal)


def test_two_plane_orientations_clustered() -> None:
    rng = np.random.default_rng(42)
    base_normals = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.0, math.sqrt(3) / 2]),
    ]
    samples = []
    for idx, base in enumerate(base_normals):
        for _ in range(200):
            samples.append(add_noise(base, 5.0, rng))
    normals = np.vstack(samples)
    params = DBSCANOrientationParams(eps_orient_deg=8.0, min_samples_orient=30, cone_prune_deg=10.0)
    summary = cluster_orientations(normals, params)
    assert summary.n_clusters == 2
    means = list(summary.cluster_means.values())
    for base in base_normals:
        angles = [
            math.degrees(math.acos(np.clip(np.abs(np.dot(mean_vec, base)), -1.0, 1.0)))
            for mean_vec in means
        ]
        assert min(angles) <= 3.0


def test_normals_to_poles_lower_hemisphere() -> None:
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
            [-0.4, 0.3, 0.5],
        ]
    )
    poles = normals_to_poles(normals)
    assert np.all(poles[:, 2] <= 1e-9)


def test_glyph_generation_line_count() -> None:
    point_cloud = open3d.geometry.PointCloud()
    points = np.array([[0.0, 0.0, float(i)] for i in range(25)])
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (25, 1))
    point_cloud.points = open3d.utility.Vector3dVector(points)
    point_cloud.normals = open3d.utility.Vector3dVector(normals)

    from normals_viewer.normals import NormalGlyphParams, build_normal_glyphs

    params = NormalGlyphParams(scale=0.1, every=4)
    glyph = build_normal_glyphs(point_cloud, params)
    line_count = np.asarray(glyph.lines).shape[0]
    expected = math.ceil(len(points) / params.every)
    assert line_count == expected
