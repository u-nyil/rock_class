"""Orientation-based clustering using DBSCAN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from .stereogeom import normals_to_poles


@dataclass(slots=True)
class DBSCANOrientationParams:
    """Parameters for orientation clustering."""

    eps_orient_deg: float = 12.0
    min_samples_orient: Optional[int] = None
    cone_prune_deg: Optional[float] = 15.0

    def eps_radians(self) -> float:
        if self.eps_orient_deg <= 0:
            raise ValueError("eps_orient_deg must be positive")
        return np.deg2rad(self.eps_orient_deg)

    def resolved_min_samples(self, n_points: int) -> int:
        if self.min_samples_orient is not None and self.min_samples_orient > 0:
            return self.min_samples_orient
        return max(20, max(1, int(round(0.001 * n_points))))


def _angular_distance(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.clip(np.abs(np.dot(u, v)), -1.0, 1.0))
    return float(np.arccos(dot))


@dataclass(slots=True)
class ClusterSummary:
    """Summary of DBSCAN orientation clustering."""

    labels: np.ndarray
    cluster_means: Dict[int, np.ndarray]
    params: DBSCANOrientationParams
    counts: Dict[int, int]

    @property
    def n_clusters(self) -> int:
        return len([cid for cid in self.cluster_means if cid != -1])


def compute_cluster_means(poles: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    means: Dict[int, np.ndarray] = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        members = poles[labels == label]
        if len(members) == 0:
            continue
        mean_vec = members.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm == 0:
            continue
        means[label] = mean_vec / norm
    return means


def _apply_cone_prune(
    poles: np.ndarray, labels: np.ndarray, means: Dict[int, np.ndarray], cone_deg: float
) -> np.ndarray:
    cone_rad = np.deg2rad(cone_deg)
    new_labels = labels.copy()
    for cluster_id, mean_vec in means.items():
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        angles = np.arccos(np.clip(np.abs(poles[mask] @ mean_vec), -1.0, 1.0))
        outside = angles > cone_rad
        new_labels[mask] = np.where(outside, -1, cluster_id)
    return new_labels


def cluster_orientations(normals: np.ndarray, params: DBSCANOrientationParams) -> ClusterSummary:
    """Cluster normals based on orientation similarity."""

    poles = normals_to_poles(normals)
    n_points = len(poles)
    if n_points == 0:
        raise ValueError("No normals provided for clustering")

    eps = params.eps_radians()
    min_samples = params.resolved_min_samples(n_points)

    def metric(u: np.ndarray, v: np.ndarray) -> float:
        return _angular_distance(u, v)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(poles)

    means = compute_cluster_means(poles, labels)
    counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}

    if params.cone_prune_deg is not None and params.cone_prune_deg > 0:
        labels = _apply_cone_prune(poles, labels, means, params.cone_prune_deg)
        means = compute_cluster_means(poles, labels)
        counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}

    return ClusterSummary(labels=labels, cluster_means=means, params=params, counts=counts)
