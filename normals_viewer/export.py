"""Export utilities for clustering results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .cluster_orient import ClusterSummary
from .stereogeom import normals_to_dip_table, normal_to_dip


def export_labels_csv(indices: Iterable[int], labels: np.ndarray, path: str | Path) -> Path:
    """Export cluster labels to CSV."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"idx": list(indices), "family_id": labels})
    df.to_csv(p, index=False)
    return p


def export_dip_table_csv(normals: np.ndarray, path: str | Path) -> Path:
    """Export dip/dip-direction table to CSV."""

    table = normals_to_dip_table(normals)
    df = pd.DataFrame(
        table,
        columns=["dip_deg", "dipdir_deg", "trend_deg", "plunge_deg"],
    )
    df.insert(0, "idx", np.arange(len(normals)))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def export_summary_json(
    summary: ClusterSummary,
    normals: np.ndarray,
    path: str | Path,
) -> Path:
    """Export clustering summary to JSON."""

    stats: Dict[str, object] = {
        "n_points": int(len(normals)),
        "eps_orient_deg": summary.params.eps_orient_deg,
        "min_samples_orient": summary.params.min_samples_orient,
        "cone_prune_deg": summary.params.cone_prune_deg,
        "counts": {str(k): int(v) for k, v in summary.counts.items()},
        "cluster_means": {},
    }
    for cid, vec in summary.cluster_means.items():
        dip = normal_to_dip(vec)
        stats["cluster_means"][str(cid)] = {
            "trend_deg": dip.trend_deg,
            "plunge_deg": dip.plunge_deg,
            "dip_deg": dip.dip_deg,
            "dip_direction_deg": dip.dip_direction_deg,
        }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2))
    return p
