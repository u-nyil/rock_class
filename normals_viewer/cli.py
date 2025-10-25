"""Typer-based CLI for the normals viewer application."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from .cluster_orient import DBSCANOrientationParams, cluster_orientations
from .export import export_dip_table_csv, export_labels_csv, export_summary_json
from .io import load_point_cloud
from .normals import (
    NormalComputationParams,
    NormalGlyphParams,
    NormalOrientation,
    build_normal_glyphs,
    ensure_normals,
    orient_normals,
)
from .visualize import colorize_by_clusters, colorize_by_normals, open_viewer


class ColorMode(str, Enum):
    NONE = "none"
    NORMAL = "normal"
    CLUSTER = "cluster"

app = typer.Typer(name="normapp", help="Orientation analysis tools for point clouds")


def _parse_camera(camera: Optional[str]) -> Optional[tuple[float, float, float]]:
    if camera is None:
        return None
    parts = [p.strip() for p in camera.split(",") if p.strip()]
    if len(parts) != 3:
        raise typer.BadParameter("Camera must be formatted as 'cx,cy,cz'")
    return tuple(float(v) for v in parts)  # type: ignore[return-value]


@app.command()
def view(
    in_path: Path = typer.Option(..., "--in", help="Input point cloud"),
    show_normals: bool = typer.Option(False, help="Display normal glyphs"),
    norm_scale: float = typer.Option(0.05, help="Normal glyph scale"),
    every: int = typer.Option(10, help="Subsample normals"),
    orient: NormalOrientation = typer.Option(NormalOrientation.NONE, help="Normal orientation"),
    camera: Optional[str] = typer.Option(None, help="Camera location for viewpoint orientation"),
    k_neighbors: int = typer.Option(30, help="k neighbours for normal estimation"),
    radius: float = typer.Option(0.0, help="Radius for normal estimation (0 disables)"),
    color_mode: ColorMode = typer.Option(ColorMode.NORMAL, help="Colour mode"),
) -> None:
    """Visualise a point cloud with normals."""

    loaded = load_point_cloud(in_path)
    cloud = loaded.cloud

    need_normals = show_normals or color_mode in (ColorMode.NORMAL, ColorMode.CLUSTER) or not cloud.has_normals()
    if need_normals:
        params = NormalComputationParams(
            k_neighbors=k_neighbors,
            radius=radius or None,
        )
        ensure_normals(cloud, params)
        camera_location = _parse_camera(camera)
        orient_normals(
            cloud,
            orient,
            camera_location=camera_location,
            consistent_k=k_neighbors,
        )

    glyph = None
    if show_normals:
        glyph = build_normal_glyphs(
            cloud,
            NormalGlyphParams(scale=norm_scale, every=every),
        )

    if color_mode is ColorMode.NORMAL and cloud.has_normals():
        colorize_by_normals(cloud)
    elif color_mode is ColorMode.CLUSTER:
        typer.echo("Cluster colouring requested but no clustering performed in view command")

    open_viewer(cloud, glyphs=glyph)


@app.command()
def cluster(
    in_path: Path = typer.Option(..., "--in", help="Input point cloud"),
    eps_orient: float = typer.Option(12.0, help="DBSCAN orientation epsilon (deg)"),
    min_samples: int = typer.Option(0, help="DBSCAN min samples (0 auto)"),
    cone: float = typer.Option(15.0, help="Cone prune (deg, 0 disables)"),
    orient: NormalOrientation = typer.Option(NormalOrientation.NONE, help="Normal orientation"),
    camera: Optional[str] = typer.Option(None, help="Camera location"),
    k_neighbors: int = typer.Option(30, help="k neighbours for normal estimation"),
    radius: float = typer.Option(0.0, help="Radius for normal estimation (0 disables)"),
    show_normals: bool = typer.Option(False, help="Show normal glyphs in viewer"),
    norm_scale: float = typer.Option(0.05, help="Normal glyph scale"),
    every: int = typer.Option(10, help="Normal subsampling"),
    view_flag: bool = typer.Option(False, "--view", help="Open Open3D viewer"),
    out: Optional[Path] = typer.Option(None, "--out", help="CSV path for labels"),
    dip_out: Optional[Path] = typer.Option(None, help="CSV path for dip table"),
    summary_out: Optional[Path] = typer.Option(None, help="JSON summary output path"),
) -> None:
    """Run orientation clustering and optionally launch the viewer."""

    loaded = load_point_cloud(in_path)
    cloud = loaded.cloud

    params = NormalComputationParams(k_neighbors=k_neighbors, radius=radius or None)
    ensure_normals(cloud, params)
    orient_normals(
        cloud,
        orient,
        camera_location=_parse_camera(camera),
        consistent_k=k_neighbors,
    )

    normals = np.asarray(cloud.normals)
    db_params = DBSCANOrientationParams(
        eps_orient_deg=eps_orient,
        min_samples_orient=min_samples or None,
        cone_prune_deg=cone if cone > 0 else None,
    )
    summary = cluster_orientations(normals, db_params)

    if out is not None:
        indices = np.arange(len(summary.labels))
        export_labels_csv(indices, summary.labels, out)
    if dip_out is not None:
        export_dip_table_csv(normals, dip_out)
    if summary_out is not None:
        export_summary_json(summary, normals, summary_out)

    if view_flag:
        display_cloud = cloud.clone()
        glyph = None
        if show_normals:
            glyph = build_normal_glyphs(
                display_cloud,
                NormalGlyphParams(scale=norm_scale, every=every),
            )
        colorize_by_clusters(display_cloud, summary.labels, show_noise=True)
        open_viewer(display_cloud, glyphs=glyph)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    app()
