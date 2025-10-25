"""PySide6 GUI for the normals viewer application."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency guard
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "PySide6 is required for normals_viewer.gui; install it via requirements.txt"
    ) from exc

from .cluster_orient import DBSCANOrientationParams, ClusterSummary, cluster_orientations
from .export import export_dip_table_csv, export_labels_csv, export_summary_json
from .io import LoadedPointCloud, load_point_cloud
from .normals import (
    NormalComputationParams,
    NormalGlyphParams,
    NormalOrientation,
    build_normal_glyphs,
    ensure_normals,
    orient_normals,
)
from .visualize import colorize_by_clusters, colorize_by_normals, open_viewer


@dataclass
class AppState:
    """Mutable GUI state."""

    loaded: Optional[LoadedPointCloud] = None
    cluster_summary: Optional[ClusterSummary] = None
    glyph_params: NormalGlyphParams = field(default_factory=NormalGlyphParams)
    show_noise: bool = True


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.setWindowTitle("Normals Viewer")
        self.resize(520, 680)
        self._build_ui()

    # region UI setup
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        self.status_label = QLabel("Load a point cloud to begin")
        layout.addWidget(self.status_label)

        layout.addWidget(self._build_load_group())
        layout.addWidget(self._build_normals_group())
        layout.addWidget(self._build_cluster_group())
        layout.addWidget(self._build_view_group())
        layout.addWidget(self._build_export_group())
        layout.addStretch()

        self.setCentralWidget(central)

    def _build_load_group(self) -> QWidget:
        group = QGroupBox("Data")
        box = QVBoxLayout(group)
        button = QPushButton("Load point cloud")
        button.clicked.connect(self.on_load_clicked)  # type: ignore[arg-type]
        box.addWidget(button)
        return group

    def _build_normals_group(self) -> QWidget:
        group = QGroupBox("Normals")
        form = QFormLayout(group)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 500)
        self.k_spin.setValue(30)
        form.addRow("k-neighbours", self.k_spin)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(4)
        self.radius_spin.setRange(0.0, 10.0)
        self.radius_spin.setSingleStep(0.01)
        form.addRow("Radius", self.radius_spin)

        self.orient_combo = QComboBox()
        self.orient_combo.addItems(
            [
                NormalOrientation.NONE.value,
                NormalOrientation.Z_UP.value,
                NormalOrientation.VIEWPOINT.value,
                NormalOrientation.CONSISTENT.value,
            ]
        )
        form.addRow("Orientation", self.orient_combo)

        self.camera_edit = QLineEdit("0,0,0")
        form.addRow("Viewpoint (x,y,z)", self.camera_edit)

        self.consistent_spin = QSpinBox()
        self.consistent_spin.setRange(1, 200)
        self.consistent_spin.setValue(30)
        form.addRow("Consistent k", self.consistent_spin)

        self.normals_button = QPushButton("Ensure normals")
        self.normals_button.clicked.connect(self.on_normals_clicked)  # type: ignore[arg-type]
        form.addRow(self.normals_button)
        return group

    def _build_cluster_group(self) -> QWidget:
        group = QGroupBox("Clustering")
        form = QFormLayout(group)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(1.0, 45.0)
        self.eps_spin.setValue(12.0)
        self.eps_spin.setSingleStep(0.5)
        form.addRow("eps orient (deg)", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(0, 1000000)
        self.min_samples_spin.setValue(0)
        form.addRow("min samples (0=auto)", self.min_samples_spin)

        self.cone_spin = QDoubleSpinBox()
        self.cone_spin.setRange(0.0, 90.0)
        self.cone_spin.setValue(15.0)
        self.cone_spin.setSingleStep(1.0)
        form.addRow("Cone prune (deg)", self.cone_spin)

        self.cluster_button = QPushButton("Run clustering")
        self.cluster_button.clicked.connect(self.on_cluster_clicked)  # type: ignore[arg-type]
        form.addRow(self.cluster_button)

        self.cluster_status = QLabel("Not clustered")
        form.addRow(self.cluster_status)
        return group

    def _build_view_group(self) -> QWidget:
        group = QGroupBox("Visualisation")
        form = QFormLayout(group)

        self.norm_scale_spin = QDoubleSpinBox()
        self.norm_scale_spin.setRange(0.001, 1.0)
        self.norm_scale_spin.setDecimals(4)
        self.norm_scale_spin.setValue(0.05)
        form.addRow("Normal scale", self.norm_scale_spin)

        self.norm_every_spin = QSpinBox()
        self.norm_every_spin.setRange(1, 1000)
        self.norm_every_spin.setValue(10)
        form.addRow("Show every", self.norm_every_spin)

        self.show_normals_check = QCheckBox("Show normal glyphs")
        self.show_normals_check.setChecked(True)
        form.addRow(self.show_normals_check)

        self.show_clusters_check = QCheckBox("Color by clusters")
        self.show_clusters_check.setChecked(True)
        form.addRow(self.show_clusters_check)

        self.show_noise_check = QCheckBox("Show noise")
        self.show_noise_check.setChecked(True)
        form.addRow(self.show_noise_check)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["cluster", "normal"])
        form.addRow("Colour mode", self.color_mode_combo)

        self.view_button = QPushButton("Open viewer")
        self.view_button.clicked.connect(self.on_view_clicked)  # type: ignore[arg-type]
        form.addRow(self.view_button)

        return group

    def _build_export_group(self) -> QWidget:
        group = QGroupBox("Export")
        layout = QVBoxLayout(group)
        self.export_labels_button = QPushButton("Export labels CSV")
        self.export_labels_button.clicked.connect(self.on_export_labels)  # type: ignore[arg-type]
        layout.addWidget(self.export_labels_button)

        self.export_dip_button = QPushButton("Export dip table CSV")
        self.export_dip_button.clicked.connect(self.on_export_dip)  # type: ignore[arg-type]
        layout.addWidget(self.export_dip_button)

        self.export_summary_button = QPushButton("Export summary JSON")
        self.export_summary_button.clicked.connect(self.on_export_summary)  # type: ignore[arg-type]
        layout.addWidget(self.export_summary_button)
        return group

    # endregion

    # region helpers
    def _require_loaded(self) -> LoadedPointCloud:
        if self.state.loaded is None:
            raise RuntimeError("No point cloud loaded")
        return self.state.loaded

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    # endregion

    # region slots
    def on_load_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open point cloud",
            str(Path.cwd()),
            "Point clouds (*.ply *.pcd *.xyz *.txt)",
        )
        if not file_path:
            return
        try:
            loaded = load_point_cloud(file_path)
        except Exception as exc:  # pragma: no cover - UI feedback
            self._show_error(str(exc))
            return
        self.state.loaded = loaded
        self.state.cluster_summary = None
        self.status_label.setText(
            f"Loaded {loaded.path.name} ({len(loaded.cloud.points)} points, normals: {loaded.has_normals})"
        )
        self.cluster_status.setText("Not clustered")

    def on_normals_clicked(self) -> None:
        try:
            loaded = self._require_loaded()
        except RuntimeError as exc:
            self._show_error(str(exc))
            return

        params = NormalComputationParams(
            k_neighbors=self.k_spin.value(),
            radius=self.radius_spin.value() or None,
        )
        try:
            ensure_normals(loaded.cloud, params)
            orientation = NormalOrientation(self.orient_combo.currentText())
            camera = None
            if orientation is NormalOrientation.VIEWPOINT:
                parts = [p.strip() for p in self.camera_edit.text().split(",") if p.strip()]
                if len(parts) != 3:
                    raise ValueError("Viewpoint must contain three comma-separated values")
                camera = tuple(float(v) for v in parts)
            orient_normals(
                loaded.cloud,
                orientation,
                camera_location=camera,
                consistent_k=self.consistent_spin.value(),
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            self._show_error(str(exc))
            return
        self.status_label.setText("Normals available and oriented")

    def on_cluster_clicked(self) -> None:
        try:
            loaded = self._require_loaded()
        except RuntimeError as exc:
            self._show_error(str(exc))
            return
        if not loaded.cloud.has_normals():
            self._show_error("Normals are required before clustering")
            return
        normals = np.asarray(loaded.cloud.normals)
        params = DBSCANOrientationParams(
            eps_orient_deg=self.eps_spin.value(),
            min_samples_orient=self.min_samples_spin.value() or None,
            cone_prune_deg=self.cone_spin.value(),
        )
        try:
            summary = cluster_orientations(normals, params)
        except Exception as exc:  # pragma: no cover - UI feedback
            self._show_error(str(exc))
            return
        self.state.cluster_summary = summary
        self.cluster_status.setText(
            f"Clusters: {summary.n_clusters} (noise={summary.counts.get(-1, 0)})"
        )

    def on_view_clicked(self) -> None:
        try:
            loaded = self._require_loaded()
        except RuntimeError as exc:
            self._show_error(str(exc))
            return
        cloud = loaded.cloud.clone()
        glyph = None
        if self.show_normals_check.isChecked() and cloud.has_normals():
            glyph_params = NormalGlyphParams(
                scale=self.norm_scale_spin.value(),
                every=self.norm_every_spin.value(),
            )
            try:
                glyph = build_normal_glyphs(cloud, glyph_params)
            except Exception as exc:  # pragma: no cover - UI feedback
                self._show_error(str(exc))
                return
        color_mode = self.color_mode_combo.currentText()
        if color_mode == "cluster" and self.state.cluster_summary is not None:
            show_noise = self.show_noise_check.isChecked()
            colorize_by_clusters(cloud, self.state.cluster_summary.labels, show_noise)
        elif color_mode == "normal" and cloud.has_normals():
            colorize_by_normals(cloud)
        open_viewer(cloud, glyphs=glyph)

    def on_export_labels(self) -> None:
        if self.state.cluster_summary is None or self.state.loaded is None:
            self._show_error("Run clustering before exporting labels")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save labels CSV", str(Path.cwd()), "CSV files (*.csv)"
        )
        if not file_path:
            return
        indices = np.arange(len(self.state.cluster_summary.labels))
        export_labels_csv(indices, self.state.cluster_summary.labels, file_path)

    def on_export_dip(self) -> None:
        if self.state.loaded is None or not self.state.loaded.cloud.has_normals():
            self._show_error("Normals required for dip export")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save dip table CSV", str(Path.cwd()), "CSV files (*.csv)"
        )
        if not file_path:
            return
        normals = np.asarray(self.state.loaded.cloud.normals)
        export_dip_table_csv(normals, file_path)

    def on_export_summary(self) -> None:
        if self.state.cluster_summary is None or self.state.loaded is None:
            self._show_error("Run clustering before exporting summary")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save summary JSON", str(Path.cwd()), "JSON files (*.json)"
        )
        if not file_path:
            return
        normals = np.asarray(self.state.loaded.cloud.normals)
        export_summary_json(self.state.cluster_summary, normals, file_path)

    # endregion


def run_gui() -> None:
    """Launch the PySide6 application."""

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    window = MainWindow()
    window.show()
    if owns_app:
        sys.exit(app.exec())
