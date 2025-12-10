#!/usr/bin/env python3
"""
Rock Cluster (MeanShift or KMeans + 2D-DBSCAN + RANSAC)

- Method "meanshift":
    * MeanShift on normals (unit sphere, fixed angular bandwidth) -> sets
    * 2D DBSCAN in local set coordinates -> patches
    * RANSAC plane per patch (threshold = mul * local spacing)

- Method "kmeans":
    * KMeans on normals -> sets
    * 2D DBSCAN in local set coordinates -> patches
    * RANSAC plane per patch (same threshold logic)

Assumptions (both methods):
- The input PLY MUST already have normals
  (e.g. computed in CloudCompare or another tool).
"""

import os
import sys
import math
import time
import csv
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, MeanShift, KMeans
from sklearn.neighbors import NearestNeighbors

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ---------------------------------------------------------------------------#
# Dataclasses for configuration

@dataclass
class SetMeanShiftParams:
    # Fixed angular bandwidth on the unit sphere (degrees)
    bandwidth_deg: float = 12.5
    bin_seeding: bool = True


@dataclass
class PatchParams:
    # Shared patch / RANSAC params for MeanShift pipeline
    eps_scale: float = 3.5          # multiplies median 2D step in the projected set
    min_samples: int = 30           # DBSCAN min_samples (MinPts)
    min_points_plane: int = 30      # minimum points to accept a plane
    ransac_n: int = 3
    ransac_iters: int = 2000
    ransac_dist_mul: float = 1.5    # threshold multiplier * local spacing
    max_plane_rms: Optional[float] = None  # max RMS (m); None = no filter


@dataclass
class KMeansPatchParams:
    # KMeans + patch/RANSAC params
    k: int = 5
    eps_scale: float = 3.5          # multiplies median 2D step in the projected set
    min_samples: int = 15           # DBSCAN min_samples (MinPts)
    min_points_plane: int = 30      # minimum points to accept a plane
    ransac_n: int = 3
    ransac_iters: int = 2000
    ransac_dist_mul: float = 1.5    # threshold multiplier * local spacing
    max_plane_rms: Optional[float] = None  # max RMS (m); None = no filter


@dataclass
class GeneralOptions:
    method_mode: str = "meanshift"      # "meanshift" or "kmeans"
    output_root: Optional[str] = None
    enable_viewer: bool = True
    enable_stereonet: bool = True
    viewer_backend: str = "pyvista"     # "pyvista" or "open3d"
    viewer_point_size: int = 3
    viewer_max_points: int = 2_000_000
    stereonet_max_points: int = 20000
    export_ascii_ply: bool = False
    log_reference_name: str = "Rock Cluster"


@dataclass
class RunConfig:
    input_path: str
    general: GeneralOptions
    set_ms: SetMeanShiftParams
    kmeans: KMeansPatchParams
    patch: PatchParams


@dataclass
class PlaneRecord:
    plane_id: int
    set_id: int
    cluster_id: int
    n_points: int
    normal: np.ndarray
    d: float
    dip: float
    dipdir: float
    rms: float
    method_tag: str
    threshold: float


# ---------------------------------------------------------------------------#
# Helpers

def unit_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n


def ensure_lower_hemisphere(N: np.ndarray) -> np.ndarray:
    N = unit_rows(N.copy())
    N[N[:, 2] > 0] *= -1.0
    return N


def dip_and_dipdir_from_normal(n: np.ndarray) -> Tuple[float, float]:
    n = np.asarray(n, float)
    n /= (np.linalg.norm(n) + 1e-12)
    if n[2] > 0:
        n *= -1.0
    dip = math.degrees(math.acos(np.clip(abs(n[2]), 0.0, 1.0)))
    up = np.array([0.0, 0.0, 1.0])
    v = -up - ((-up) @ n) * n
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return float(dip), float("nan")
    v /= nv
    dipdir = (math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0
    return float(dip), float(dipdir)


def orthobasis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(n, float)
    n /= (np.linalg.norm(n) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)
    return u, v, n


def median_step_2d(Q: np.ndarray) -> float:
    """Median nearest-neighbor step in 2D; fall back to bounding box size."""
    if Q.shape[0] < 2:
        return float(np.linalg.norm(Q.ptp(axis=0)))
    nn = NearestNeighbors(n_neighbors=2).fit(Q)
    dists, _ = nn.kneighbors(Q)
    return float(np.median(dists[:, 1]))


def convex_hull_2d(Q: np.ndarray) -> Optional[np.ndarray]:
    if len(Q) < 3:
        return None
    Q = np.asarray(Q, float)
    Q = Q[np.lexsort((Q[:, 0], Q[:, 1]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in Q:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in Q[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], float)


def polygon_area_perimeter(poly: np.ndarray) -> Tuple[float, float]:
    if poly is None or len(poly) < 3:
        return 0.0, 0.0
    area = 0.0
    peri = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
        peri += math.hypot(x2 - x1, y2 - y1)
    return abs(area) * 0.5, peri


def plane_fit(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Least-squares plane fit; returns (normal, d, rms)."""
    if len(points) < 3:
        raise ValueError("Need >=3 points for plane fit")
    C = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - C, full_matrices=False)
    n = Vt[-1]
    n /= (np.linalg.norm(n) + 1e-12)
    if n[2] > 0:
        n *= -1.0
    d = float(-np.dot(n, C))
    rms = float(np.std((points - C) @ n, ddof=1)) if len(points) >= 4 else 0.0
    return n, d, rms


def color_palette(k: int) -> np.ndarray:
    base = np.array([
        [0.90, 0.10, 0.10],
        [0.10, 0.70, 0.10],
        [0.10, 0.40, 0.95],
        [0.90, 0.70, 0.10],
        [0.70, 0.10, 0.85],
        [0.10, 0.80, 0.80],
        [0.75, 0.50, 0.30],
        [0.50, 0.50, 0.50],
        [0.20, 0.20, 0.20],
        [0.95, 0.45, 0.10],
    ])
    if k <= len(base):
        return base[:k]
    rng = np.random.default_rng(42)
    extra = rng.random((k - len(base), 3)) * 0.6 + 0.2
    return np.vstack([base, extra])


# ---------------------------------------------------------------------------#
# Logger with Rock Cluster format

class DSLogger:
    HEADER = [
        "Welcome to Rock Cluster 1.0 (MeanShift or KMeans + RANSAC)",
        "Method references: MeanShift or KMeans set detection + 2D-DBSCAN patching + RANSAC plane extraction.",
    ]

    def __init__(self):
        self.entries: List[str] = []
        self.entries.extend(self.HEADER)

    def log(self, msg: str) -> None:
        line = f"{dt.datetime.now():%d-%b-%Y %H:%M:%S} - {msg}"
        print(line)
        self.entries.append(line)


# ---------------------------------------------------------------------------#
# GUI dialog

class ParameterDialog:
    def __init__(self,
                 default_general: GeneralOptions,
                 default_set: SetMeanShiftParams,
                 default_patch: PatchParams,
                 default_kmeans: KMeansPatchParams):
        self.default_general = default_general
        self.default_set = default_set
        self.default_patch = default_patch
        self.default_kmeans = default_kmeans
        self.result: Optional[RunConfig] = None
        self.root: Optional[tk.Tk] = None

    def show(self) -> Optional[RunConfig]:
        self.root = tk.Tk()
        self.root.title("Rock Cluster (MeanShift / KMeans)")
        self.root.geometry("860x560")
        self.root.resizable(True, True)

        nb = ttk.Notebook(self.root)
        general_frame = ttk.Frame(nb, padding=10)
        set_frame = ttk.Frame(nb, padding=10)
        patch_frame = ttk.Frame(nb, padding=10)
        kmeans_frame = ttk.Frame(nb, padding=10)
        nb.add(general_frame, text="General")
        nb.add(set_frame, text="Set MeanShift")
        nb.add(patch_frame, text="Patch DBSCAN + RANSAC")
        nb.add(kmeans_frame, text="KMeans + Patch RANSAC")
        nb.pack(fill="both", expand=True)

        # General tab fields
        self.input_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=self.default_general.output_root or "")
        self.method_var = tk.StringVar(value=self.default_general.method_mode)
        self.viewer_backend_var = tk.StringVar(value=self.default_general.viewer_backend)
        self.viewer_point_size_var = tk.StringVar(value=str(self.default_general.viewer_point_size))
        self.viewer_max_points_var = tk.StringVar(value=str(self.default_general.viewer_max_points))
        self.stereonet_max_points_var = tk.StringVar(value=str(self.default_general.stereonet_max_points))
        self.enable_viewer_var = tk.BooleanVar(value=self.default_general.enable_viewer)
        self.enable_stereonet_var = tk.BooleanVar(value=self.default_general.enable_stereonet)
        self.export_ascii_var = tk.BooleanVar(value=self.default_general.export_ascii_ply)
        self.log_name_var = tk.StringVar(value=self.default_general.log_reference_name)

        row = 0
        ttk.Label(general_frame, text="Input .ply (with normals):").grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(general_frame, textvariable=self.input_var, width=70)
        entry.grid(row=row, column=1, sticky="we")
        ttk.Button(general_frame, text="Browse", command=self._browse_input).grid(row=row, column=2, padx=8)
        row += 1

        ttk.Label(general_frame, text="Output base directory (default: alongside input)").grid(row=row, column=0, sticky="w")
        out_entry = ttk.Entry(general_frame, textvariable=self.output_dir_var, width=70)
        out_entry.grid(row=row, column=1, sticky="we")
        ttk.Button(general_frame, text="Browse", command=self._browse_output).grid(row=row, column=2, padx=8)
        row += 1

        ttk.Label(general_frame, text="Method").grid(row=row, column=0, sticky="w")
        ttk.Combobox(general_frame, textvariable=self.method_var,
                     values=["meanshift", "kmeans"], width=18, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(general_frame, text="Viewer backend").grid(row=row, column=0, sticky="w")
        ttk.Combobox(general_frame, textvariable=self.viewer_backend_var,
                     values=["pyvista", "open3d"], width=18, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(general_frame, text="Viewer point size").grid(row=row, column=0, sticky="w")
        ttk.Entry(general_frame, textvariable=self.viewer_point_size_var, width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(general_frame, text="Viewer max points (subsample)").grid(row=row, column=0, sticky="w")
        ttk.Entry(general_frame, textvariable=self.viewer_max_points_var, width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Checkbutton(general_frame, text="Launch viewer after run",
                        variable=self.enable_viewer_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(general_frame, text="Show stereonet after run",
                        variable=self.enable_stereonet_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(general_frame, text="Stereonet max points").grid(row=row, column=0, sticky="w")
        ttk.Entry(general_frame, textvariable=self.stereonet_max_points_var, width=20).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Checkbutton(general_frame, text="Write colored PLY as ASCII",
                        variable=self.export_ascii_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(general_frame, text="Log reference name").grid(row=row, column=0, sticky="w")
        ttk.Entry(general_frame, textvariable=self.log_name_var, width=40).grid(row=row, column=1, sticky="w")
        row += 1
        general_frame.columnconfigure(1, weight=1)

        # Set MeanShift tab
        ttk.Label(set_frame, text="Bandwidth (deg)").grid(row=0, column=0, sticky="w")
        self.set_bw_var = tk.StringVar(value=str(self.default_set.bandwidth_deg))
        ttk.Entry(set_frame, textvariable=self.set_bw_var, width=12).grid(row=0, column=1, sticky="w")

        self.bin_seed_var = tk.BooleanVar(value=self.default_set.bin_seeding)
        ttk.Checkbutton(set_frame, text="Enable bin seeding", variable=self.bin_seed_var).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )

        # Patch tab (MeanShift pipeline)
        self.patch_vars: Dict[str, tk.StringVar] = {}

        def add_p_field(label, key, value):
            r = len(self.patch_vars)
            ttk.Label(patch_frame, text=label).grid(row=r, column=0, sticky="w")
            var = tk.StringVar(value=str(value))
            self.patch_vars[key] = var
            ttk.Entry(patch_frame, textvariable=var, width=14).grid(row=r, column=1, sticky="w")

        add_p_field("DBSCAN eps scale", "eps_scale", self.default_patch.eps_scale)
        add_p_field("DBSCAN min_samples", "min_samples", self.default_patch.min_samples)
        add_p_field("Min points per plane", "min_points_plane", self.default_patch.min_points_plane)
        add_p_field("RANSAC minimal sample", "ransac_n", self.default_patch.ransac_n)
        add_p_field("RANSAC iterations", "ransac_iters", self.default_patch.ransac_iters)
        add_p_field("RANSAC threshold multiplier", "ransac_dist_mul", self.default_patch.ransac_dist_mul)
        add_p_field("Max plane RMS (m, blank=off)", "max_plane_rms",
                    "" if self.default_patch.max_plane_rms is None else self.default_patch.max_plane_rms)

        # KMeans tab
        self.k_vars: Dict[str, tk.StringVar] = {}

        def add_k_field(label, key, value):
            r = len(self.k_vars)
            ttk.Label(kmeans_frame, text=label).grid(row=r, column=0, sticky="w")
            var = tk.StringVar(value=str(value))
            self.k_vars[key] = var
            ttk.Entry(kmeans_frame, textvariable=var, width=16).grid(row=r, column=1, sticky="w")

        add_k_field("K (sets)", "k", self.default_kmeans.k)
        add_k_field("DBSCAN min_samples", "min_samples", self.default_kmeans.min_samples)
        add_k_field("DBSCAN eps scale", "eps_scale", self.default_kmeans.eps_scale)
        add_k_field("Min points per plane", "min_points_plane", self.default_kmeans.min_points_plane)
        add_k_field("RANSAC minimal sample", "ransac_n", self.default_kmeans.ransac_n)
        add_k_field("RANSAC iterations", "ransac_iters", self.default_kmeans.ransac_iters)
        add_k_field("RANSAC threshold multiplier", "ransac_dist_mul", self.default_kmeans.ransac_dist_mul)
        add_k_field("Max plane RMS (m, blank=off)", "max_plane_rms",
                    "" if self.default_kmeans.max_plane_rms is None else self.default_kmeans.max_plane_rms)

        # Buttons
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Run", command=self._confirm).pack(side="right")

        self.root.mainloop()
        return self.result

    def _browse_input(self):
        path = filedialog.askopenfilename(title="Select .ply point cloud with normals",
                                          filetypes=[("PLY", "*.ply"), ("All files", "*.*")])
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)

    def _cancel(self):
        self.result = None
        self.root.destroy()

    def _confirm(self):
        path = self.input_var.get().strip()
        if not path:
            messagebox.showerror("Missing input", "Please select a .ply file.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Invalid path", "Input file not found.")
            return

        try:
            method_mode = (self.method_var.get() or "meanshift").strip().lower()
            if method_mode not in ("meanshift", "kmeans"):
                method_mode = "meanshift"

            general = GeneralOptions(
                method_mode=method_mode,
                output_root=self.output_dir_var.get().strip() or None,
                enable_viewer=bool(self.enable_viewer_var.get()),
                enable_stereonet=bool(self.enable_stereonet_var.get()),
                viewer_backend=self.viewer_backend_var.get(),
                viewer_point_size=int(self.viewer_point_size_var.get()),
                viewer_max_points=int(self.viewer_max_points_var.get()),
                stereonet_max_points=int(self.stereonet_max_points_var.get()),
                export_ascii_ply=bool(self.export_ascii_var.get()),
                log_reference_name=self.log_name_var.get().strip() or "Rock Cluster",
            )

            set_ms = SetMeanShiftParams(
                bandwidth_deg=float(self.set_bw_var.get()),
                bin_seeding=bool(self.bin_seed_var.get()),
            )

            max_plane_rms_txt = str(self.patch_vars["max_plane_rms"].get()).strip()
            max_plane_rms = float(max_plane_rms_txt) if max_plane_rms_txt not in ("", "None") else None

            patch = PatchParams(
                eps_scale=float(self.patch_vars["eps_scale"].get()),
                min_samples=int(self.patch_vars["min_samples"].get()),
                min_points_plane=int(self.patch_vars["min_points_plane"].get()),
                ransac_n=int(self.patch_vars["ransac_n"].get()),
                ransac_iters=int(self.patch_vars["ransac_iters"].get()),
                ransac_dist_mul=float(self.patch_vars["ransac_dist_mul"].get()),
                max_plane_rms=max_plane_rms,
            )

            max_plane_rms_k_txt = str(self.k_vars["max_plane_rms"].get()).strip()
            max_plane_rms_k = float(max_plane_rms_k_txt) if max_plane_rms_k_txt not in ("", "None") else None

            kmeans = KMeansPatchParams(
                k=int(self.k_vars["k"].get()),
                eps_scale=float(self.k_vars["eps_scale"].get()),
                min_samples=int(self.k_vars["min_samples"].get()),
                min_points_plane=int(self.k_vars["min_points_plane"].get()),
                ransac_n=int(self.k_vars["ransac_n"].get()),
                ransac_iters=int(self.k_vars["ransac_iters"].get()),
                ransac_dist_mul=float(self.k_vars["ransac_dist_mul"].get()),
                max_plane_rms=max_plane_rms_k,
            )
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.result = RunConfig(path, general, set_ms, kmeans, patch)
        self.root.destroy()


# ---------------------------------------------------------------------------#
# Set/plane statistics + Rock Cluster writers

def compute_set_stats(points: np.ndarray, normals: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
    mask = labels >= 0
    assigned = labels[mask]
    nrm = normals[mask]
    stats = []
    if assigned.size == 0:
        return stats
    total = assigned.size
    unique = sorted(int(u) for u in np.unique(assigned))
    counts = {u: int((assigned == u).sum()) for u in unique}
    max_cnt = max(counts.values())
    for sid in unique:
        idx = (assigned == sid)
        nn = ensure_lower_hemisphere(nrm[idx])
        m = unit_rows(nn).mean(axis=0)
        m /= (np.linalg.norm(m) + 1e-12)
        dip, dipdir = dip_and_dipdir_from_normal(m)
        stats.append({
            "set_id": sid,
            "count": counts[sid],
            "dip": dip,
            "dipdir": dipdir,
            "density": counts[sid] / max(1, max_cnt),
            "percent": 100.0 * counts[sid] / total,
            "normal": m,
        })
    return stats


def write_rock_cluster_report(base_name: str,
                              out_dir: str,
                              input_path: str,
                              stats: List[Dict[str, Any]],
                              planes: List[PlaneRecord],
                              params_summary: Dict[str, Any],
                              total_points: int,
                              assigned_points: int) -> str:
    report_path = os.path.join(out_dir, f"{base_name} - report.txt")
    with open(report_path, "w", newline="") as f:
        f.write(f"Rock Cluster, {dt.datetime.now():%d-%b-%Y}. Report of the used parameters.\n")
        f.write(f"File: {input_path}\n\n")
        f.write("Used parameters:\n")
        for section, values in params_summary.items():
            f.write(f"- {section}:\n")
            if isinstance(values, dict):
                for k, v in values.items():
                    f.write(f"  * {k}: {v}\n")
            else:
                f.write(f"  * {values}\n")
        f.write("\nResults\n")
        f.write(f"- Number of points of the original point cloud: {total_points}\n")
        f.write(f"- Number of points of the classified point cloud: {assigned_points}\n")
        f.write(f"- Number of unassigned points: {total_points - assigned_points}\n")
        f.write(f"- Number of discontinuity sets: {len(stats)}\n")
        f.write("- Extracted discontinuity sets:\n")
        f.write("\t\tDip dir\t\tDip\t\tDensity\t\t%\n")
        for stat in stats:
            f.write(f"\t\t{stat['dipdir']:7.2f}\t\t{stat['dip']:7.2f}\t\t"
                    f"{stat['density']:7.4f}\t\t{stat['percent']:6.2f}\n")
        f.write("\t\tWhere % is the number of assigned points to a DS over the total number of points\n\n")
        f.write(" - Extracted clusters and its corresponding plane equation (Ax+By+Cz+D=0)\n")
        f.write("\t    DS\t\tcluster\t\tn_pts\t\t  dip_dir\t\tdip\t\tA\t\t  B\t\t  C\t\t  D\t\ttsigma\n")
        for plane in planes:
            f.write(f"\t{plane.set_id:6d}\t{plane.cluster_id:8d}\t{plane.n_points:10d}\t"
                    f"{plane.dipdir:9.0f}\t{plane.dip:9.0f}\t"
                    f"{plane.normal[0]:+0.4f}\t{plane.normal[1]:+0.4f}\t"
                    f"{plane.normal[2]:+0.4f}\t{plane.d:+0.4f}\t{plane.rms:6.4f}\n")
    return report_path


def write_rock_cluster_log(base_name: str, out_dir: str, logger: DSLogger) -> str:
    log_path = os.path.join(out_dir, f"{base_name} - log.txt")
    with open(log_path, "w", newline="") as f:
        for line in logger.entries:
            f.write(line + "\n")
    return log_path


def write_early_classification(base_name: str,
                               out_dir: str,
                               points: np.ndarray,
                               labels: np.ndarray) -> str:
    out_path = os.path.join(out_dir, f"{base_name} XYZ-JS-early_classification.txt")
    mask = labels >= 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for p, lbl in zip(points[mask], labels[mask]):
            writer.writerow([f"{p[0]:0.8f}", f"{p[1]:0.8f}", f"{p[2]:0.8f}", int(lbl)])
    return out_path


# ---------------------------------------------------------------------------#
# Pipeline (KMeans + 2D-DBSCAN + RANSAC)

class KMeansPatchPipeline:
    """KMeans for set clustering, 2D DBSCAN for patch clustering, RANSAC for planes."""

    def __init__(self, params: KMeansPatchParams):
        self.params = params

    def run(self,
            pcd: o3d.geometry.PointCloud,
            input_path: str,
            out_dir: str,
            logger: DSLogger,
            export_ascii: bool
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[PlaneRecord], np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        pts = np.asarray(pcd.points, dtype=np.float64)

        if not pcd.has_normals():
            raise RuntimeError(
                "Input point cloud has no normals. "
                "Please compute normals beforehand (e.g. in CloudCompare)."
            )

        normals = np.asarray(pcd.normals, dtype=np.float64)
        normals_u = ensure_lower_hemisphere(unit_rows(normals))

        load_time = time.perf_counter() - t0
        logger.log(f"Point cloud loaded successfully. File: {os.path.basename(input_path)} number of points: {len(pts)}")

        km = KMeans(n_clusters=self.params.k, n_init=20, algorithm="lloyd", random_state=42)
        t1 = time.perf_counter()
        labels = km.fit_predict(normals_u)
        km_time = time.perf_counter() - t1

        # Set labels = KMeans labels for all points
        set_labels = labels.astype(np.int32, copy=True)

        colors = color_palette(self.params.k)
        plane_labels = np.full(len(pts), -1, dtype=np.int32)
        planes: List[PlaneRecord] = []
        plane_id = 0
        set_start = time.perf_counter()

        for sid in range(self.params.k):
            idx = np.where(labels == sid)[0]
            if idx.size == 0:
                continue
            Pset = pts[idx]
            Nset = normals_u[idx]

            # Mean set normal
            m_nm = unit_rows(Nset).mean(axis=0)
            m_nm /= (np.linalg.norm(m_nm) + 1e-12)
            if m_nm[2] > 0:
                m_nm *= -1.0

            # Local 2D coordinates
            c_set = Pset.mean(axis=0)
            u, v, _ = orthobasis_from_normal(m_nm)
            Q2 = np.c_[(Pset - c_set) @ u, (Pset - c_set) @ v]

            # 2D DBSCAN for patches
            step2d = median_step_2d(Q2)
            eps = max(1e-6, self.params.eps_scale * step2d)

            labels_2d = None
            patch_ids: List[int] = []
            for mul in (1.0, 1.5, 2.0, 3.0):
                db = DBSCAN(eps=eps * mul, min_samples=self.params.min_samples).fit(Q2)
                patch_ids = [pid for pid in np.unique(db.labels_) if pid >= 0]
                logger.log(f"[Set {sid}] 2D-DBSCAN eps={eps * mul:0.4f} min_samples={self.params.min_samples} patches={len(patch_ids)}")
                if patch_ids:
                    labels_2d = db.labels_
                    break
            if labels_2d is None:
                logger.log(f"[Set {sid}] No 2D-DBSCAN patches found; skipping this set for plane extraction.")
                continue

            patch_poles: List[np.ndarray] = []
            patch_weights: List[float] = []

            for patch_idx, pid in enumerate(patch_ids, start=1):
                local = np.where(labels_2d == pid)[0]
                P = Pset[local]
                if len(P) < max(3, self.params.min_points_plane):
                    logger.log(f"[Set {sid} patch {patch_idx}] too few points ({len(P)}), skipping.")
                    continue

                # Optional downsample for RANSAC speed
                if len(P) <= 8000:
                    P_ds = P
                else:
                    rng = np.random.default_rng(42)
                    P_ds = P[rng.choice(len(P), 8000, replace=False)]

                # Local spacing -> RANSAC threshold
                try:
                    nn = NearestNeighbors(n_neighbors=2).fit(P_ds)
                    d2, _ = nn.kneighbors(P_ds)
                    step_local = float(np.median(d2[:, 1]))
                except Exception:
                    step_local = float(np.linalg.norm(P_ds.ptp(axis=0))) * 0.01
                th = max(0.0, float(self.params.ransac_dist_mul) * step_local)

                n_hat = None
                D = 0.0
                rms = 0.0
                try:
                    pc_patch = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_ds))
                    model, inliers = pc_patch.segment_plane(distance_threshold=th,
                                                            ransac_n=int(self.params.ransac_n),
                                                            num_iterations=int(self.params.ransac_iters))
                    a, b, c, d = model
                    n_hat = np.array([a, b, c], float)
                    n_hat /= (np.linalg.norm(n_hat) + 1e-12)
                    if n_hat[2] > 0:
                        n_hat *= -1.0

                    if len(inliers) >= 3:
                        inlier_pts = P_ds[np.asarray(inliers, int)]
                    else:
                        inlier_pts = P

                    if len(inlier_pts) > 3:
                        rms = float(np.std((inlier_pts - inlier_pts.mean(axis=0)) @ n_hat, ddof=1))
                    else:
                        rms = 0.0
                    D = float(d)
                except Exception:
                    n_hat, D, rms = plane_fit(P)

                # Optional RMS filter
                if self.params.max_plane_rms is not None and rms > self.params.max_plane_rms:
                    continue

                dip_p, dd_p = dip_and_dipdir_from_normal(n_hat)
                poly = convex_hull_2d(Q2[local])
                area, _ = polygon_area_perimeter(poly)
                patch_poles.append(n_hat)
                patch_weights.append(area if area > 0 else len(P))

                plane_labels[idx[local]] = plane_id
                planes.append(PlaneRecord(
                    plane_id=plane_id,
                    set_id=sid,
                    cluster_id=patch_idx,
                    n_points=len(P),
                    normal=n_hat,
                    d=D,
                    dip=dip_p,
                    dipdir=dd_p,
                    rms=rms,
                    method_tag="kmeans_patch_ransac",
                    threshold=th
                ))
                plane_id += 1

            # Weighted mean set orientation from patch poles (DSE-like summary)
            if patch_poles:
                Pp = np.vstack(patch_poles)
                w = np.asarray(patch_weights, float)
                w = w / (w.sum() + 1e-12)
                m = (w[:, None] * Pp).sum(axis=0)
                m /= (np.linalg.norm(m) + 1e-12)
                if m[2] > 0:
                    m *= -1.0
            else:
                m = m_nm

            dip_w, dd_w = dip_and_dipdir_from_normal(m)
            logger.log(f"Set {sid}: weighted mean dip={dip_w:0.1f} deg, dipdir={dd_w:0.1f} deg")

        # Colored PLY by set labels
        cols = np.tile(np.array([[0.4, 0.4, 0.4]]), (len(pts), 1))
        mask_assigned = set_labels >= 0
        cols[mask_assigned] = colors[set_labels[mask_assigned] % len(colors)]
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd_out.colors = o3d.utility.Vector3dVector(cols)
        colored_path = os.path.join(out_dir, "colored_clusters.ply")
        o3d.io.write_point_cloud(colored_path, pcd_out, write_ascii=export_ascii, compressed=not export_ascii)
        np.save(os.path.join(out_dir, "labels.npy"), set_labels)
        logger.log("Normals clustered via K-Means and colored cloud saved.")

        timings = {
            "load_sec": load_time,
            "kmeans_sec": km_time,
            "per_set_sec": time.perf_counter() - set_start,
            "total_sec": time.perf_counter() - t0,
        }
        return pts, normals_u, set_labels.astype(np.int32), plane_labels, planes, colors, timings


# ---------------------------------------------------------------------------#
# Pipeline (MeanShift + 2D-DBSCAN + RANSAC)

class MeanShiftDBSCANRansacPipeline:
    """MeanShift for set clustering, 2D DBSCAN for patch clustering, RANSAC for planes."""

    def __init__(self, set_params: SetMeanShiftParams, patch_params: PatchParams):
        self.set_params = set_params
        self.patch_params = patch_params

    def run(self,
            pcd: o3d.geometry.PointCloud,
            input_path: str,
            out_dir: str,
            logger: DSLogger,
            export_ascii: bool
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[PlaneRecord], np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        pts = np.asarray(pcd.points, dtype=np.float64)
        if not pcd.has_normals():
            raise RuntimeError(
                "Input point cloud has no normals. "
                "Please compute normals beforehand (e.g. in CloudCompare)."
            )
        normals = np.asarray(pcd.normals, dtype=np.float64)
        normals_u = ensure_lower_hemisphere(unit_rows(normals))
        load_time = time.perf_counter() - t0

        logger.log(f"Point cloud loaded successfully. File: {os.path.basename(input_path)} number of points: {len(pts)}")

        # MeanShift on normals (unit sphere chord distance) with fixed bandwidth
        bw_chord = 2.0 * math.sin(math.radians(self.set_params.bandwidth_deg) / 2.0)
        t1 = time.perf_counter()
        ms_sets = MeanShift(
            bandwidth=bw_chord,
            bin_seeding=bool(self.set_params.bin_seeding),
            cluster_all=False
        )
        raw_labels = ms_sets.fit_predict(normals_u)
        set_ms_time = time.perf_counter() - t1

        valid = [u for u in np.unique(raw_labels) if u >= 0]
        label_map = {old: new for new, old in enumerate(valid)}
        set_labels = np.full_like(raw_labels, -1)
        for old, new in label_map.items():
            set_labels[raw_labels == old] = new
        n_sets = len(valid)

        if n_sets == 0:
            logger.log("No sets found by MeanShift; assigning all points to Set 0.")
            set_labels[:] = 0
            valid = [0]
            n_sets = 1

        colors = color_palette(max(1, n_sets))
        cols = np.tile(np.array([[0.4, 0.4, 0.4]]), (len(pts), 1))
        mask_assigned = set_labels >= 0
        cols[mask_assigned] = colors[set_labels[mask_assigned] % len(colors)]
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd_out.colors = o3d.utility.Vector3dVector(cols)
        colored_path = os.path.join(out_dir, "colored_clusters.ply")
        o3d.io.write_point_cloud(colored_path, pcd_out,
                                 write_ascii=export_ascii,
                                 compressed=not export_ascii)
        np.save(os.path.join(out_dir, "labels.npy"), set_labels)
        logger.log("Normals clustered via MeanShift and colored cloud saved.")

        plane_labels = np.full(len(pts), -1, dtype=np.int32)
        planes: List[PlaneRecord] = []
        plane_id = 0
        set_start = time.perf_counter()

        for sid in valid:
            idx = np.where(set_labels == sid)[0]
            if idx.size == 0:
                continue

            Pset = pts[idx]
            Nset = normals_u[idx]

            # Mean set normal
            m_nm = unit_rows(Nset).mean(axis=0)
            m_nm /= (np.linalg.norm(m_nm) + 1e-12)
            if m_nm[2] > 0:
                m_nm *= -1.0

            # Local 2D coordinates
            c_set = Pset.mean(axis=0)
            u, v, _ = orthobasis_from_normal(m_nm)
            Q2 = np.c_[(Pset - c_set) @ u, (Pset - c_set) @ v]

            # 2D DBSCAN for patches
            step2d = median_step_2d(Q2)
            eps = max(1e-6, self.patch_params.eps_scale * step2d)

            labels_2d = None
            patch_ids: List[int] = []
            for mul in (1.0, 1.5, 2.0, 3.0):
                db = DBSCAN(eps=eps * mul, min_samples=self.patch_params.min_samples).fit(Q2)
                patch_ids = [pid for pid in np.unique(db.labels_) if pid >= 0]
                logger.log(
                    f"[Set {sid}] 2D-DBSCAN eps={eps * mul:0.4f} "
                    f"min_samples={self.patch_params.min_samples} patches={len(patch_ids)}"
                )
                if patch_ids:
                    labels_2d = db.labels_
                    break
            if labels_2d is None:
                logger.log(f"[Set {sid}] No 2D-DBSCAN patches found; skipping this set for plane extraction.")
                continue

            patch_poles: List[np.ndarray] = []
            patch_weights: List[float] = []

            for patch_idx, pid in enumerate(patch_ids, start=1):
                local = np.where(labels_2d == pid)[0]
                P = Pset[local]
                if len(P) < max(3, self.patch_params.min_points_plane):
                    continue

                # Optional downsample for RANSAC speed
                if len(P) <= 8000:
                    P_ds = P
                else:
                    rng = np.random.default_rng(42)
                    P_ds = P[rng.choice(len(P), 8000, replace=False)]

                # Local spacing -> RANSAC threshold
                try:
                    nn = NearestNeighbors(n_neighbors=2).fit(P_ds)
                    d2, _ = nn.kneighbors(P_ds)
                    step_local = float(np.median(d2[:, 1]))
                except Exception:
                    step_local = float(np.linalg.norm(P_ds.ptp(axis=0))) * 0.01
                th = max(0.0, float(self.patch_params.ransac_dist_mul) * step_local)

                n_hat = None
                D = 0.0
                rms = 0.0

                try:
                    pc_patch = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_ds))
                    model, inliers = pc_patch.segment_plane(
                        distance_threshold=th,
                        ransac_n=int(self.patch_params.ransac_n),
                        num_iterations=int(self.patch_params.ransac_iters)
                    )
                    a, b, c, d = model
                    n_hat = np.array([a, b, c], float)
                    n_hat /= (np.linalg.norm(n_hat) + 1e-12)
                    if n_hat[2] > 0:
                        n_hat *= -1.0

                    if len(inliers) >= 3:
                        inlier_pts = P_ds[np.asarray(inliers, int)]
                    else:
                        inlier_pts = P

                    if len(inlier_pts) > 3:
                        rms = float(np.std((inlier_pts - inlier_pts.mean(axis=0)) @ n_hat, ddof=1))
                    else:
                        rms = 0.0

                    D = float(d)
                except Exception:
                    # fallback: LSQ plane
                    n_hat, D, rms = plane_fit(P)

                # Optional RMS filter
                if self.patch_params.max_plane_rms is not None and rms > self.patch_params.max_plane_rms:
                    continue

                dip_p, dd_p = dip_and_dipdir_from_normal(n_hat)
                poly = convex_hull_2d(Q2[local])
                area, _ = polygon_area_perimeter(poly)
                patch_poles.append(n_hat)
                patch_weights.append(area if area > 0 else len(P))

                plane_labels[idx[local]] = plane_id
                planes.append(PlaneRecord(
                    plane_id=plane_id,
                    set_id=sid,
                    cluster_id=patch_idx,
                    n_points=len(P),
                    normal=n_hat,
                    d=D,
                    dip=dip_p,
                    dipdir=dd_p,
                    rms=rms,
                    method_tag="ms_dbscan_ransac",
                    threshold=th
                ))
                plane_id += 1

            # Weighted mean set orientation from patch poles
            if patch_poles:
                Pp = np.vstack(patch_poles)
                w = np.asarray(patch_weights, float)
                w = w / (w.sum() + 1e-12)
                m = (w[:, None] * Pp).sum(axis=0)
                m /= (np.linalg.norm(m) + 1e-12)
                if m[2] > 0:
                    m *= -1.0
            else:
                m = m_nm

            dip_w, dd_w = dip_and_dipdir_from_normal(m)
            logger.log(f"Set {sid}: weighted mean dip={dip_w:0.1f} deg, dipdir={dd_w:0.1f} deg")

        timings = {
            "load_sec": load_time,
            "set_meanshift_sec": set_ms_time,
            "per_set_sec": time.perf_counter() - set_start,
            "total_sec": time.perf_counter() - t0,
        }
        return pts, normals_u, set_labels.astype(np.int32), plane_labels, planes, colors, timings


# ---------------------------------------------------------------------------#
# Viewer

def launch_viewer(points: np.ndarray,
                  labels: np.ndarray,
                  colors: np.ndarray,
                  general: GeneralOptions) -> None:
    if not general.enable_viewer:
        return
    mask = labels >= 0
    if mask.sum() == 0:
        print("Viewer skipped: no classified points.")
        return
    pts = points.copy()
    lbls = labels.copy()

    if len(pts) > general.viewer_max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(pts), general.viewer_max_points, replace=False)
        pts = pts[idx]
        lbls = lbls[idx]
        mask = lbls >= 0

    if general.viewer_backend.lower() == "pyvista":
        try:
            import pyvista as pv
        except Exception as exc:
            print("PyVista unavailable, falling back to Open3D viewer.", exc)
        else:
            pv.set_plot_theme("document")
            plotter = pv.Plotter()
            actors = {}
            unique = sorted(int(u) for u in np.unique(lbls[mask]))
            legend = []
            for sid in unique:
                sub = pts[lbls == sid]
                if not len(sub):
                    continue
                cloud = pv.PolyData(sub)
                actor = plotter.add_mesh(cloud, render_points_as_spheres=True,
                                         point_size=general.viewer_point_size,
                                         color=tuple(colors[sid % len(colors)]))
                actors[sid] = actor
                legend.append([f"Set {sid}", tuple(colors[sid % len(colors)])])
            plotter.add_legend(legend)

            for sid in unique:
                def make_cb(s=sid):
                    def callback(state):
                        actors[s].SetVisibility(state)
                        plotter.render()
                    return callback
                plotter.add_checkbox_button_widget(make_cb(), value=True,
                                                   position=(10, 10 + sid * 32),
                                                   color_on="lime", color_off="grey",
                                                   size=25, border_size=1,
                                                   background_color=tuple(colors[sid % len(colors)]))

            def show_all():
                for actor in actors.values():
                    actor.SetVisibility(True)
                plotter.render()

            def hide_all():
                for actor in actors.values():
                    actor.SetVisibility(False)
                plotter.render()

            if hasattr(plotter, "add_button_widget"):
                plotter.add_button_widget(show_all, position=(10, 10 + len(unique) * 36), size=28,
                                          color="white", style="modern", border_size=1,
                                          font_size=10, text="Show all")
                plotter.add_button_widget(hide_all, position=(110, 10 + len(unique) * 36), size=28,
                                          color="white", style="modern", border_size=1,
                                          font_size=10, text="Hide all")

            plotter.show(title="Rock Cluster viewer")
            return

    # Open3D fallback
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[mask]))
    cols = np.zeros((mask.sum(), 3), float)
    unique = sorted(int(u) for u in np.unique(lbls[mask]))
    filtered_pts = pts[mask]
    filtered_lbls = lbls[mask]
    for sid in unique:
        sub = filtered_lbls == sid
        cols[sub] = colors[sid % len(colors)]
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.visualization.draw_geometries([pcd], window_name="Sets viewer")


def launch_stereonet(normals: np.ndarray,
                     labels: np.ndarray,
                     colors: np.ndarray,
                     general: GeneralOptions) -> None:
    if not general.enable_stereonet:
        return
    try:
        import matplotlib.pyplot as plt
        import mplstereonet  # type: ignore
    except Exception as exc:
        print("Stereonet dependencies unavailable, skipping stereonet.", exc)
        return

    normals = unit_rows(normals)
    mask = labels >= 0
    if mask.any():
        normals = normals[mask]
        labels = labels[mask]

    if normals.size == 0:
        print("Stereonet skipped: no normals available.")
        return

    if len(normals) > general.stereonet_max_points:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(normals), general.stereonet_max_points, replace=False)
        normals = normals[idx]
        if mask.any():
            labels = labels[idx]

    nx = normals[:, 0]
    ny = normals[:, 1]
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    az_norm = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    plunge = np.degrees(np.arcsin(np.clip(-nz, -1.0, 1.0)))
    dip = 90.0 - plunge
    dipdir = (az_norm + 180.0) % 360.0
    strike = (dipdir - 90.0) % 360.0

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid()

    try:
        ax.density_contourf(strike, dip, measurement="poles", cmap="Greys")
        ax.density_contour(strike, dip, measurement="poles", colors="r", linewidths=0.6)
    except Exception:
        ax.density_contour(strike, dip, measurement="poles", colors="r", linewidths=0.6)
    ax.pole(strike, dip, markersize=1, alpha=0.2, color="r")
    ax.set_title("Equal-angle contours of pole concentration\nIsolines each 1.25%")
    plt.show()


# ---------------------------------------------------------------------------#
# Main orchestration

def run_suite():
    defaults = GeneralOptions()
    default_set = SetMeanShiftParams()
    default_patch = PatchParams()
    default_kmeans = KMeansPatchParams()
    dialog = ParameterDialog(defaults, default_set, default_patch, default_kmeans)
    config = dialog.show()
    if config is None:
        return

    logger = DSLogger()
    try:
        os.makedirs(config.general.output_root or os.path.join(os.path.dirname(config.input_path), "Rock_cluster OUTPUT"),
                    exist_ok=True)
    except Exception:
        pass

    pcd = o3d.io.read_point_cloud(config.input_path)
    if len(pcd.points) == 0:
        raise RuntimeError("Input point cloud is empty.")

    method_mode = (config.general.method_mode or "meanshift").strip().lower()
    if method_mode not in ("meanshift", "kmeans"):
        method_mode = "meanshift"

    # BOTH methods require pre-computed normals
    if not pcd.has_normals():
        raise RuntimeError(
            "Input point cloud has no normals. "
            "Please compute normals beforehand."
        )

    points = np.asarray(pcd.points, dtype=np.float64)
    logger.log("Normals ready for processing.")

    if config.general.output_root:
        root = config.general.output_root
    else:
        root = os.path.join(os.path.dirname(config.input_path), "Rock_cluster OUTPUT")
    base_name = os.path.splitext(os.path.basename(config.input_path))[0]
    run_dir = os.path.join(root, base_name)
    os.makedirs(run_dir, exist_ok=True)

    if method_mode == "kmeans":
        pipeline = KMeansPatchPipeline(config.kmeans)
        pts, nrm, set_labels, plane_labels, planes, colors, timings = pipeline.run(
            pcd, config.input_path, run_dir, logger, config.general.export_ascii_ply)
        method_name = "KMeans + 2D-DBSCAN + RANSAC"
        param_block = {
            "KMeans + Patch DBSCAN": {
                "k": config.kmeans.k,
                "eps_scale": config.kmeans.eps_scale,
                "min_samples": config.kmeans.min_samples,
                "min_points_plane": config.kmeans.min_points_plane,
            },
            "RANSAC": {
                "ransac_n": config.kmeans.ransac_n,
                "iterations": config.kmeans.ransac_iters,
                "threshold_mul": config.kmeans.ransac_dist_mul,
                "max_plane_rms": config.kmeans.max_plane_rms,
            },
        }
    else:
        pipeline = MeanShiftDBSCANRansacPipeline(config.set_ms, config.patch)
        pts, nrm, set_labels, plane_labels, planes, colors, timings = pipeline.run(
            pcd, config.input_path, run_dir, logger, config.general.export_ascii_ply)
        method_name = "MeanShift + 2D-DBSCAN + RANSAC"
        param_block = {
            "Set MeanShift": asdict(config.set_ms),
            "Patch DBSCAN + RANSAC": {
                "eps_scale": config.patch.eps_scale,
                "min_samples": config.patch.min_samples,
                "min_points_plane": config.patch.min_points_plane,
                "ransac_n": config.patch.ransac_n,
                "iterations": config.patch.ransac_iters,
                "threshold_mul": config.patch.ransac_dist_mul,
                "max_plane_rms": config.patch.max_plane_rms,
            },
        }

    stats = compute_set_stats(pts, nrm, set_labels)
    assigned = int((set_labels >= 0).sum())

    report_path = write_rock_cluster_report(base_name, run_dir, config.input_path,
                                            stats, planes, param_block,
                                            total_points=len(pts), assigned_points=assigned)
    logger.log(f"Saved all the results to text files; report: {report_path}")
    log_path = write_rock_cluster_log(base_name, run_dir, logger)
    early_path = write_early_classification(base_name, run_dir, pts, set_labels)

    print(f"\nOutputs written to {run_dir}")
    print(f"- Report: {report_path}")
    print(f"- Log: {log_path}")
    print(f"- Early classification: {early_path}")
    print(f"- Colored PLY + labels.npy + CSVs reside in {run_dir}")

    if config.general.enable_viewer:
        launch_viewer(pts, set_labels, colors, config.general)
    launch_stereonet(nrm, set_labels, colors, config.general)

    print(f"Done in {timings.get('total_sec', 0.0):0.2f}s ({method_name}).")


if __name__ == "__main__":
    try:
        run_suite()
    except Exception as exc:
        import traceback as tb
        tb.print_exc()
        try:
            messagebox.showerror("Clustering failed", str(exc))
        except Exception:
            pass
        sys.exit(1)
