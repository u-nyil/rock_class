# Point Cloud Normal Computation Tool
# Supports: .ply, .pcd, .xyz, .txt, .pts, .las
# Methods: Local Hough, KNN PCA, Robust PCA

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import open3d as o3d
import os
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from sklearn.neighbors import KDTree
import ttkbootstrap as ttk


# ============ Visualization ============

def _bbox_diag(pcd):
    return np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())

def color_by_normals_rgb(pcd):
    if pcd.has_normals():
        n = np.asarray(pcd.normals)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(0.5 * (n + 1.0), 0.0, 1.0))

def make_normal_glyphs(pcd, scale=0.05, every=10):
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    idx = np.arange(0, len(pts), max(1, int(every)))
    p, q = pts[idx], pts[idx] + nrm[idx] * float(scale)
    all_pts = np.vstack([p, q])
    n = len(p)
    lines = np.column_stack([np.arange(n), np.arange(n) + n]).astype(int)
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(all_pts), lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(lines), 1)))
    return ls

def draw_with_normals(pcd, show_normals=True, scale=0.05, every=10, color_mode="normals", window_name="Normals Viewer"):
    geoms = [pcd]
    if color_mode == "normals" and pcd.has_normals():
        color_by_normals_rgb(pcd)
    elif color_mode == "uniform":
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(pcd.points), 3), 0.85))
    if show_normals and pcd.has_normals():
        diag = _bbox_diag(pcd)
        scale_use = (diag * scale) if scale < 1.0 else float(scale)
        geoms.append(make_normal_glyphs(pcd, scale=scale_use, every=max(1, int(every))))
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


# ============ Normal Estimation Helpers ============

def build_index(points, k_neighbors=None, radius=None):
    return KDTree(points, metric="euclidean")

def query_neighbors(index, points, k_neighbors=None, radius=None):
    if radius is not None and radius > 0:
        return index.query_radius(points, r=radius, return_distance=True)
    k = k_neighbors or 30
    dist, idx = index.query(points, k=k)
    return idx, dist

def orient_normals(normals, points, orient="zup", viewpoint=None):
    normals = normals.copy()
    if orient == "none":
        return normals
    if orient == "zup":
        normals[normals[:, 2] < 0] *= -1
    elif orient == "zdown":
        normals[normals[:, 2] > 0] *= -1
    elif orient == "viewpoint":
        vp = np.asarray(viewpoint or [0, 0, 0], dtype=float)
        to_vp = vp[None, :] - points
        normals[(normals * to_vp).sum(axis=1) < 0] *= -1
    return normals


# ============ Local Hough Transform ============

@dataclass
class LocalHoughParams:
    k_neighbors: int = 80
    num_planes: int = 1000
    acc_steps: int = 20
    num_rotations: int = 5
    tol_deg: float = 15.0
    density_k: int = 8
    use_density: bool = True
    seed: Optional[int] = None
    workers: int = 0

def _random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    r, s = np.sqrt(1.0 - u1), np.sqrt(u1)
    theta, phi = 2 * np.pi * u2, 2 * np.pi * u3
    qx, qy, qz, qw = r * np.sin(theta), r * np.cos(theta), s * np.sin(phi), s * np.cos(phi)
    x2, y2, z2 = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([[1-2*(y2+z2), 2*(xy-wz), 2*(xz+wy)],
                     [2*(xy+wz), 1-2*(x2+z2), 2*(yz-wx)],
                     [2*(xz-wy), 2*(yz+wx), 1-2*(x2+y2)]], dtype=np.float64)

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def _to_lower_hemisphere(n):
    n = n.copy()
    n[n[..., 2] > 0] *= -1.0
    return n

def _trend_plunge_from_vec(n):
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    trend = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    plunge = np.degrees(np.arcsin(np.clip(-nz, -1.0, 1.0)))
    return trend, plunge

def _vec_from_trend_plunge(trend_deg, plunge_deg):
    t, p = np.radians(trend_deg), np.radians(plunge_deg)
    return _unit(np.stack([np.sin(t)*np.cos(p), np.cos(t)*np.cos(p), -np.sin(p)], axis=-1))

def _accumulate_bins(trend_deg, plunge_deg, acc_steps):
    nphi_tr, nphi_pl = 2 * acc_steps, acc_steps
    it = np.clip(np.floor(trend_deg / 360.0 * nphi_tr).astype(int), 0, nphi_tr - 1)
    ip = np.clip(np.floor(plunge_deg / 90.0 * nphi_pl).astype(int), 0, nphi_pl - 1)
    acc = np.zeros((nphi_pl, nphi_tr), dtype=np.float32)
    np.add.at(acc, (ip, it), 1.0)
    return acc

def _smooth_box(acc, r):
    if r <= 0:
        return acc
    a = acc.copy()
    for _ in range(r):
        a = (a + np.roll(a, 1, 1) + np.roll(a, -1, 1) +
             np.pad(a[:-1], ((1,0),(0,0)), mode="edge") + np.pad(a[1:], ((0,1),(0,0)), mode="edge") +
             np.roll(np.pad(a[:-1], ((1,0),(0,0)), mode="edge"), 1, 1) +
             np.roll(np.pad(a[:-1], ((1,0),(0,0)), mode="edge"), -1, 1) +
             np.roll(np.pad(a[1:], ((0,1),(0,0)), mode="edge"), 1, 1) +
             np.roll(np.pad(a[1:], ((0,1),(0,0)), mode="edge"), -1, 1)) / 9.0
    return a

def _triplet_normals(pts, rng, T):
    m = pts.shape[0]
    if m < 3:
        return np.empty((0, 3), dtype=np.float64)
    idx = rng.integers(0, m, size=(T, 3))
    mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
    for _ in range(5):
        if mask.all():
            break
        red = np.where(~mask)[0]
        idx[red] = rng.integers(0, m, size=(len(red), 3))
        mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
    q, r = pts[idx[:, 1]] - pts[idx[:, 0]], pts[idx[:, 2]] - pts[idx[:, 0]]
    n = _unit(np.cross(q, r))
    return n[np.linalg.norm(np.cross(q, r), axis=1) > 1e-10]

def local_hough_normals(points, params, kdtree=None):
    N = points.shape[0]
    rng = np.random.default_rng(params.seed)
    tree = kdtree or KDTree(points, metric="euclidean")
    nphi = params.acc_steps
    bins_tr, bins_pl = 2 * nphi, nphi
    trend_centers = (np.arange(bins_tr) + 0.5) * (360.0 / bins_tr)
    plunge_centers = (np.arange(bins_pl) + 0.5) * (90.0 / bins_pl)
    bin_dirs = _vec_from_trend_plunge(
        np.repeat(trend_centers[None, :], bins_pl, axis=0).ravel(),
        np.repeat(plunge_centers[:, None], bins_tr, axis=1).ravel()
    ).reshape(bins_pl, bins_tr, 3)
    normals = np.zeros((N, 3), dtype=np.float64)
    peak_height = np.zeros(N, dtype=np.float32)
    _, nbr_idx = tree.query(points, k=min(params.k_neighbors, N))

    for i in range(N):
        nbrs = points[nbr_idx[i]]
        if nbrs.shape[0] < 3:
            normals[i], peak_height[i] = [0, 0, -1.0], 0
            continue
        n_trip = _triplet_normals(nbrs, rng, params.num_planes)
        if n_trip.size == 0:
            normals[i], peak_height[i] = [0, 0, -1.0], 0
            continue
        n_trip = _to_lower_hemisphere(n_trip)
        acc = np.zeros((bins_pl, bins_tr), dtype=np.float32)
        for _ in range(max(1, params.num_rotations)):
            R = _random_rotation_matrix(rng)
            n_rot = _to_lower_hemisphere((R @ n_trip.T).T)
            tr, pl = _trend_plunge_from_vec(n_rot)
            acc += _accumulate_bins(tr, pl, nphi)
        if params.tol_deg > 0:
            acc = _smooth_box(acc, max(0, int(round(params.tol_deg / (90.0 / nphi)))))
        ip, it = np.unravel_index(np.argmax(acc), acc.shape)
        normals[i], peak_height[i] = bin_dirs[ip, it], acc[ip, it]
    return normals, {"lh_peak": peak_height}


# ============ PCA Normals ============

def local_pca_normals(points, k_neighbors=30, radius=None, min_neighbors=10, orient="zup", viewpoint=None):
    idx = build_index(points, k_neighbors, radius)
    inds, _ = query_neighbors(idx, points, k_neighbors, radius)
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=float)
    for i in range(N):
        nbrs = inds[i] if isinstance(inds, np.ndarray) else inds[i]
        if len(nbrs) < min_neighbors:
            normals[i] = [0, 0, 1.0]
            continue
        P = points[nbrs]
        Q = P - P.mean(axis=0)
        _, v = np.linalg.eigh((Q.T @ Q) / max(1, Q.shape[0]))
        n = v[:, 0]
        normals[i] = n / np.linalg.norm(n)
    return orient_normals(normals, points, orient, viewpoint)


# ============ Robust PCA ============

def _plane_from_points(P):
    Q = P - P.mean(axis=0)
    _, v = np.linalg.eigh((Q.T @ Q) / max(1, Q.shape[0]))
    n = v[:, 0]
    return n / np.linalg.norm(n)

def _mad_sigma(res):
    med = np.median(res)
    return 1.4826 * (np.median(np.abs(res - med)) + 1e-12)

def _irls_weights(res, kind, sigma):
    r = np.abs(res) / max(sigma, 1e-9)
    if kind == "huber":
        delta = 1.345
        w = np.ones_like(r)
        w[r > delta] = delta / r[r > delta]
        return w
    c = 4.685
    w = np.zeros_like(r)
    m = r < c
    z = r[m] / c
    w[m] = (1 - z**2)**2
    return w

def _weighted_cov(Q, w):
    m = Q.shape[0]
    ww = w / (w.sum() + 1e-12) * m
    Qw = Q * ww[:, None]
    return (Qw.T @ Q) / max(1, m)

def _ransac_plane(P, residual_threshold, min_samples, max_trials=50):
    m = P.shape[0]
    if m < min_samples:
        return np.array([0, 0, 1.0]), np.zeros(m, dtype=bool)
    best_n, best_in, best_count = None, None, -1
    rng = np.random.default_rng(42)
    for _ in range(max_trials):
        idx = rng.choice(m, size=min_samples, replace=False)
        tri = P[idx]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        inliers = np.abs(P @ n - n.dot(tri[0])) <= residual_threshold
        if inliers.sum() > best_count:
            best_count, best_in, best_n = inliers.sum(), inliers, n
    if best_n is None:
        return _plane_from_points(P), np.ones(m, dtype=bool)
    return best_n, best_in

def robust_pca_normals(points, k_neighbors=30, radius=None, min_neighbors=10, loss="huber",
                       max_iters=10, tol=1e-3, ransac_residual_threshold=0.01, ransac_min_samples=3,
                       orient="zup", viewpoint=None):
    idx = build_index(points, k_neighbors, radius)
    inds, _ = query_neighbors(idx, points, k_neighbors, radius)
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=float)
    for i in range(N):
        nbrs = inds[i] if isinstance(inds, np.ndarray) else inds[i]
        if len(nbrs) < min_neighbors:
            normals[i] = [0, 0, 1.0]
            continue
        P = points[nbrs]
        n = _plane_from_points(P)
        mu = P.mean(axis=0)
        d = -n.dot(mu)
        converged = False
        for _ in range(max_iters):
            res = P @ n + d
            sigma = _mad_sigma(res)
            w = _irls_weights(res, loss, sigma)
            Q = P - (P * w[:, None]).sum(axis=0) / (w.sum() + 1e-12)
            try:
                _, v = np.linalg.eigh(_weighted_cov(Q, w))
                n_new = v[:, 0] / np.linalg.norm(v[:, 0])
            except:
                break
            mu_w = (P * w[:, None]).sum(axis=0) / (w.sum() + 1e-12)
            d_new = -n_new.dot(mu_w)
            if np.arccos(np.clip(np.abs(n.dot(n_new)), 0, 1)) < tol:
                converged = True
                break
            n, d = n_new, d_new
        if not converged:
            n, _ = _ransac_plane(P, ransac_residual_threshold, ransac_min_samples)
        normals[i] = n
    return orient_normals(normals, points, orient, viewpoint)


# ============ Orientation Strategies ============

@dataclass
class OrientationResult:
    normals: np.ndarray
    flipped_count: int
    total: int
    details: Dict[str, Any]

def _as_unit(v):
    return v / (np.linalg.norm(v) + 1e-12)

def _o3d_pcd(points, normals):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def orient_viewpoint(points, normals, camera):
    n = normals.copy()
    v = points - camera[None, :]
    flip = (n * v).sum(axis=1) > 0.0
    n[flip] *= -1.0
    return OrientationResult(n, int(flip.sum()), n.shape[0], {"strategy": "viewpoint", "camera": camera.tolist()})

def orient_consistent(points, normals, k_consistent):
    pcd = _o3d_pcd(points, normals)
    pcd.orient_normals_consistent_tangent_plane(k_consistent)
    n2 = np.asarray(pcd.normals)
    flip = (normals * n2).sum(axis=1) < 0.0
    return OrientationResult(n2, int(flip.sum()), n2.shape[0], {"strategy": "consistent", "k_consistent": int(k_consistent)})

def orient_refvec(normals, refvec):
    n = normals.copy()
    v = _as_unit(refvec.astype(float))
    flip = (n @ v) < 0.0
    n[flip] *= -1.0
    return n, flip

def orient_reference(points, normals, refvec):
    n2, flip = orient_refvec(normals, refvec)
    return OrientationResult(n2, int(flip.sum()), n2.shape[0], {"strategy": "refvec", "refvec": _as_unit(refvec).tolist()})

def apply_orientation_strategy(points, normals, strategy, params, planes=None):
    strategy = strategy.lower()
    if strategy == "viewpoint":
        camera = np.asarray(params.get("camera_location", [0, 0, 0]), dtype=float)
        pcd = _o3d_pcd(points, normals)
        pcd.orient_normals_towards_camera_location(camera)
        oriented = np.asarray(pcd.normals)
        flip = ((normals * (points - camera[None, :])).sum(axis=1) > 0.0)
        return OrientationResult(oriented, int(flip.sum()), normals.shape[0], {"strategy": "viewpoint", "camera": camera.tolist()})
    if strategy == "consistent":
        return orient_consistent(points, normals, int(params.get("k_consistent", 10)))
    if strategy == "refvec":
        v = np.asarray(params.get("reference_vector", [0, 0, 1]), dtype=float)
        return orient_reference(points, normals, v)
    raise ValueError(f"Unknown strategy: {strategy}")
# ============ GUI Class ============

class NormalComputerGUI:
    def __init__(self, master=None, initial_file=None):
        self._owns_root = master is None
        self.root = tk.Tk() if master is None else tk.Toplevel(master)
        if not self._owns_root:
            self.root.transient(master)
        self.root.title("Point Cloud Normal Computer")
        self.root.geometry("950x850")
        self.style = ttk.Style(theme="litera")
        self.style.configure("TButton", font=("Arial", 11, "bold"), padding=6)
        
        self.points = self.normals = self.colors = self.input_file = self.pcd = None
        self.selected_method = tk.StringVar(value="method_a")
        self.params = {}
        self.create_gui()
        if initial_file:
            try:
                self.load_file(initial_file)
            except:
                pass

    def create_gui(self):
        # Scrollable main canvas
        main_canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas)
        scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        main_canvas.bind_all("<MouseWheel>", lambda e: main_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        main_frame = tk.Frame(scrollable_frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        tk.Label(main_frame, text="Point Cloud Normal Computer", font=("Arial", 16, "bold"), fg="#2196F3").pack(pady=10)
        
        # File Selection
        file_frame = tk.LabelFrame(main_frame, text="1. Input File", font=("Arial", 11, "bold"), padx=15, pady=10)
        file_frame.pack(fill="x", pady=5)
        ttk.Button(file_frame, text="Load Point Cloud File", command=self.load_file, width=26, bootstyle="info-outline").pack(pady=5)
        self.file_label = tk.Label(file_frame, text="No file loaded", wraplength=700, fg="gray")
        self.file_label.pack(pady=5)
        self.info_label = tk.Label(file_frame, text="", fg="#2196F3", font=("Arial", 9))
        self.info_label.pack()
        
        # Method Selection
        method_frame = tk.LabelFrame(main_frame, text="2. Select Method", font=("Arial", 11, "bold"), padx=15, pady=10)
        method_frame.pack(fill="x", pady=5)
        for val, txt, desc in [("method_a", "Method A: Local Hough Transform", "Best for noisy data"),
                               ("method_b", "Method B: KNN + PCA", "Fast for smooth surfaces"),
                               ("method_c", "Method C: Robust PCA (IRLS)", "Robust to outliers")]:
            tk.Radiobutton(method_frame, text=txt, variable=self.selected_method, value=val,
                          command=self.on_method_change, font=("Arial", 10, "bold")).pack(anchor="w", pady=2)
            tk.Label(method_frame, text=f"  → {desc}", fg="gray", font=("Arial", 9)).pack(anchor="w", padx=20)
        
        # Parameters Frame
        self.params_container = tk.LabelFrame(main_frame, text="3. Method Parameters", font=("Arial", 11, "bold"), padx=15, pady=10)
        self.params_container.pack(fill="x", pady=5)
        self.canvas = tk.Canvas(self.params_container, height=200)
        params_scrollbar = tk.Scrollbar(self.params_container, orient="vertical", command=self.canvas.yview)
        self.params_frame = tk.Frame(self.canvas)
        self.params_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=params_scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        params_scrollbar.pack(side="right", fill="y")
        self.create_method_a_params()
        
        # Orientation Section
        orient_frame = tk.LabelFrame(main_frame, text="4. Orientation (Optional)", font=("Arial", 11, "bold"), padx=15, pady=10)
        orient_frame.pack(fill="x", pady=5)
        self.use_orientation = tk.BooleanVar(value=False)
        tk.Checkbutton(orient_frame, text="Apply Normal Orientation", variable=self.use_orientation,
                      command=self.toggle_orientation, font=("Arial", 10, "bold")).pack(anchor="w")
        self.orient_params_frame = tk.Frame(orient_frame)
        self.orient_params_frame.pack(fill="x", pady=5)
        self.create_orientation_params()
        self.toggle_orientation()
        
        # Viewer Settings
        viewer_frame = tk.LabelFrame(main_frame, text="5. Viewer Settings", font=("Arial", 11, "bold"), padx=15, pady=10)
        viewer_frame.pack(fill="x", pady=5)
        
        length_frame = tk.Frame(viewer_frame)
        length_frame.pack(fill="x", pady=5)
        tk.Label(length_frame, text="Arrow Length:", font=("Arial", 9, "bold")).pack(side="left", padx=5)
        self.arrow_length_var = tk.DoubleVar(value=0.05)
        tk.Scale(length_frame, from_=0.01, to=0.2, resolution=0.01, orient="horizontal",
                variable=self.arrow_length_var, length=250).pack(side="left", padx=5)
        
        density_frame = tk.Frame(viewer_frame)
        density_frame.pack(fill="x", pady=5)
        tk.Label(density_frame, text="Arrow Density (%):", font=("Arial", 9, "bold")).pack(side="left", padx=5)
        self.arrow_density_var = tk.IntVar(value=10)
        tk.Scale(density_frame, from_=1, to=100, resolution=1, orient="horizontal",
                variable=self.arrow_density_var, length=250).pack(side="left", padx=5)
        
        color_frame = tk.Frame(viewer_frame)
        color_frame.pack(fill="x", pady=5)
        tk.Label(color_frame, text="Color Mode:", font=("Arial", 9, "bold")).pack(side="left", padx=5)
        self.color_mode_var = tk.StringVar(value="normals")
        ttk.Combobox(color_frame, textvariable=self.color_mode_var, values=["normals", "original", "uniform"],
                    state="readonly", width=15).pack(side="left", padx=5)
        
        # Action Buttons
        action_frame = tk.Frame(main_frame, pady=10)
        action_frame.pack(fill="x")
        ttk.Button(action_frame, text="Compute Normals", command=self.compute_normals, width=17, bootstyle="warning").pack(side="left", padx=5)
        ttk.Button(action_frame, text="Preview", command=self.preview_normals, width=17, bootstyle="info").pack(side="left", padx=5)
        ttk.Button(action_frame, text="Export to PLY", command=self.export_normals, width=17, bootstyle="success").pack(side="left", padx=5)
        self.stereonet_button = ttk.Button(action_frame, text="Stereonet", command=self.show_stereonet_plot, width=17, state="disabled", bootstyle="secondary")
        self.stereonet_button.pack(side="left", padx=5)
        self.dip_export_button = ttk.Button(action_frame, text="Export Dip Table", command=self.export_dip_data, width=17, state="disabled", bootstyle="secondary-outline")
        self.dip_export_button.pack(side="left", padx=5)
        
        # Log
        log_frame = tk.LabelFrame(main_frame, text="Status Log", font=("Arial", 10, "bold"), padx=10, pady=5)
        log_frame.pack(fill="x", pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, font=("Courier", 9), state="disabled")
        self.log_text.pack(fill="both", expand=False)
        self.log("Welcome! Load a point cloud file to start.")
        self.refresh_normal_actions()

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[INFO] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def normals_available(self):
        return self.points is not None and self.normals is not None and len(self.points) == len(self.normals)

    def refresh_normal_actions(self):
        state = "normal" if self.normals_available() else "disabled"
        for btn in (self.stereonet_button, self.dip_export_button):
            if btn:
                btn.config(state=state)
        self.root.update()

    def _add_numeric_param(self, frame, row, label, var, min_val, max_val, step, desc, width=10):
        ttk.Label(frame, text=label, font=("Arial", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5, padx=5)
        ttk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val, increment=step, width=width,
                   bootstyle="secondary", justify="center").grid(row=row, column=1, sticky="w", pady=5)
        tk.Label(frame, text=f"{desc} ({min_val}-{max_val})", fg="gray", font=("Arial", 8)).grid(row=row, column=2, sticky="w", padx=10)

    def on_method_change(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        method = self.selected_method.get()
        if method == "method_a":
            self.create_method_a_params()
        elif method == "method_b":
            self.create_method_b_params()
        else:
            self.create_method_c_params()
        self.log(f"Switched to {method.replace('_', ' ').title()}")

    def create_method_a_params(self):
        f = self.params_frame
        self.params['k_neighbors'] = tk.IntVar(value=80)
        self._add_numeric_param(f, 0, "K Neighbors:", self.params['k_neighbors'], 10, 200, 1, "Local neighbors")
        self.params['num_planes'] = tk.IntVar(value=1000)
        self._add_numeric_param(f, 1, "Num Planes:", self.params['num_planes'], 100, 5000, 100, "Random triplets")
        self.params['acc_steps'] = tk.IntVar(value=20)
        self._add_numeric_param(f, 2, "Acc Steps:", self.params['acc_steps'], 10, 50, 1, "Angular resolution")
        self.params['num_rotations'] = tk.IntVar(value=5)
        self._add_numeric_param(f, 3, "Num Rotations:", self.params['num_rotations'], 1, 10, 1, "Random rotations")
        self.params['tol_deg'] = tk.DoubleVar(value=15.0)
        self._add_numeric_param(f, 4, "Tolerance (deg):", self.params['tol_deg'], 5.0, 30.0, 0.5, "Smoothing tolerance")
        tk.Label(f, text="Random Seed:", font=("Arial", 9, "bold")).grid(row=5, column=0, sticky="w", pady=5, padx=5)
        self.params['seed'] = tk.IntVar(value=42)
        tk.Entry(f, textvariable=self.params['seed'], width=10).grid(row=5, column=1, sticky="w", pady=5)
        tk.Label(f, text="-1 for random", fg="gray", font=("Arial", 8)).grid(row=5, column=2, sticky="w", padx=10)

    def create_method_b_params(self):
        f = self.params_frame
        self.params['k_neighbors_b'] = tk.IntVar(value=30)
        self._add_numeric_param(f, 0, "K Neighbors:", self.params['k_neighbors_b'], 5, 100, 1, "Nearest neighbors")
        self.params['radius_b'] = tk.DoubleVar(value=0.0)
        self._add_numeric_param(f, 1, "Radius:", self.params['radius_b'], 0.0, 1.0, 0.01, "0=K only")
        self.params['min_neighbors_b'] = tk.IntVar(value=10)
        self._add_numeric_param(f, 2, "Min Neighbors:", self.params['min_neighbors_b'], 3, 50, 1, "Minimum required")
        tk.Label(f, text="Initial Orient:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky="w", pady=5, padx=5)
        self.params['orient_b'] = tk.StringVar(value="zup")
        ttk.Combobox(f, textvariable=self.params['orient_b'], values=["zup", "zdown", "none"],
                    state="readonly", width=15).grid(row=3, column=1, sticky="w", pady=5)

    def create_method_c_params(self):
        f = self.params_frame
        self.params['k_neighbors_c'] = tk.IntVar(value=30)
        self._add_numeric_param(f, 0, "K Neighbors:", self.params['k_neighbors_c'], 5, 100, 1, "Nearest neighbors")
        self.params['radius_c'] = tk.DoubleVar(value=0.0)
        self._add_numeric_param(f, 1, "Radius:", self.params['radius_c'], 0.0, 1.0, 0.01, "0=K only")
        self.params['min_neighbors_c'] = tk.IntVar(value=10)
        self._add_numeric_param(f, 2, "Min Neighbors:", self.params['min_neighbors_c'], 3, 50, 1, "Minimum required")
        tk.Label(f, text="Loss Function:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky="w", pady=5, padx=5)
        self.params['loss_c'] = tk.StringVar(value="huber")
        ttk.Combobox(f, textvariable=self.params['loss_c'], values=["huber", "tukey"],
                    state="readonly", width=15).grid(row=3, column=1, sticky="w", pady=5)
        self.params['max_iters_c'] = tk.IntVar(value=10)
        self._add_numeric_param(f, 4, "Max Iterations:", self.params['max_iters_c'], 5, 50, 1, "IRLS iterations")
        self.params['tol_c'] = tk.DoubleVar(value=0.001)
        self._add_numeric_param(f, 5, "Tolerance:", self.params['tol_c'], 0.0001, 0.01, 0.0001, "Convergence")
        tk.Label(f, text="Initial Orient:", font=("Arial", 9, "bold")).grid(row=6, column=0, sticky="w", pady=5, padx=5)
        self.params['orient_c'] = tk.StringVar(value="zup")
        ttk.Combobox(f, textvariable=self.params['orient_c'], values=["zup", "zdown", "none"],
                    state="readonly", width=15).grid(row=6, column=1, sticky="w", pady=5)

    def create_orientation_params(self):
        f = self.orient_params_frame
        for w in f.winfo_children():
            w.destroy()
        tk.Label(f, text="Strategy:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", pady=5, padx=5)
        self.params['orient_strategy'] = tk.StringVar(value="refvec")
        strategy_combo = ttk.Combobox(f, textvariable=self.params['orient_strategy'],
                                     values=["refvec", "viewpoint", "consistent"], state="readonly", width=15)
        strategy_combo.grid(row=0, column=1, sticky="w", pady=5)
        strategy_combo.bind("<<ComboboxSelected>>", lambda e: self.update_orient_params())
        
        # Reference vector
        self.orient_refvec_frame = tk.Frame(f)
        self.orient_refvec_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        tk.Label(self.orient_refvec_frame, text="Ref Vector:", font=("Arial", 9)).grid(row=0, column=0, sticky="w", padx=5)
        self.params['refvec_x'] = tk.DoubleVar(value=0.0)
        self.params['refvec_y'] = tk.DoubleVar(value=0.0)
        self.params['refvec_z'] = tk.DoubleVar(value=1.0)
        for i, (lbl, var) in enumerate([("X:", 'refvec_x'), ("Y:", 'refvec_y'), ("Z:", 'refvec_z')]):
            tk.Label(self.orient_refvec_frame, text=lbl).grid(row=0, column=1+i*2, padx=2)
            tk.Entry(self.orient_refvec_frame, textvariable=self.params[var], width=8).grid(row=0, column=2+i*2, padx=2)
        
        # Viewpoint
        self.orient_viewpoint_frame = tk.Frame(f)
        self.orient_viewpoint_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)
        tk.Label(self.orient_viewpoint_frame, text="Camera:", font=("Arial", 9)).grid(row=0, column=0, sticky="w", padx=5)
        self.params['cam_x'] = tk.DoubleVar(value=0.0)
        self.params['cam_y'] = tk.DoubleVar(value=0.0)
        self.params['cam_z'] = tk.DoubleVar(value=0.0)
        for i, (lbl, var) in enumerate([("X:", 'cam_x'), ("Y:", 'cam_y'), ("Z:", 'cam_z')]):
            tk.Label(self.orient_viewpoint_frame, text=lbl).grid(row=0, column=1+i*2, padx=2)
            tk.Entry(self.orient_viewpoint_frame, textvariable=self.params[var], width=8).grid(row=0, column=2+i*2, padx=2)
        
        # Consistent
        self.orient_consistent_frame = tk.Frame(f)
        self.orient_consistent_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=5)
        tk.Label(self.orient_consistent_frame, text="K Consistent:", font=("Arial", 9)).grid(row=0, column=0, sticky="w", padx=5)
        self.params['k_consistent'] = tk.IntVar(value=10)
        ttk.Spinbox(self.orient_consistent_frame, textvariable=self.params['k_consistent'],
                   from_=5, to=50, increment=1, width=8, bootstyle="secondary").grid(row=0, column=1, sticky="w", pady=5)
        self.update_orient_params()

    def update_orient_params(self):
        strategy = self.params['orient_strategy'].get()
        self.orient_refvec_frame.grid_remove()
        self.orient_viewpoint_frame.grid_remove()
        self.orient_consistent_frame.grid_remove()
        if strategy == "refvec":
            self.orient_refvec_frame.grid()
        elif strategy == "viewpoint":
            self.orient_viewpoint_frame.grid()
        elif strategy == "consistent":
            self.orient_consistent_frame.grid()

    def toggle_orientation(self):
        state = "normal" if self.use_orientation.get() else "disabled"
        def set_state(widget):
            try:
                widget.configure(state=state)
            except:
                pass
            for child in widget.winfo_children():
                set_state(child)
        for child in self.orient_params_frame.winfo_children():
            set_state(child)

    def load_file(self, filename=None):
        filetypes = [("All Supported", "*.ply *.pcd *.xyz *.txt *.pts *.las"), ("PLY", "*.ply"),
                    ("PCD", "*.pcd"), ("XYZ", "*.xyz"), ("Text", "*.txt"), ("PTS", "*.pts"),
                    ("LAS", "*.las"), ("All", "*.*")]
        if not filename:
            filename = filedialog.askopenfilename(title="Select Point Cloud", filetypes=filetypes)
        if not filename:
            return
        try:
            self.log(f"Loading: {os.path.basename(filename)}")
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.ply', '.pcd']:
                pcd = o3d.io.read_point_cloud(filename)
                self.points = np.asarray(pcd.points)
                self.colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                self.normals = np.asarray(pcd.normals) if pcd.has_normals() else None
            elif ext == '.las':
                self.points, self.colors, self.normals = self.load_las(filename)
            else:
                self.points, self.colors, self.normals = self.load_xyz_format(filename)
            if self.points is None or len(self.points) == 0:
                messagebox.showerror("Error", "Failed to load or empty file")
                return
            self.input_file = filename
            self.file_label.config(text=f"✓ {os.path.basename(filename)}", fg="green")
            self.info_label.config(text=f"Points: {len(self.points):,} | Colors: {self.colors is not None} | Normals: {self.normals is not None}")
            self.log(f"Loaded {len(self.points):,} points")
            self.refresh_normal_actions()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")
            self.log(f"ERROR: {e}")

    def load_xyz_format(self, filename):
        data = np.loadtxt(filename)
        if data.shape[1] < 3:
            raise ValueError("Need at least 3 columns (XYZ)")
        points = data[:, :3]
        colors = normals = None
        if data.shape[1] >= 9:
            colors = data[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
            normals = data[:, 6:9]
        elif data.shape[1] >= 6:
            pot = data[:, 3:6]
            if self.looks_like_colors(pot):
                colors = pot / 255.0 if pot.max() > 1.0 else pot
            else:
                normals = pot
        return points, colors, normals

    def looks_like_colors(self, data):
        if data.max() > 1.5:
            return True
        if data.min() < -0.1:
            return False
        return np.mean(np.linalg.norm(data, axis=1)) < 0.8 or np.mean(np.linalg.norm(data, axis=1)) > 1.2

    def load_las(self, filename):
        try:
            import laspy
            las = laspy.read(filename)
            points = np.vstack((las.x, las.y, las.z)).T
            colors = None
            if hasattr(las, 'red'):
                colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            return points, colors, None
        except ImportError:
            self.log("laspy not installed")
            return self.load_xyz_format(filename)

    def compute_normals(self):
        if self.points is None:
            messagebox.showwarning("Warning", "Load a point cloud first")
            return
        thread = threading.Thread(target=self._compute_normals_thread, daemon=True)
        thread.start()

    def _compute_normals_thread(self):
        try:
            method = self.selected_method.get()
            self.log(f"Computing normals ({method})...")
            if method == "method_a":
                self.normals = self.compute_method_a()
            elif method == "method_b":
                self.normals = self.compute_method_b()
            else:
                self.normals = self.compute_method_c()
            if self.use_orientation.get():
                self.normals = self.apply_orientation()
            self.log("Done!")
            messagebox.showinfo("Success", "Normals computed!")
            self.root.after(0, self.refresh_normal_actions)
        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Failed:\n{e}")

    def compute_method_a(self):
        seed = self.params['seed'].get()
        params = LocalHoughParams(
            k_neighbors=self.params['k_neighbors'].get(),
            num_planes=self.params['num_planes'].get(),
            acc_steps=self.params['acc_steps'].get(),
            num_rotations=self.params['num_rotations'].get(),
            tol_deg=self.params['tol_deg'].get(),
            seed=None if seed < 0 else seed
        )
        normals, _ = local_hough_normals(self.points, params)
        return normals

    def compute_method_b(self):
        radius = self.params['radius_b'].get()
        return local_pca_normals(
            self.points,
            k_neighbors=self.params['k_neighbors_b'].get(),
            radius=None if radius <= 0 else radius,
            min_neighbors=self.params['min_neighbors_b'].get(),
            orient=self.params['orient_b'].get()
        )

    def compute_method_c(self):
        radius = self.params['radius_c'].get()
        return robust_pca_normals(
            self.points,
            k_neighbors=self.params['k_neighbors_c'].get(),
            radius=None if radius <= 0 else radius,
            min_neighbors=self.params['min_neighbors_c'].get(),
            loss=self.params['loss_c'].get(),
            max_iters=self.params['max_iters_c'].get(),
            tol=self.params['tol_c'].get(),
            orient=self.params['orient_c'].get()
        )

    def apply_orientation(self):
        strategy = self.params['orient_strategy'].get()
        params = {}
        if strategy == "refvec":
            params["reference_vector"] = [self.params['refvec_x'].get(), self.params['refvec_y'].get(), self.params['refvec_z'].get()]
        elif strategy == "viewpoint":
            params["camera_location"] = [self.params['cam_x'].get(), self.params['cam_y'].get(), self.params['cam_z'].get()]
        elif strategy == "consistent":
            params["k_consistent"] = self.params['k_consistent'].get()
        result = apply_orientation_strategy(self.points, self.normals, strategy, params)
        self.log(f"Orientation: flipped {result.flipped_count}/{result.total}")
        return result.normals

    def preview_normals(self):
        if self.normals is None:
            messagebox.showwarning("Warning", "Compute normals first")
            return
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
            if self.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.colors)
            density = self.arrow_density_var.get()
            every_n = max(1, 100 // density)
            draw_with_normals(pcd, show_normals=True, scale=self.arrow_length_var.get(),
                            every=every_n, color_mode=self.color_mode_var.get())
            self.log("Viewer closed")
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed:\n{e}")

    def _normalized_normals(self, hemisphere="lower"):
        norms = np.linalg.norm(self.normals, axis=1)
        mask = norms > 1e-9
        unit = self.normals[mask] / norms[mask, None]
        if hemisphere == "lower":
            unit[unit[:, 2] > 0] *= -1
        return unit, mask

    def compute_dip_metrics(self):
        unit, mask = self._normalized_normals("lower")
        nx, ny, nz = unit[:, 0], unit[:, 1], unit[:, 2]
        az = (np.degrees(np.arctan2(nx, ny)) + 360) % 360
        plunge = np.degrees(np.arcsin(np.clip(-nz, -1, 1)))
        dip = np.clip(90 - plunge, 0, 90)
        dipdir = (az + 180) % 360
        return dip, dipdir, mask

    def show_stereonet_plot(self):
        if not self.normals_available():
            messagebox.showwarning("Warning", "Normals required")
            return
        try:
            import matplotlib.pyplot as plt
            import mplstereonet
        except ImportError:
            messagebox.showwarning("Missing", "Install mplstereonet matplotlib")
            return
        try:
            unit, mask = self._normalized_normals("lower")
            nx, ny, nz = unit[:, 0], unit[:, 1], unit[:, 2]
            az = (np.degrees(np.arctan2(nx, ny)) + 360) % 360
            plunge = np.degrees(np.arcsin(np.clip(-nz, -1, 1)))
            dip = np.clip(90 - plunge, 0, 90)
            strike = ((az + 180) % 360 - 90) % 360
            
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='stereonet')
            ax.set_title("Pole Plot (Lower-Hemisphere)")
            contour = ax.density_contourf(strike, dip, measurement='poles', cmap='viridis', levels=12, alpha=0.95)
            ax.pole(strike, dip, marker='o', markersize=2, color='#1f77b4', alpha=0.45)
            ax.grid(color="#bbb", linestyle="--", linewidth=0.6)
            fig.colorbar(contour, ax=ax, fraction=0.05, pad=0.08, label="Concentration")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_dip_data(self):
        if not self.normals_available():
            messagebox.showwarning("Warning", "Normals required")
            return
        try:
            dip, dipdir, mask = self.compute_dip_metrics()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        default = "dip_table.csv"
        if self.input_file:
            default = f"{os.path.splitext(os.path.basename(self.input_file))[0]}_dip_table.csv"
        filename = filedialog.asksaveasfilename(title="Save Dip Table", defaultextension=".csv",
                                               initialfile=default, filetypes=[("CSV", "*.csv")])
        if not filename:
            return
        pts = self.points[mask]
        norms = self.normals[mask]
        data = np.column_stack([pts, norms, dip[:, None], dipdir[:, None]])
        np.savetxt(filename, data, delimiter=",", header="X,Y,Z,NX,NY,NZ,Dip_deg,DipDirection_deg", comments="", fmt="%.6f")
        self.log(f"Exported: {os.path.basename(filename)}")
        messagebox.showinfo("Success", f"Saved: {filename}")

    def export_normals(self):
        if self.normals is None:
            messagebox.showwarning("Warning", "Compute normals first")
            return
        default = "pointcloud_with_normals.ply"
        if self.input_file:
            default = f"{os.path.splitext(os.path.basename(self.input_file))[0]}_with_normals.ply"
        filename = filedialog.asksaveasfilename(title="Save PLY", defaultextension=".ply",
                                               initialfile=default, filetypes=[("PLY", "*.ply")])
        if not filename:
            return
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
            if self.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.colors)
            o3d.io.write_point_cloud(filename, pcd)
            self.log(f"Exported: {os.path.basename(filename)}")
            messagebox.showinfo("Success", f"Saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def run(self):
        if self._owns_root:
            self.root.mainloop()


if __name__ == "__main__":
    print("="*50)
    print("Point Cloud Normal Computer")
    print("Formats: .ply .pcd .xyz .txt .pts .las")
    print("="*50)
    app = NormalComputerGUI()
    app.run()
