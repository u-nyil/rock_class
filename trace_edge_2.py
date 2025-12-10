# Fourier-based trace detection for point clouds
# Extracts ridge/valley lines using truncated Fourier series curvature

from __future__ import annotations
import argparse, os, sys
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from tqdm import tqdm

def pick_file_via_dialog():
    """Open file picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except:
        return None
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    filetypes = [
        ("Point clouds", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts *.txt"),
        ("All files", "*.*"),
    ]
    path = filedialog.askopenfilename(title="Select point cloud", filetypes=filetypes,
                                       initialdir=os.path.expanduser("~"))
    root.destroy()
    return path if path else None

def load_point_cloud(path):
    ext = path.lower().split(".")[-1]
    if ext in ("ply", "pcd", "xyz", "xyzn", "xyzrgb", "pts", "txt"):
        pcd = o3d.io.read_point_cloud(path)
        if len(np.asarray(pcd.points)) == 0 and ext == "txt":
            arr = np.loadtxt(path)
            if arr.shape[1] >= 3:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr[:, :3]))
        return pcd
    raise ValueError(f"Unsupported format: {ext}")

def save_points(path, xyz, colors=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)

def estimate_average_spacing(xyz, k=2):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(xyz)
    dists, _ = nbrs.kneighbors(xyz)
    return float(np.mean(dists[:, 1]))

def pca_directions(xyz, k=40):
    tree = cKDTree(xyz)
    eigvecs = np.zeros((len(xyz), 3, 3))
    for i in tqdm(range(len(xyz)), desc="PCA", leave=False):
        _, idxs = tree.query(xyz[i], k=k)
        pts = xyz[idxs]
        pts_centered = pts - pts.mean(0)
        C = pts_centered.T @ pts_centered / max(len(pts)-1, 1)
        _, V = np.linalg.eigh(C)
        eigvecs[i] = V
    return eigvecs

def propagate_normal_orientation(xyz, normals, radius):
    tree = cKDTree(xyz)
    N = len(xyz)
    visited = np.zeros(N, dtype=bool)
    q = [0]
    visited[0] = True
    while q:
        i = q.pop()
        idxs = tree.query_ball_point(xyz[i], r=radius)
        for j in idxs:
            if j == i: continue
            if np.dot(normals[i], normals[j]) < 0:
                normals[j] = -normals[j]
            if not visited[j]:
                visited[j] = True
                q.append(j)
    return normals

@dataclass
class LocalFrame:
    R: np.ndarray

def local_frames_from_pca(eigs):
    e1, e2, e3 = eigs[:, :, 0], eigs[:, :, 1], eigs[:, :, 2]
    return np.stack([e3, e2, e1], axis=2)

def to_local(xyz, origin, R):
    return (xyz - origin) @ R

def idw_four_quadrant_sample(local_pts, ul, axis, eps=1e-9):
    """Interpolate z at (u=ul, v=0) using 4-quadrant nearest neighbor IDW."""
    x, y, z = local_pts[:, 0], local_pts[:, 1], local_pts[:, 2]
    target = np.array([ul, 0.0])

    def pick(mask):
        if not np.any(mask): return None
        pts2 = np.column_stack([x[mask], y[mask]])
        d = np.linalg.norm(pts2 - target, axis=1)
        j = np.argmin(d)
        return np.where(mask)[0][j]

    tol_y = np.percentile(np.abs(y), 20)
    if tol_y == 0: tol_y = np.std(y) + 1e-6
    near_y = np.abs(y) <= tol_y

    m1 = (x <= ul) & near_y
    m2 = (x <= ul) & ~near_y
    m3 = (x > ul) & near_y
    m4 = (x > ul) & ~near_y

    q_idxs = [pick(m1), pick(m2), pick(m3), pick(m4)]
    chosen = [i for i in q_idxs if i is not None]
    
    if len(chosen) < 4:
        all_idx = np.arange(len(x))
        remaining = [i for i in all_idx if i not in chosen]
        if remaining:
            pts2 = np.column_stack([x[remaining], y[remaining]])
            d = np.linalg.norm(pts2 - target, axis=1)
            order = np.argsort(d)
            for k in order:
                chosen.append(remaining[k])
                if len(chosen) == 4: break

    chosen = chosen[:4]
    pts = np.column_stack([x[chosen], y[chosen]])
    dz = z[chosen]
    d = np.linalg.norm(pts - target, axis=1)
    w = 1.0 / (d + eps)
    return float(np.sum(w * dz) / np.sum(w))

def fft_coefficients_1d(values):
    M = len(values)
    F = np.fft.rfft(values)
    a0 = (1.0/M) * np.real(F[0])
    an = (2.0/M) * np.real(F[1:])
    bn = (2.0/M) * -np.imag(F[1:])
    return a0, an, bn

def curvature_midpoint_from_truncated_series(an1, bn1, L):
    fp = bn1 / (L + 1e-12)
    fpp = -an1 / ((L + 1e-12)**2)
    return fpp / (1.0 + fp*fp)**1.5

def compute_curvatures_mean(xyz, frames, k_neighbors, M_samples, step_len):
    N = len(xyz)
    tree = cKDTree(xyz)
    H = np.zeros(N, dtype=float)
    L = (M_samples//2) * step_len
    samples = np.linspace(-L, L, M_samples)

    for i in tqdm(range(N), desc="Fourier curvature", leave=False):
        _, idxs = tree.query(xyz[i], k=k_neighbors)
        pts = xyz[idxs]
        R = frames[i]
        loc = to_local(pts, xyz[i], R)

        g = np.array([idw_four_quadrant_sample(loc, ul, axis='x') for ul in samples], dtype=float)
        _, an, bn = fft_coefficients_1d(g)
        a1 = an[0] if len(an) > 0 else 0.0
        b1 = bn[0] if len(bn) > 0 else 0.0
        kx = curvature_midpoint_from_truncated_series(a1, b1, 2*L)

        loc_y = loc[:, [1, 0, 2]]
        gy = np.array([idw_four_quadrant_sample(loc_y, ul, axis='x') for ul in samples], dtype=float)
        _, any_, bny = fft_coefficients_1d(gy)
        a1y = any_[0] if len(any_) > 0 else 0.0
        b1y = bny[0] if len(bny) > 0 else 0.0
        ky = curvature_midpoint_from_truncated_series(a1y, b1y, 2*L)

        H[i] = 0.5 * (kx + ky)

    return H

def select_potential_trace_points(H, tau=0.05):
    ridge_mask = H < -tau
    valley_mask = H > tau
    return ridge_mask, valley_mask

def curvature_weighted_laplacian(xyz, H, mask, radius=0.7, iterations=3):
    if not np.any(mask):
        return np.array([], dtype=int), np.zeros((0, 3))
    idxs = np.where(mask)[0]
    pts = xyz[idxs].copy()
    tree_full = cKDTree(xyz)
    
    for _ in range(iterations):
        new_pts = pts.copy()
        for j, pt in enumerate(pts):
            neighbors = tree_full.query_ball_point(pt, r=radius)
            if len(neighbors) < 2: continue
            nbr_pts = xyz[neighbors]
            w = np.abs(H[neighbors]) + 1e-6
            w /= w.sum()
            new_pts[j] = (nbr_pts * w[:, None]).sum(axis=0)
        pts = new_pts
    return idxs, pts

def distance_thinning(idxs, pts, H, avg_d, factor=0.5):
    if len(pts) == 0:
        return np.array([], dtype=int), np.zeros((0, 3))
    tree = cKDTree(pts)
    keep = np.ones(len(pts), dtype=bool)
    radius = factor * avg_d
    
    for i in range(len(pts)):
        if not keep[i]: continue
        neighbors = tree.query_ball_point(pts[i], r=radius)
        for j in neighbors:
            if j != i and keep[j]:
                if abs(H[idxs[j]]) < abs(H[idxs[i]]):
                    keep[j] = False
                    
    return idxs[keep], pts[keep]

def line_growing(thin_xyz, thin_idx, H, radius, min_pts=5):
    alpha, beta, gamma = 1.0, 0.5, 0.2
    used = np.zeros(len(thin_xyz), dtype=bool)
    tree = cKDTree(thin_xyz)
    lines = []
    
    for seed in range(len(thin_xyz)):
        if used[seed]: continue
        line = [seed]
        used[seed] = True
        direction = None
        
        while True:
            i = line[-1]
            idxs = tree.query_ball_point(thin_xyz[i], r=radius)
            idxs = [j for j in idxs if j != i and not used[j]]
            if not idxs: break
            
            best_j, best_score, best_v = None, -1e9, None
            for j in idxs:
                v = thin_xyz[j] - thin_xyz[i]
                dist = np.linalg.norm(v) + 1e-9
                v = v / dist
                ang_pen = 0.0
                if direction is not None:
                    cosang = np.clip(np.dot(direction, v), -1.0, 1.0)
                    ang_pen = np.arccos(cosang)
                score = alpha*abs(H[thin_idx[j]]) - beta*ang_pen - gamma*dist
                if score > best_score:
                    best_score, best_j, best_v = score, j, v
                    
            if best_j is None: break
            line.append(best_j)
            used[best_j] = True
            direction = best_v
            
        if len(line) >= min_pts:
            lines.append(np.array(line, dtype=int))
            
    return [thin_xyz[idxs] for idxs in lines]

def make_lineset_from_polylines(polylines, color=(1.0, 0.6, 0.0)):
    points_list, lines = [], []
    offset = 0
    for poly in polylines:
        n = len(poly)
        if n < 2: continue
        points_list.append(poly)
        lines.extend([[offset+i, offset+i+1] for i in range(n-1)])
        offset += n
    if not points_list:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        return ls
    pts = np.vstack(points_list)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="Input point cloud")
    ap.add_argument("--nogui", action="store_true", help="Disable file dialog")
    ap.add_argument("--preview", action="store_true", help="Open 3D preview")
    ap.add_argument("--k_pca", type=int, default=40)
    ap.add_argument("--normal_radius", type=float, default=0.25)
    ap.add_argument("--k_curv", type=int, default=50)
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--step_factor", type=float, default=0.8)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--lap_radius", type=float, default=0.7)
    ap.add_argument("--grow_radius", type=float, default=0.4)
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    inp_path = args.inp
    if inp_path is None and not args.nogui:
        inp_path = pick_file_via_dialog()
        if not inp_path:
            print("No file selected.", file=sys.stderr)
            sys.exit(1)
    if inp_path is None:
        ap.error("Provide --in <path> or use file picker.")

    pcd = load_point_cloud(inp_path)
    xyz = np.asarray(pcd.points, dtype=float)
    if xyz.shape[0] < 10:
        raise RuntimeError("Not enough points.")

    base = args.out_prefix or inp_path.rsplit(".", 1)[0]
    avg_d = estimate_average_spacing(xyz)
    step_len = args.step_factor * avg_d

    eigs = pca_directions(xyz, k=args.k_pca)
    normals = eigs[:, :, 0].copy()
    normals = propagate_normal_orientation(xyz, normals, radius=args.normal_radius)
    frames = local_frames_from_pca(eigs)

    if args.samples % 2 == 0:
        args.samples += 1
    H = compute_curvatures_mean(xyz, frames, args.k_curv, args.samples, step_len)

    ridge_mask, valley_mask = select_potential_trace_points(H, tau=args.threshold)

    if ridge_mask.any():
        save_points(f"{base}_trace_points_ridge.ply", xyz[ridge_mask], np.tile([1, 0, 0], (ridge_mask.sum(), 1)))
    if valley_mask.any():
        save_points(f"{base}_trace_points_valley.ply", xyz[valley_mask], np.tile([0, 0, 1], (valley_mask.sum(), 1)))

    ridx, ridge_smooth = curvature_weighted_laplacian(xyz, H, ridge_mask, radius=args.lap_radius)
    vidx, valley_smooth = curvature_weighted_laplacian(xyz, H, valley_mask, radius=args.lap_radius)

    ridx_thin, ridge_thin = distance_thinning(ridx, ridge_smooth, H, avg_d=avg_d)
    vidx_thin, valley_thin = distance_thinning(vidx, valley_smooth, H, avg_d=avg_d)

    if len(ridge_thin):
        save_points(f"{base}_trace_points_ridge_thin.ply", ridge_thin, np.tile([0, 1, 0], (len(ridge_thin), 1)))
    if len(valley_thin):
        save_points(f"{base}_trace_points_valley_thin.ply", valley_thin, np.tile([1, 0, 1], (len(valley_thin), 1)))

    ridge_lines = line_growing(ridge_thin, ridx_thin, H, radius=args.grow_radius)
    valley_lines = line_growing(valley_thin, vidx_thin, H, radius=args.grow_radius)

    def save_lines_as_segments(lines, path, color):
        seg_points, seg_colors = [], []
        for L in lines:
            if len(L) < 2: continue
            for i in range(len(L)-1):
                seg_points.extend([L[i], L[i+1]])
                seg_colors.extend([color, color])
        if seg_points:
            save_points(path, np.vstack(seg_points), np.vstack(seg_colors))

    save_lines_as_segments(ridge_lines, f"{base}_trace_lines_ridge.ply", np.array([1, 0.6, 0]))
    save_lines_as_segments(valley_lines, f"{base}_trace_lines_valley.ply", np.array([0, 1, 1]))

    np.savez(f"{base}_diagnostics.npz", xyz=xyz, H=H, ridge_idx=ridx_thin, ridge_points=ridge_thin,
             valley_idx=vidx_thin, valley_points=valley_thin, avg_spacing=avg_d)

    if args.preview:
        geoms = []
        pcd_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pcd_vis.paint_uniform_color([0.7, 0.7, 0.7])
        geoms.append(pcd_vis)

        if len(ridge_thin):
            p_ridge = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ridge_thin))
            p_ridge.paint_uniform_color([1.0, 0.0, 0.0])
            geoms.append(p_ridge)
        if len(valley_thin):
            p_valley = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valley_thin))
            p_valley.paint_uniform_color([0.0, 0.0, 1.0])
            geoms.append(p_valley)

        if ridge_lines:
            geoms.append(make_lineset_from_polylines(ridge_lines, color=(1.0, 0.6, 0.0)))
        if valley_lines:
            geoms.append(make_lineset_from_polylines(valley_lines, color=(0.0, 1.0, 1.0)))

        o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
