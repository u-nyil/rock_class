# Edge-based point cloud splitting using curvature
# Interactive CDF threshold selection with Open3D preview
# References: "Lu, G., Cao, B., Zhu, X., Lin, Z., Bai, D., Tao, C., & Li, Y. (2024). 
            #Identification of rock mass discontinuity from 3D point clouds 
            #using improved fuzzy C-means and convolutional neural network.
#https://github.com/rockslopeworking/Rockmass-discontinuity

import os, sys, time, argparse
import numpy as np

def try_pick_file_gui():
    """Open file picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        fname = filedialog.askopenfilename(
            title="Select point cloud",
            filetypes=[("All files", "*.*"), ("PCD", "*.pcd"), ("Text", "*.txt *.csv *.xyz *.pts")]
        )
        try: root.destroy()
        except: pass
        return fname if fname else None
    except:
        return None

def load_pointcloud_any(path):
    """Load point cloud from PCD or text file."""
    import open3d as o3d
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pcd":
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError("PCD loaded empty.")
        return np.asarray(pcd.points, dtype=np.float64), True
    else:
        arr = np.loadtxt(path, dtype=float, ndmin=2)
        if arr.shape[1] < 3:
            raise ValueError("Need at least 3 columns (XYZ).")
        return arr[:, :3].astype(np.float64), False

def compute_knn_indices(xyz, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    return nbrs.kneighbors(xyz)

def pca_normal_and_curvature(Q):
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = (Qc.T @ Qc) / max(Q.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    normal = vecs[:, 0]
    nrm = np.linalg.norm(normal)
    if nrm > 0:
        normal = normal / nrm
    s = (vals[0] / max(vals.sum(), np.finfo(float).eps)) * 100.0
    return normal, s

def compute_normals_curvature_parallel(xyz, knn_idx):
    from joblib import Parallel, delayed
    m = xyz.shape[0]
    def per_i(i):
        return pca_normal_and_curvature(xyz[knn_idx[i, :], :])
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(per_i)(i) for i in range(m))
    normals = np.vstack([r[0] for r in results])
    curv = np.array([r[1] for r in results], dtype=np.float64)
    return normals, curv

def flip_normals_toward_sensor(normals, sensor_center=np.array([1.0, 1.0, 1.0])):
    sc = sensor_center / np.linalg.norm(sensor_center)
    dots = normals @ sc
    out = normals.copy()
    out[dots < 0] *= -1.0
    return out

def jet_colormap01(t):
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0*t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0*t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0*t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)

def colorize_by_scalar(values, vmin=None, vmax=None):
    v = values.astype(np.float64)
    if vmin is None: vmin = float(np.nanmin(v))
    if vmax is None: vmax = float(np.nanmax(v))
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-9
    t = (v - vmin) / (vmax - vmin)
    return jet_colormap01(t)

def preview_point_cloud(xyz, scalars):
    import open3d as o3d
    max_preview = 200_000
    if xyz.shape[0] > max_preview:
        step = max(1, xyz.shape[0] // max_preview)
        xyz, scalars = xyz[::step], scalars[::step]

    colors = colorize_by_scalar(scalars)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Preview (close to continue)",
                                       width=1280, height=720, left=60, top=60)

def export_two_plys(out_base, xyz, normals, edge_mask):
    import open3d as o3d
    noedge_mask = ~edge_mask

    def mk(x, n=None):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(x)
        if n is not None:
            p.normals = o3d.utility.Vector3dVector(n)
        return p

    pc_edge = mk(xyz[edge_mask], normals[edge_mask] if normals is not None else None)
    pc_noedge = mk(xyz[noedge_mask], normals[noedge_mask] if normals is not None else None)

    edge_out = f"{out_base}_edge_only.ply"
    noedge_out = f"{out_base}_no_edge.ply"

    o3d.io.write_point_cloud(edge_out, pc_edge, write_ascii=False, compressed=False, print_progress=True)
    o3d.io.write_point_cloud(noedge_out, pc_noedge, write_ascii=False, compressed=False, print_progress=True)

    print(f"[OK] Edge points: {edge_out}")
    print(f"[OK] Non-edge points: {noedge_out}")

def interactive_cdf_loop(pcData, curvature, show_preview=True, default_cdf=0.85):
    def ask_float(prompt, default=None):
        while True:
            s = input(prompt).strip()
            if not s and default is not None:
                return default
            try:
                return float(s)
            except:
                print("Please enter a number, e.g., 0.85")

    cdfvalue = ask_float("Input CDF threshold (0.8-0.9 recommended): ", default_cdf)
    s = curvature
    
    while True:
        r1 = float(np.quantile(s, cdfvalue))
        kept_mask = s <= r1
        
        if show_preview:
            kept_xyz = pcData[kept_mask, 0:3]
            kept_s = s[kept_mask]
            print(f"[Preview] {kept_xyz.shape[0]} kept points (threshold={r1:.6f})...")
            preview_point_cloud(kept_xyz, kept_s)
            ans = input("Acceptable? [y]es/[n]o: ").strip().lower()
            if ans in ("y", "yes"):
                return r1, kept_mask
            elif ans in ("n", "no"):
                cdfvalue = ask_float("Enter new CDF value: ", cdfvalue)
            else:
                print("Please answer 'y' or 'n'.")
        else:
            print(f"[No-Preview] Using r1={r1:.6f} at CDF={cdfvalue}")
            return r1, kept_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="Input point cloud path")
    parser.add_argument("--k", type=int, default=40, help="k for k-NN (20-40 recommended)")
    parser.add_argument("--no-preview", action="store_true", help="Skip interactive preview")
    parser.add_argument("--cdf", type=float, default=None, help="CDF value for non-interactive mode")
    args = parser.parse_args()

    # Load file
    if args.path and os.path.isfile(args.path):
        in_path = args.path
    else:
        print("[Info] No --path provided. Opening file picker...")
        in_path = try_pick_file_gui()
        if not in_path:
            in_path = input("Enter path to point cloud: ").strip('"').strip()
            if not in_path or not os.path.isfile(in_path):
                print("[Error] File not found.")
                sys.exit(1)

    in_dir, in_name = os.path.split(in_path)
    name_noext = os.path.splitext(in_name)[0]
    print(f"[Load] {in_path}")
    xyz, _ = load_pointcloud_any(in_path)
    m = xyz.shape[0]
    print(f"[Load] {m} points")

    pcData = np.zeros((m, 7), dtype=np.float64)
    pcData[:, 0:3] = xyz

    # Compute features
    t0 = time.time()
    k = int(args.k)
    print(f"[KNN] Building index (k={k})...")
    distances, idx = compute_knn_indices(pcData[:, 0:3], k)
    print("[PCA] Computing normals + curvature...")
    normals, curvature = compute_normals_curvature_parallel(pcData[:, 0:3], idx)
    pcData[:, 3:6] = normals
    pcData[:, 6] = curvature
    print(f"[Timing] {time.time() - t0:.2f}s")

    print("[Normals] Flipping toward sensor...")
    pcData[:, 3:6] = flip_normals_toward_sensor(pcData[:, 3:6])

    print("[Trim] Interactive CDF selection...")
    use_preview = not args.no_preview
    default_cdf = 0.85 if args.cdf is None else float(args.cdf)
    r1, kept_mask = interactive_cdf_loop(pcData, pcData[:, 6], show_preview=use_preview, default_cdf=default_cdf)
    edge_mask = pcData[:, 6] > r1
    print(f"[Trim] Threshold={r1:.6f} | kept={np.count_nonzero(kept_mask)} | edges={np.count_nonzero(edge_mask)}")

    out_base = os.path.join(in_dir, name_noext)
    print("[Export] Writing PLYs...")
    export_two_plys(out_base, pcData[:, 0:3], pcData[:, 3:6], edge_mask)
    print("[Done]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
