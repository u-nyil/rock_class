# Discontinuity Spacing Calculator
# Projects planes onto reference plane perpendicular to mean normal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def pick_csv_files():
    """Select RANSAC plane CSV files via dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select RANSAC planes CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        return [Path(p) for p in paths]
    except Exception as e:
        print(f"File picker error: {e}")
        p = input("Type path to RANSAC CSV: ").strip()
        return [Path(p)] if p else []


class SpacingAnalyzer:
    def __init__(self):
        self.planes_data = {}
        self.results = {}

    def load_ransac_csv(self, filepath, set_name=None):
        """Load RANSAC plane data from CloudCompare CSV."""
        filepath = Path(filepath)
        set_name = set_name or filepath.stem

        df = pd.read_csv(filepath, delimiter=";")
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all")

        planes = []
        for idx, row in df.iterrows():
            center = np.array([row["Cx"], row["Cy"], row["Cz"]])
            normal = np.array([row["Nx"], row["Ny"], row["Nz"]])
            normal = normal / np.linalg.norm(normal)
            w = row.get("Width", 1.0)
            h = row.get("Height", 1.0)
            corners = self._make_corners(center, normal, w, h)
            planes.append({
                "id": idx + 1, "center": center, "normal": normal,
                "dip": row["Dip"], "dip_direction": row["Dip dir"],
                "width": w, "height": h, "corners": corners
            })

        self.planes_data[set_name] = {
            "planes": planes, "df": df,
            "mean_dip": df["Dip"].mean(),
            "mean_dip_dir": df["Dip dir"].mean()
        }
        print(f"Loaded {len(planes)} planes from '{set_name}'")
        return planes

    def _make_corners(self, center, normal, w, h):
        """Create 4 corners for rectangular plane."""
        up = np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0])
        v1 = np.cross(normal, up)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        hw, hh = w/2, h/2
        return np.array([
            center - v1*hw - v2*hh, center + v1*hw - v2*hh,
            center + v1*hw + v2*hh, center - v1*hw + v2*hh
        ])

    def _mean_normal(self, planes):
        normals = np.array([p["normal"] for p in planes])
        mn = np.mean(normals, axis=0)
        return mn / np.linalg.norm(mn)

    def _rotation_matrix(self, mean_normal):
        """Rotation to align mean normal with Z-axis."""
        z = np.array([0, 0, 1])
        if np.allclose(mean_normal, z) or np.allclose(mean_normal, -z):
            return np.eye(3)
        axis = np.cross(mean_normal, z)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(mean_normal, z), -1, 1))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*np.dot(K, K)

    def calculate_spacing(self, set_name):
        """Calculate spacing using plane projections."""
        if set_name not in self.planes_data:
            print(f"Set '{set_name}' not loaded!")
            return None

        data = self.planes_data[set_name]
        planes = data["planes"]
        if len(planes) < 2:
            print("Need at least 2 planes!")
            return None

        mean_normal = self._mean_normal(planes)
        R = self._rotation_matrix(mean_normal)
        ref = np.mean([p["center"] for p in planes], axis=0)

        # Project planes to 2D
        projected = []
        for p in planes:
            rot = np.array([np.dot(R, c - ref) for c in p["corners"]])
            corners_2d = rot[:, :2]
            z_pos = np.mean(rot[:, 2])
            projected.append({
                "id": p["id"], "corners_3d": p["corners"], "corners_rotated": rot,
                "corners_2d": corners_2d, "center_2d": np.mean(corners_2d, axis=0),
                "z_position": z_pos, "original": p
            })

        projected.sort(key=lambda x: x["z_position"])

        # Calculate spacings
        spacings = []
        details = []
        for i in range(len(projected) - 1):
            p1, p2 = projected[i], projected[i + 1]
            sp = abs(p2["z_position"] - p1["z_position"])
            spacings.append(sp)
            details.append({
                "from_plane": p1["id"], "to_plane": p2["id"],
                "spacing": sp, "from_z": p1["z_position"], "to_z": p2["z_position"]
            })

        sp_arr = np.array(spacings) if spacings else np.array([])
        results = {
            "set_name": set_name, "n_planes": len(planes),
            "mean_dip": data["mean_dip"], "mean_dip_direction": data["mean_dip_dir"],
            "mean_normal": mean_normal, "rotation_matrix": R, "reference_point": ref,
            "projected_planes": projected, "spacing_details": details, "spacings": sp_arr,
            "mean_spacing": np.mean(sp_arr) if len(sp_arr) else 0,
            "std_spacing": np.std(sp_arr) if len(sp_arr) else 0,
            "min_spacing": np.min(sp_arr) if len(sp_arr) else 0,
            "max_spacing": np.max(sp_arr) if len(sp_arr) else 0
        }
        self.results[set_name] = results
        return results

    def plot_3d_view(self, set_name, figsize=(14, 10), save_path=None):
        """3D visualization of planes."""
        if set_name not in self.results:
            self.calculate_spacing(set_name)
        r = self.results[set_name]
        planes = self.planes_data[set_name]["planes"]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        colors = plt.cm.Set2(np.linspace(0, 1, len(planes)))

        for i, p in enumerate(planes):
            corners = p["corners"]
            verts = [list(zip(corners[:, 0], corners[:, 1], corners[:, 2]))]
            poly = Poly3DCollection(verts, alpha=0.6)
            poly.set_facecolor(colors[i])
            poly.set_edgecolor("black")
            ax.add_collection3d(poly)
            ax.text(p["center"][0], p["center"][1], p["center"][2],
                    f"P{p['id']}\n{p['dip']:.0f}/{p['dip_direction']:.0f}", fontsize=9, ha="center")
            n = p["normal"] * 0.3
            ax.quiver(p["center"][0], p["center"][1], p["center"][2], n[0], n[1], n[2],
                      color="red", alpha=0.7, linewidth=2)

        # Spacing lines
        for d in r["spacing_details"]:
            p1 = next(p for p in planes if p["id"] == d["from_plane"])
            p2 = next(p for p in planes if p["id"] == d["to_plane"])
            ax.plot([p1["center"][0], p2["center"][0]], [p1["center"][1], p2["center"][1]],
                    [p1["center"][2], p2["center"][2]], "g--", lw=2, alpha=0.8)
            mid = (p1["center"] + p2["center"]) / 2
            ax.text(mid[0], mid[1], mid[2], f"{d['spacing']*100:.1f} cm",
                    fontsize=10, color="green", fontweight="bold")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3D View: {set_name}\nMean Spacing: {r['mean_spacing']*100:.1f} cm")

        all_c = np.vstack([p["corners"] for p in planes])
        rng = np.max(np.ptp(all_c, axis=0)) / 2
        mid = np.mean(all_c, axis=0)
        ax.set_xlim(mid[0]-rng, mid[0]+rng)
        ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        return fig, ax

    def plot_2d_projection(self, set_name, figsize=(12, 10), save_path=None, n_scanlines=10):
        """2D trace map projection."""
        if set_name not in self.results:
            self.calculate_spacing(set_name)
        r = self.results[set_name]

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, len(r["projected_planes"])))

        all_c = np.vstack([pp["corners_2d"] for pp in r["projected_planes"]])

        # Scanlines
        y_min, y_max = all_c[:, 1].min(), all_c[:, 1].max()
        y_rng = y_max - y_min
        for y in np.linspace(y_min - y_rng*0.2, y_max + y_rng*0.2, n_scanlines):
            ax.axhline(y=y, color="grey", lw=0.8, alpha=0.5, zorder=1)

        for i, pp in enumerate(r["projected_planes"]):
            poly = plt.Polygon(pp["corners_2d"], fill=True, facecolor=colors[i],
                               edgecolor="black", lw=2, alpha=0.7, zorder=2)
            ax.add_patch(poly)
            orig = pp["original"]
            ax.annotate(f"P{pp['id']}\n{orig['dip']:.0f}/{orig['dip_direction']:.0f}",
                        pp["center_2d"], fontsize=9, ha="center", va="center", fontweight="bold", zorder=3)

        # Spacing arrows
        sorted_pp = sorted(r["projected_planes"], key=lambda x: x["z_position"])
        for i in range(len(sorted_pp) - 1):
            p1, p2 = sorted_pp[i], sorted_pp[i + 1]
            c1, c2 = p1["center_2d"], p2["center_2d"]
            sp = abs(p2["z_position"] - p1["z_position"])
            ax.annotate("", xy=c2, xytext=c1, arrowprops=dict(arrowstyle="<->", color="green", lw=2), zorder=4)
            mid = (c1 + c2) / 2
            ax.text(mid[0], mid[1], f"{sp*100:.1f} cm", fontsize=10, color="green",
                    fontweight="bold", ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"2D Projection: {set_name}\nMean Spacing: {r['mean_spacing']*100:.1f} cm ± {r['std_spacing']*100:.1f} cm")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        margin = 0.15
        ax.set_xlim(all_c[:, 0].min() - (all_c[:, 0].ptp())*margin, all_c[:, 0].max() + (all_c[:, 0].ptp())*margin)
        ax.set_ylim(all_c[:, 1].min() - (all_c[:, 1].ptp())*margin, all_c[:, 1].max() + (all_c[:, 1].ptp())*margin)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        return fig, ax

    def plot_combined_view(self, set_name, figsize=(16, 8), save_path=None):
        """Side-by-side 3D and 2D views."""
        if set_name not in self.results:
            self.calculate_spacing(set_name)
        r = self.results[set_name]
        planes = self.planes_data[set_name]["planes"]

        fig = plt.figure(figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, len(planes)))

        # 3D
        ax1 = fig.add_subplot(121, projection="3d")
        for i, p in enumerate(planes):
            corners = p["corners"]
            verts = [list(zip(corners[:, 0], corners[:, 1], corners[:, 2]))]
            poly = Poly3DCollection(verts, alpha=0.6)
            poly.set_facecolor(colors[i])
            poly.set_edgecolor("black")
            ax1.add_collection3d(poly)
            ax1.text(p["center"][0], p["center"][1], p["center"][2], f"P{p['id']}", fontsize=9, ha="center")

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title(f"3D View\n{r['n_planes']} planes")

        all_c = np.vstack([p["corners"] for p in planes])
        rng = np.max(np.ptp(all_c, axis=0)) / 2
        mid = np.mean(all_c, axis=0)
        ax1.set_xlim(mid[0]-rng, mid[0]+rng)
        ax1.set_ylim(mid[1]-rng, mid[1]+rng)
        ax1.set_zlim(mid[2]-rng, mid[2]+rng)

        # 2D
        ax2 = fig.add_subplot(122)
        all_2d = []
        for i, pp in enumerate(r["projected_planes"]):
            corners = pp["corners_2d"]
            all_2d.extend(corners.tolist())
            poly = plt.Polygon(corners, fill=True, facecolor=colors[i], edgecolor="black", lw=2, alpha=0.7)
            ax2.add_patch(poly)
            ax2.text(pp["center_2d"][0], pp["center_2d"][1], f"P{pp['id']}", fontsize=9,
                     ha="center", va="center", fontweight="bold")
        all_2d = np.array(all_2d)

        sorted_pp = sorted(r["projected_planes"], key=lambda x: x["z_position"])
        for i in range(len(sorted_pp) - 1):
            p1, p2 = sorted_pp[i], sorted_pp[i + 1]
            sp = abs(p2["z_position"] - p1["z_position"])
            c1, c2 = p1["center_2d"], p2["center_2d"]
            ax2.annotate("", xy=c2, xytext=c1, arrowprops=dict(arrowstyle="<->", color="green", lw=2))
            ax2.text((c1[0]+c2[0])/2, (c1[1]+c2[1])/2, f"{sp*100:.1f} cm",
                     fontsize=10, color="green", fontweight="bold", ha="center")

        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title(f"2D Projection\nMean Spacing: {r['mean_spacing']*100:.1f} cm")
        ax2.set_aspect("equal")
        ax2.grid(alpha=0.3)

        margin = 0.15
        ax2.set_xlim(all_2d[:, 0].min() - all_2d[:, 0].ptp()*margin, all_2d[:, 0].max() + all_2d[:, 0].ptp()*margin)
        ax2.set_ylim(all_2d[:, 1].min() - all_2d[:, 1].ptp()*margin, all_2d[:, 1].max() + all_2d[:, 1].ptp()*margin)

        plt.suptitle(f"Discontinuity Spacing: {set_name}\nDip: {r['mean_dip']:.1f}, Dir: {r['mean_dip_direction']:.1f}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        return fig

    def print_report(self, set_name):
        """Print spacing analysis report."""
        if set_name not in self.results:
            self.calculate_spacing(set_name)
        r = self.results[set_name]

        print("\n" + "="*50)
        print(f"SPACING ANALYSIS: {set_name}")
        print("="*50)
        print(f"Planes: {r['n_planes']}, Dip: {r['mean_dip']:.1f}, Dir: {r['mean_dip_direction']:.1f}")
        print("\nSpacings:")
        for d in r["spacing_details"]:
            print(f"  P{d['from_plane']} → P{d['to_plane']}: {d['spacing']*100:.1f} cm")
        print(f"\nStats: mean={r['mean_spacing']*100:.1f} cm, std={r['std_spacing']*100:.1f} cm")
        print(f"       min={r['min_spacing']*100:.1f} cm, max={r['max_spacing']*100:.1f} cm")


def main():
    analyzer = SpacingAnalyzer()

    print("Select RANSAC plane CSV files (CloudCompare export).")
    csv_paths = pick_csv_files()
    if not csv_paths:
        print("No files selected.")
        return analyzer

    for path in csv_paths:
        set_name = path.stem
        print(f"\n--- {set_name} ---")

        analyzer.load_ransac_csv(path, set_name=set_name)
        analyzer.calculate_spacing(set_name)
        analyzer.print_report(set_name)

        base = path.with_suffix("")
        analyzer.plot_3d_view(set_name, save_path=str(base) + "_3D.png")
        analyzer.plot_2d_projection(set_name, save_path=str(base) + "_2D.png")
        analyzer.plot_combined_view(set_name, save_path=str(base) + "_Combined.png")
        plt.close("all")

    print("\nDone.")
    return analyzer


if __name__ == "__main__":
    analyzer = main()
