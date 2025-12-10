# Rock Mass Point Cloud & Classification Tools

This repository collects small utilities for rock mass characterization from 3D point clouds and photogrammetry-derived data. The scripts cover normal estimation, discontinuity set clustering, spacing and roughness calculations, trace detection, and classic rock mass classifications (Q, RMR, RMi).

## Scripts

- **`rock_class_inputs_gui.py`**  
  Tkinter GUI to compute Q-system, RMR and RMi classifications. You enter basic field parameters and the app returns index values and corresponding rock mass classes.

- **`normal_computation.py`**  
  GUI tool for point cloud normal computation. Supports multiple file formats and methods (e.g. KNN PCA / robust methods), offers Open3D visualization, stereonet plots and export of normals/dip tables.

- **`rock_cluster.py`**  
  Clusters pre-normalized point clouds into discontinuity sets using MeanShift or KMeans on normals, 2D DBSCAN on local coordinates, and RANSAC plane fitting. Exports planes/labels, includes a PyVista/Open3D viewer and optional stereonet plots.

- **`spacing_calculator.py`**  
  Discontinuity spacing calculator. Reads RANSAC plane CSVs (e.g. from CloudCompare), builds simple rectangular planes, projects them onto a reference plane and computes spacing statistics; includes basic 3D plotting. Needs RANSAC planes .csv files from CloudCompare.

- **`roughness_calculator_1.py`**  
  Jr roughness calculator for a whole set of facets. Needs FACETS.csv file from CloudCompare. Reads a FACETS.csv, computes JRC/JRC20 from RMS and area using the Oppikofer formula, converts to Jr, prints statistics and saves plots and CSV summaries.

- **`roughness_calculator_2.py`**  
  Jr calculator with optional per-plane / per-spot analysis. Similar JRC/JRC20 → Jr workflow, but can group facets by RANSAC “spots” (orientation), outputting facet-level results and per-spot summaries plus histograms. Needs RANSAC and FACETS .csv files from CloudCompare.

- **`trace_edge_1.py`**  
  Edge-based point cloud splitter using curvature. Estimates normals and curvature from KNN neighborhoods, lets you interactively pick a curvature CDF threshold (with Open3D preview), and exports separate edge and non-edge PLY point clouds. 

- **`trace_edge_2.py`**  
  Fourier-based ridge/valley trace detection. Builds local PCA frames, computes curvature via truncated Fourier series along local profiles, selects ridge/valley points, smooths and thins them, then grows polylines and exports trace points/lines and diagnostics; optional 3D preview. The input is the edge_only .ply point clouds from trace_edge_1.py

## Installation

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
