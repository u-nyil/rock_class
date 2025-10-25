# Normals Viewer

Normals Viewer is a Python 3.12 toolkit that loads 3D point clouds, estimates and orients
surface normals, clusters orientations using DBSCAN on the unit sphere, and visualises the
results through an Open3D viewer or a PySide6 GUI.

## Features

- Load `.ply`, `.pcd`, `.xyz`, or text point clouds with optional normals.
- Estimate PCA normals (k-NN or radius) and orient them toward +Z, a viewpoint, or a
  consistent tangent plane.
- Convert normals to stereographic quantities (dip, dip-direction, trend, plunge) in ENU
  coordinates, enforcing lower-hemisphere poles.
- DBSCAN clustering on orientation space using the angular antipodal-safe metric with
  optional cone pruning.
- Open3D visualisation with cluster colour maps, optional noise suppression, and normal
  glyph overlays.
- PySide6 desktop GUI plus Typer CLI for headless workflows.
- CSV/JSON exports for labels, dip/dip-direction tables, and clustering summaries.

## Assumptions

- Coordinate system follows ENU: `x` = East, `y` = North, `z` = Up.
- Normals are treated as unit vectors; poles are forced onto the lower hemisphere.
- Angles are expressed in degrees; DBSCAN uses radians internally.
- Dip/dip-direction conversions follow the formulas specified in the project brief.

## Rationale

- **DBSCAN on poles**: orientations are mapped to unit poles on the lower hemisphere,
  providing a distance metric invariant to normal sign by using `arccos(|u·v|)`.
  DBSCAN can detect an unknown number of structural families without prespecifying
  cluster counts and is robust to noise/outliers.
- **Cone pruning**: optional angular filtering around each cluster mean removes stray
  members after clustering, sharpening families for stereonet analysis or mapping.

## Installation

Install the package in an isolated virtual environment on Windows 11 with Python 3.12.5.

```powershell
# 0) Confirm Python version
python --version  # should be Python 3.12.5

# 1) Create & activate venv
python -m venv .venv
. .venv\Scripts\activate

# 2) Install
pip install -U pip
pip install -e .

# 3) Run GUI
python -m normals_viewer

# 4) Example CLI (view & cluster)
normapp view --in .\data\sample.ply --show-normals on --every 20 --orient zup
normapp cluster --in .\data\sample.ply --eps-orient 12 --min-samples 50 --cone 15 --view
```

> `open3d>=0.18` provides wheels for Python 3.12; ensure the platform has the matching
> Visual C++ runtime installed.

## Usage

### GUI

```powershell
python -m normals_viewer
```

1. **Load** a point cloud (normals detected automatically).
2. **Ensure normals** to compute and orient as required.
3. **Run clustering** by setting DBSCAN parameters and reviewing summary statistics.
4. **Open viewer** to inspect clusters or normal glyphs.
5. **Export** labels, dip tables, and summaries for reporting.

### CLI

```powershell
# View with normals coloured by RGB
normapp view --in data\wall.ply --show-normals on --norm-scale 0.02 --every 10 --orient zup --color-mode normal

# Orientation DBSCAN + export
normapp cluster --in data\wall.ply --eps-orient 10 --min-samples 40 --cone 12 --view --out results\labels.csv --dip-out results\dip.csv --summary-out results\summary.json
```

## Testing

```powershell
pytest
```

The synthetic tests cover:

1. Two distinct plane families clustered with mean poles within 3° of ground truth.
2. Normal-to-pole conversion enforcing the lower hemisphere.
3. Normal glyph generation producing the expected number of line segments.

## Screenshots

Add GUI or viewer screenshots here once captured:

![GUI placeholder](docs/images/gui_placeholder.png)

## License

MIT
