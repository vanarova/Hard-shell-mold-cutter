# Mold Liner — STL Viewer

Desktop application for viewing STL files and slicing them into multiple pieces.

## Features

- **View STL files** — interactive 3D viewport with orbit, pan, and zoom
- **Triangle reduction** — decimate a model before slicing and save as `name_reduced.stl`
- **Z-axis slicing** — split a model into horizontal layers by height (mm)
- **Equidistant boundaries** — per-slice offset wall using current slice XY and previous-layer Z stack
- **Mold negative** — subtract the stitched boundary from an oversized block to produce a cavity STL
- **Organized output** — slices are saved as `1.stl`, `2.stl`, … in a folder named after the STL file; boundaries go in a `boundaries_v2` subfolder

## Requirements

- Python 3.10+
- Windows, macOS, or Linux

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Usage

1. Click **Open STL…** and select a `.stl` file.
2. Inspect the model in the 3D viewer (drag to rotate, scroll to zoom).
3. Optional: under **Prepare for Slicing**, set how much to reduce triangles (%) and click **Reduce & Save**. This writes `my_part_reduced.stl` next to the original and loads it for slicing.
4. Enter the slice height in millimeters (default 10 mm).
5. Click **Slice into Pieces**.

Sliced files are written next to the original STL:

```
my_part.stl
my_part/
  1.stl
  2.stl
  3.stl
  ...
```

The final slice may be shorter than the requested height if the model height is not an exact multiple.

### Boundaries

1. Slice the model first.
2. Enter the **Offset distance (mm)** and ensure **Height per slice (mm)** matches your slice settings.
3. Click **Create Boundaries**.

Each boundary is a thin wall (0.1 mm) offset in XY from the current slice footprint. Z placement uses the topmost point from the previous *x* slice layers, where *x* = boundary offset ÷ slice height.

Boundary files are saved as:

```
my_part/
  1.stl
  2.stl
  boundaries_v2/
    1.stl
    2.stl
    ...
```

### Stitch boundaries

1. Create slices and boundaries first.
2. Click **Stitch Boundaries**.

The tool reads `boundaries_v2/1.stl`, `2.stl`, … in order and lofts them into one unified watertight shell with shared rings at each layer interface.

Output file:

```
my_part/boundaries_v2/stitched.stl
```

### Mold negative

1. Stitch boundaries first.
2. Set **Block extra size x (mm)** — the box will be x mm longer, wider, and taller than the stitched model.
3. Click **Create Mold Negative**.

The tool builds an axis-aligned box around the stitched model, subtracts the stitch, and saves the cavity:

```
my_part/boundaries_v2/negative.stl
```
