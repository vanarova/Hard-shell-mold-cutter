# Mold Liner — Algorithm & Implementation Specification

**Version:** 1.0  
**Purpose:** Complete, code-generatable specification of every algorithm, data structure, file layout, and processing pipeline implemented by the Mold Liner STL desktop tool. A developer (or code generator) should be able to re-implement the system from this document alone.

**Scope:** Algorithms and geometry processing only. UI layout is summarized for workflow context; visual styling is out of scope.

---

## 1. System Overview

Mold Liner is a desktop application that:

1. Loads a triangular-mesh STL model.
2. Optionally decimates (reduces) triangle count.
3. Slices the model into numbered layer STLs along **X**, **Y**, or **Z**.
4. For each axis independently, generates **thin equidistant boundary walls** per slice layer using a multi-layer stack-depth rule.
5. **Stitches** per-axis boundary layers into one unified lofted shell per axis.
6. **Combines** the three axis shells (Z, X, Y) into one STL.
7. Optionally creates a **mold negative** (bounding box minus Z-axis stitched shell) for the Z axis only.

### 1.1 Technology Stack

| Component | Library | Role |
|-----------|---------|------|
| Mesh I/O & clipping | PyVista | `PolyData`, `clip_box`, `slice`, `decimate_pro`, STL save/load |
| 2D geometry | Shapely | Footprints, offsets, polygonize, distance queries |
| 3D mesh assembly | trimesh | Wall mesh, loft caps, boolean ops, concatenate |
| Boolean subtraction | manifold3d (via trimesh) | `engine="manifold"` for box − stitch |
| Triangulation | mapbox-earcut (via trimesh) | Cap triangulation |
| UI | PyQt6 + pyvistaqt | Not specified in algorithm detail |

### 1.2 Module Map (Reference Implementation)

```
src/app/
  slice_axis.py           — Axis enum, paths, coordinate remapping
  mesh_slicer.py          — Axis-aware slicing
  mesh_reducer.py         — Triangle decimation
  boundary_generator.py   — Footprint, offset outline, slice file listing
  boundary_generator_v2.py — Per-slice boundary walls + validation
  boundary_stitcher.py    — Unified lofted stitch per axis
  model_combiner.py       — Merge Z/X/Y stitched models
  stitched_negative.py    — Mold cavity via boolean subtract
  main_window.py          — UI orchestration
  stl_viewer.py           — 3D viewport
  footer_bar.py             — Progress + view mode
```

---

## 2. Global Constants

```python
BOUNDARY_WALL_THICKNESS_MM = 0.1   # Physical thickness of generated boundary wall mesh
_LOOP_POINTS = 128                   # Boundary outline resampling count (boundary v2)
_LOOP_SEGMENTS = 256                 # Stitch ring resampling count
_SLICE_NAME_RE = r"^\d+$"            # Valid slice/boundary filenames: 1.stl, 2.stl, ...
_EXCLUDED_BOUNDARY_STEMS = {"stitched", "stitched_v2", "negative"}
```

### 2.1 User Parameters

| Parameter | Symbol | Valid range | Default | Used by |
|-----------|--------|-------------|---------|---------|
| Slice thickness | `slice_height_mm` | > 0 | 10.0 mm | Slicing, boundary stack depth |
| Boundary offset | `offset_mm` | > 0 | 5.0 mm | Boundary generation |
| Reduce percent | `reduction_percent` | (0, 100) exclusive | 50% | Mesh reduction |
| Block extra size | `margin_mm` | > 0 | 10.0 mm | Mold negative |

---

## 3. Slice Axis Model

### 3.1 Enum

```python
class SliceAxis(str, Enum):
    Z = "Z"
    X = "X"
    Y = "Y"
```

Processing is **identical in canonical space**; world-axis differences are handled purely by coordinate permutation before/after canonical algorithms.

### 3.2 Canonical Coordinate System

**Definition:** In canonical space:

- **Slice plane** = XY plane (2D footprint lives in XY).
- **Stack / depth direction** = +Z (layers stack along Z; slicing advances along Z in canonical space).

Every axis-specific operation on boundaries and stitching:

1. Transform world coordinates → canonical (`to_canonical=True`).
2. Run Z-oriented algorithm unchanged.
3. Transform canonical → world (`to_canonical=False`).

### 3.3 Point Permutation Functions

Input: `points` — array shape `(N, 3)`, columns `[x, y, z]` in **world** or **canonical** space depending on direction.

#### 3.3.1 World → Canonical (`to_canonical=True`)

| Axis | Output columns `[u, v, w]` |
|------|----------------------------|
| Z | `[x, y, z]` (identity) |
| X | `[y, z, x]` — footprint YZ, depth X |
| Y | `[x, z, y]` — footprint XZ, depth Y |

#### 3.3.2 Canonical → World (`to_canonical=False`)

| Axis | Input `[u, v, w]` → world `[x, y, z]` |
|------|---------------------------------------|
| Z | `[u, v, w]` (identity) |
| X | `[w, u, v]` |
| Y | `[u, w, v]` |

**Round-trip invariant:** `from_canonical(to_canonical(p, axis), axis) == p`.

#### 3.3.3 Mesh Transforms

- **PyVista PolyData:** deep-copy mesh; replace `.points` with permuted points.
- **trimesh.Trimesh:** copy mesh; replace `.vertices` with permuted vertices.

### 3.4 Axis Extent

For bounds `[xmin, xmax, ymin, ymax, zmin, zmax]`:

| Axis | Extent (mm) |
|------|-------------|
| X | `xmax - xmin` |
| Y | `ymax - ymin` |
| Z | `zmax - zmin` |

---

## 4. File System Layout

Given source STL path `{parent}/{name}.stl`:

```
{parent}/{name}/                          ← model root (output_dir_for_stl)
  1.stl, 2.stl, ...                       ← Z-axis slices (legacy root location)
  boundaries_v2/
    1.stl, 2.stl, ...                     ← Z-axis per-layer boundaries
    stitched.stl                          ← Z-axis unified shell
    negative.stl                          ← Z-axis mold negative (optional)
  x/
    1.stl, 2.stl, ...                     ← X-axis slices
    boundaries_v2/
      1.stl, ...
      stitched.stl
  y/
    1.stl, 2.stl, ...
    boundaries_v2/
      1.stl, ...
      stitched.stl
  combined.stl                            ← merged Z + X + Y stitched shells
```

### 4.1 Path Functions

```
output_dir_for_stl(stl_path)     → stl_path.parent / stl_path.stem
slice_dir_for_axis(stl_path, Z)  → output_dir_for_stl(stl_path)
slice_dir_for_axis(stl_path, X)  → output_dir_for_stl(stl_path) / "x"
slice_dir_for_axis(stl_path, Y)  → output_dir_for_stl(stl_path) / "y"
boundaries_v2_output_dir(slice_dir) → slice_dir / "boundaries_v2"
stitched_output_path(slice_dir)    → boundaries_v2_output_dir(slice_dir) / "stitched.stl"
negative_output_path(slice_dir)    → boundaries_v2_output_dir(slice_dir) / "negative.stl"
combined_model_path(stl_path)      → output_dir_for_stl(stl_path) / "combined.stl"
reduced_output_path(stl_path)      → stl_path.with_name(f"{stem}_reduced{suffix}")
```

### 4.2 Slice File Discovery

`list_slice_files(slice_dir)`:

1. Glob `slice_dir/*.stl`.
2. Keep files whose stem matches `^\d+$` (integer only).
3. Sort by integer stem ascending.
4. Return ordered path list `[1.stl, 2.stl, ...]`.

`list_boundary_files(boundary_dir)` — same logic but **exclude** stems in `_EXCLUDED_BOUNDARY_STEMS`.

---

## 5. Progress Callback Contract

```python
ProgressCallback = Callable[[message: str, step: int, total: int], None]
```

- `step` is 0-based current progress index.
- `total` is expected maximum step count for the operation.
- UI uses this to drive footer progress bar and stage label.

---

## 6. Algorithm: Mesh Reduction

**Module:** `mesh_reducer.py`  
**Entry:** `reduce_and_save_stl(source_path, mesh, reduction_percent, progress?)`

### 6.1 Preconditions

- `0 < reduction_percent < 100`
- Mesh has ≥ 4 triangles after surface extraction

### 6.2 Steps

1. `original_triangles = mesh.n_cells`
2. Extract surface: `surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()`
3. `target_reduction = min(reduction_percent / 100.0, 0.95)` — cap at 95% removal
4. `reduced = surface.decimate_pro(target_reduction, preserve_topology=True)`
5. `reduced = reduced.clean().triangulate()`
6. Fail if `reduced.n_cells == 0`
7. Save to `reduced_output_path(source_path)`
8. `actual_percent = 100 * (1 - reduced_triangles / original_triangles)`

### 6.3 Output

```python
ReduceResult(
    output_path,
    original_triangles,
    reduced_triangles,
    reduction_percent=actual_percent,
)
```

---

## 7. Algorithm: Axis-Aware Slicing

**Module:** `mesh_slicer.py`  
**Entry:** `slice_mesh_along_axis(mesh, output_dir, slice_height_mm, axis, progress?)`

### 7.1 Slice Count Estimate

```
estimate_slice_count(mesh, slice_height_mm, axis):
  if slice_height_mm <= 0: return 0
  extent = axis_extent_mm(mesh, axis)
  if extent <= 0: return 0
  return ceil(extent / slice_height_mm)
```

### 7.2 Slice Plane Generation

Let `[xmin, xmax, ymin, ymax, zmin, zmax] = mesh.bounds`.

| Axis | Start values | Clip box `[xmin, xmax, ymin, ymax, zmin, zmax]` per layer |
|------|--------------|-------------------------------------------------------------|
| X | `np.arange(xmin, xmax, slice_height_mm)` | `[start, end, ymin, ymax, zmin, zmax]` |
| Y | `np.arange(ymin, ymax, slice_height_mm)` | `[xmin, xmax, start, end, zmin, zmax]` |
| Z | `np.arange(zmin, zmax, slice_height_mm)` | `[xmin, xmax, ymin, ymax, start, end]` |

Where `end = min(start + slice_height_mm, axis_max)` and skip if `end <= start`.

### 7.3 Per-Layer Processing

For each layer index `i` starting at 1:

1. `piece = mesh.clip_box(box_bounds, invert=False)`
2. Skip (increment `skipped`) if `piece.n_cells == 0`
3. `surface = piece.extract_surface(algorithm="dataset_surface").triangulate()`
4. Skip if `surface.n_cells == 0`
5. Save `output_dir / f"{i}.stl"` (enumeration follows successful layers only in order — index is enumerate position in `starts`, not renumbered on skip)

**Note:** Layer numbering uses `enumerate(starts, start=1)` index even when layers are skipped; gaps in numbering can occur if intermediate layers are empty.

### 7.4 Output

```python
SliceResult(output_dir, piece_paths, skipped_slices, axis)
```

---

## 8. Algorithm: Footprint & Offset Outline

**Module:** `boundary_generator.py`  
Used in canonical space (after axis transform) where footprint is always in XY.

### 8.1 Triangle Footprint from XY Projection

For each triangle face (indices `i,j,k` from PyVista face array):

1. Take vertex XY coordinates.
2. Skip if fewer than 3 unique XY points.
3. Build Shapely `Polygon(coords)`.
4. Keep if valid, non-empty, area > 0.

### 8.2 Cross-Section Footprint

1. `z_mid = (zmin + zmax) / 2` from mesh bounds (canonical Z).
2. `section = mesh.slice(normal=(0,0,1), origin=(0,0,z_mid))`
3. Parse section polyline connectivity into 2D LineStrings (XY of section points).
4. `polygons = polygonize(linestrings)`
5. `footprint = unary_union(polygons).buffer(0)`

### 8.3 Combined Footprint

```
footprint_from_mesh(mesh):
  parts = []
  if cross_section footprint exists: append
  if triangle polygons exist: append unary_union(triangle_polygons)
  if parts empty: return None
  combined = unary_union(parts).buffer(0)
  return combined if not empty else None
```

### 8.4 Equidistant Offset Outline

**Input:** footprint (Polygon or MultiPolygon), `offset_mm > 0`

1. `combined = unary_union(footprint).buffer(0)`
2. If empty → error.
3. **Multi-body rule:** if `MultiPolygon` with `len(geoms) > 1`, use `combined.convex_hull` as source; else use `combined`.
4. `outline = source.buffer(offset_mm, join_style=1, resolution=32)`
   - `join_style=1` = mitre joins in Shapely.
5. Return `outline.buffer(0)` to clean topology.

**Semantic:** Single closed outer outline at exactly `offset_mm` from the source boundary (convex hull wrap when multiple disconnected bodies).

---

## 9. Algorithm: Boundary Generation v2

**Module:** `boundary_generator_v2.py`  
**Entry:** `create_boundaries_v2_for_slices(slice_dir, offset_mm, slice_height_mm, axis, progress?)`

### 9.1 Previous Layer Count

```
previous_layer_count(offset_mm, slice_height_mm):
  require slice_height_mm > 0
  return max(1, int(offset_mm / slice_height_mm))   # integer division, floor
```

Example: offset 5 mm, slice height 1 mm → 5 previous layers.  
Example: offset 5 mm, slice height 2 mm → 2 previous layers.

For slice index `step` (1-based) in ordered slice list:

```
previous_paths = slice_files[max(0, step - 1 - layer_count) : step - 1]
```

Slice 1 has no previous paths.

### 9.2 Per-Slice Pipeline (`create_boundary_v2_for_slice`)

**Inputs:** current slice path, list of previous slice paths, output path, `offset_mm`, `axis`

1. Load current and previous meshes as PyVista PolyData.
2. Transform all to **canonical** space via `transform_polydata(..., axis, to_canonical=True)`.
3. `footprint = footprint_from_mesh(current_mesh)` — if None, return `(False, empty_stats)`.
4. `outline = offset_outline(footprint, offset_mm)`.
5. Extract resampled boundary loop (see §9.3).
6. Compute per-point depth values (see §9.4) → `z_values`, placement stats.
7. Build thin wall mesh (see §9.5).
8. Transform wall mesh to **world** space: `transform_trimesh(..., axis, to_canonical=False)`.
9. Export STL to output path.

### 9.3 Outline Loop Extraction

1. If outline is MultiPolygon, take largest-area polygon.
2. Take `exterior.coords[:-1]` (drop closing duplicate).
3. Require ≥ 3 points.
4. **Resample** to exactly `_LOOP_POINTS` (128) points evenly by arc length:
   - Close loop: append first point to end.
   - Build Shapely `LineString(closed)`.
   - Sample at `np.linspace(0, length, 128, endpoint=False)`.
   - Interpolate each distance along line.

### 9.4 Stack Depth (Canonical Z) for Each Loop Point

For each loop point `(x, y)` in XY (canonical):

**Search radius:** `search_radius = max(offset_mm * 2.0, 1.0)`

**Local column queries** on PyVista mesh points:

```
_local_z_top(mesh, x, y, r):
  mask = hypot(px - x, py - y) <= r
  if any: return max(pz[mask])
  else: return mesh.bounds zmax (index 5)

_local_z_bottom(mesh, x, y, r):
  mask = hypot(px - x, py - y) <= r
  if any: return min(pz[mask])
  else: return mesh.bounds zmin (index 4)
```

**Depth placement rule:**

```
if no previous_meshes:
  z = local_z_bottom(current_mesh, x, y, r) + offset_mm
else:
  z_topmost = max(local_z_top(prev_mesh, x, y, r) for prev_mesh in previous_meshes)
  z = z_topmost + offset_mm
```

**Interpretation:**

- **In-plane:** loop point lies on offset outline → `offset_mm` from current slice footprint boundary.
- **Stack direction:** `offset_mm` above the **topmost** surface point among all previous `layer_count` slices at the same `(x,y)` column.
- **First slice:** `offset_mm` above current slice bottom at that column.

### 9.5 Placement Validation (Post-Compute)

**Tolerance:** `tol = max(0.02, offset_mm * 0.05)`

For each placed point `(x, y, z)`:

| Check | Formula | Pass condition |
|-------|---------|----------------|
| In-plane offset | `xy_dist = Point(x,y).distance(footprint.boundary)` | `|xy_dist - offset_mm| <= tol` |
| Stack offset | `z_topmost` from previous layers | if no previous: pass; else `z - z_topmost >= offset_mm - tol` |

Count `valid_points` where both pass. Aggregate into `BoundaryV2PlacementStats`.

### 9.6 Thin Wall Mesh from Loop

**Input:** `loop_xy` shape `(N, 2)`, `z_values` shape `(N,)`, `wall_thickness_mm = 0.1`

For each edge `(p0, p1)` between consecutive loop points (closed polygon):

1. `z0, z1` = depth values at endpoints.
2. Edge vector `e = p1 - p0`; if `|e| < 1e-9`, in-plane normal `n = [1,0]`; else:
   - `tangent = e / |e|`
   - `normal = [-tangent_y, tangent_x]` (90° CCW in XY)
3. Half-width offset: `± normal * (wall_thickness_mm / 2)` → corners `a0, a1` at p0 and `b0, b1` at p1.
4. Half-depth: `half_z = wall_thickness_mm / 2`
5. Build **8 vertices** per edge segment (quad prism extruded in Z locally):

```
a0 at z0-half_z, a1 at z0-half_z, a1 at z0+half_z, a0 at z0+half_z
b0 at z1-half_z, b1 at z1-half_z, b1 at z1+half_z, b0 at z1+half_z
```

6. **12 triangles** (2 per face of the quad prism) with fixed local indexing pattern (see reference implementation face list in `thin_wall_v2_from_loop`).

7. Return `trimesh.Trimesh(vertices, faces, process=True)`.

**World meaning after inverse transform:** Wall is thin in canonical Z (stack direction) and thin in-plane normal direction; follows the offset loop in the slice plane.

### 9.7 Batch Output

```python
BoundaryV2Result(
  output_dir=boundaries_v2_output_dir(slice_dir),
  boundary_paths=[...],
  skipped_slices,
  axis,
  previous_layer_count=layer_count,
  placement_stats=aggregated,
)
```

On exception during single slice: mark skipped, continue.

---

## 10. Algorithm: Unified Lofted Stitching

**Module:** `boundary_stitcher.py`  
**Entry:** `stitch_boundaries(slice_dir, axis, progress?)`

Builds **one watertight shell** per axis by lofting between shared rings at layer interfaces — no overlapping per-layer full-height walls.

### 10.1 Inputs

- Numbered boundary STLs in `boundaries_v2/`
- Matching numbered slice STLs in `slice_dir` (same stems)
- All processing in canonical space, then inverse transform on output.

### 10.2 Per-Layer Loop Extraction

For each boundary/slice pair:

1. Load boundary as trimesh; transform to canonical.
2. Load slice as trimesh → wrap PyVista → transform to canonical.
3. `z_bottom = slice_mesh.bounds[4]`, `z_top = slice_mesh.bounds[5]` (canonical Z).
4. `z_mid = (z_bottom + z_top) / 2`
5. `loop = extract_xy_loop(boundary_mesh, z_hint=z_mid)`

**extract_xy_loop algorithm:**

1. Convert trimesh to PyVista with triangular faces.
2. Slice at plane Z = `z_hint`, normal (0,0,1).
3. If section has geometry: polygonize XY section lines → take **largest area** polygon exterior coords.
4. Fallback: 2D convex hull of all vertex XY coordinates.

### 10.3 Loop Resampling (Stitch)

Same arc-length resampling as §9.3 but default **`_LOOP_SEGMENTS = 256`** points.

### 10.4 Loop Alignment

**Problem:** Two loops of same point count may have different starting indices.

```
align_loops(reference, target):
  best_shift = argmin over shift in 0..N-1 of sum(|reference - roll(target, -shift)|)
  return roll(target, -best_shift)
```

Uses sum of Euclidean distances point-wise (not cyclic DTW).

### 10.5 Ring Stack Construction

Given `layer_loops[]`, `z_bottoms[]`, `z_tops[]` for N layers:

```
z_levels = [z_bottoms[0]]
rings = [resample(layer_loops[0], 256)]

for i in 0 .. N-2:
  z_levels.append(z_tops[i])                    # interface plane
  lower = resample(layer_loops[i], 256)
  upper = align_loops(lower, resample(layer_loops[i+1], 256))
  rings.append(0.5 * lower + 0.5 * upper)      # blended interface ring

z_levels.append(z_tops[N-1])
rings.append(resample(layer_loops[N-1], 256))

# Global alignment to first ring
reference = rings[0]
aligned_rings = [reference]
for ring in rings[1:]:
  aligned_rings.append(align_loops(reference, ring))
```

**Ring count:** `2 * N - 1` for N layers (N bottom/top levels + N-1 interfaces, collapsed to one ring per z_level).

Example N=3: z_levels = `[z0_bot, z0_top, z1_top, z2_top]` — 4 levels when z0_top==z1_bot at shared interface (interface ring sits at shared Z).

Actually: starts with z_bottom[0], adds each z_tops[i] for i=0..N-2, adds z_tops[-1]. Total = 1 + (N-1) + 1 = N+1? Let's trace N=3:
- z_levels = [z_bottom[0]]
- i=0: append z_tops[0]; i=1: append z_tops[1]
- append z_tops[2]
→ [zb0, zt0, zt1, zt2] = 4 levels = N+1 for N=3. Reference code says ring_count = len(z_levels) = 2*N - 1 only when... N=3 → 4, not 5. So formula is **N+1** ring levels in implementation (not 2N-1). Document as implemented: **len(z_levels) = N + 1** for N ≥ 1.

Wait recount N=1: z_levels=[zb0, zt0] → 2 rings. N+1=2 ✓  
N=2: [zb0, zt0, zt1] → 3 = N+1 ✓

### 10.6 Loft Mesh Construction

Given aligned `rings[i]` (each Npts × 2) and `z_levels[i]`:

**Side surface:**

- Vertex index layout: ring `r` point `p` → vertex `r * Npts + p` at `(ring[p].x, ring[p].y, z_levels[r])`.
- For each consecutive ring pair, quads split into 2 triangles CCW:
  - `(v00, v10, v11)`, `(v00, v11, v01)` where indices wrap modulo Npts.

**Caps:**

- Bottom: triangulate `rings[0]` at `z_levels[0]`, **reverse** face winding.
- Top: triangulate `rings[-1]` at `z_levels[-1]`, normal winding.
- Triangulation: Shapely Polygon → `trimesh.creation.triangulate_polygon(..., engine="earcut")`.

**Post-process:**

```
merge_vertices()
update_faces(unique_faces())
remove_unreferenced_vertices()
```

### 10.7 Output Transform & Export

1. Transform combined mesh canonical → world.
2. Save `stitched.stl` in `boundaries_v2/`.

```python
StitchResult(output_path, layers_stitched=N, ring_count=len(z_levels), triangle_count)
```

---

## 11. Algorithm: Combine Axis Models

**Module:** `model_combiner.py`  
**Entry:** `combine_axis_models(stl_path, progress?)`

### 11.1 Preconditions

Must exist for **all three** axes Z, X, Y:

```
stitched_output_path(slice_dir_for_axis(stl_path, axis))
```

If any missing → `FileNotFoundError` listing missing axes.

### 11.2 Merge

1. Load each stitched mesh via trimesh (concatenate Scene if needed).
2. `combined = trimesh.util.concatenate([mesh_Z, mesh_X, mesh_Y])`
3. Clean: merge_vertices, unique_faces, remove_unreferenced_vertices.
4. Export `combined_model_path(stl_path)`.

**Semantic:** Three independent triangle soups in one file — **no boolean union**, simple concatenation of geometry.

```python
CombineResult(output_path, axis_paths={Z: path, X: path, Y: path}, triangle_count)
```

---

## 12. Algorithm: Mold Negative (Z Axis Only)

**Module:** `stitched_negative.py`  
**Entry:** `create_stitched_negative(slice_dir, margin_mm, progress?)`

Uses **Z-axis** `slice_dir` only (typically `output_dir_for_stl` root, not `x/` or `y/`).

### 12.1 Bounding Box Envelope

Given stitched mesh bounds `[min, max]` per axis:

```
center = (min + max) / 2
size   = (max - min) + margin_mm   # added to EACH axis extent (same margin all axes)
box    = trimesh.creation.box(extents=size) translated to center
```

**Semantic:** Axis-aligned box centered on stitch AABB, **`margin_mm` longer in X, Y, and Z** (added once to each extent dimension).

### 12.2 Mesh Preparation for Boolean

```
remove_infinite_values()
merge_vertices()
update_faces(unique_faces() & nondegenerate_faces())
remove_unreferenced_vertices()
```

Do **not** call `process(validate=True)` (requires scipy for normal fixing).

### 12.3 Boolean Difference

```
negative = trimesh.boolean.difference([box, stitched], engine="manifold", check_volume=False)
```

Fail if result empty or zero faces.

Prepare result mesh same as §12.2.

Export `negative.stl` in `boundaries_v2/`.

---

## 13. End-to-End Workflow

### 13.1 Per-Axis Pipeline (repeat for Z, X, Y)

```
1. slice_mesh_along_axis(mesh, slice_dir_for_axis(stl, axis), slice_height_mm, axis)
2. create_boundaries_v2_for_slices(slice_dir, offset_mm, slice_height_mm, axis)
3. stitch_boundaries(slice_dir, axis)
```

### 13.2 Global Combine

After all three axes stitched:

```
combine_axis_models(stl_path)
```

### 13.3 Optional Mold Negative (Z only)

After Z stitch:

```
create_stitched_negative(slice_dir_for_axis(stl, Z), margin_mm)
```

### 13.4 Typical User Parameter Relationships

- `previous_layer_count = int(offset_mm / slice_height_mm)` controls how many prior slices contribute to stack-depth Z placement.
- Same `slice_height_mm` and `offset_mm` apply to all three axes when invoked from UI.

---

## 14. Error Handling Summary

| Condition | Behavior |
|-----------|----------|
| `slice_height_mm <= 0` | ValueError |
| `offset_mm <= 0` | ValueError |
| `reduction_percent` out of range | ValueError |
| `margin_mm <= 0` | ValueError |
| Missing slice directory | FileNotFoundError |
| No numbered slice files | FileNotFoundError with hint |
| Empty slice layer | Skip layer, increment skipped count |
| Boundary footprint None | Skip slice boundary |
| Boundary exception | Skip slice, continue batch |
| Missing stitched for combine | FileNotFoundError listing axes |
| Boolean produces empty mesh | ValueError |
| Missing stitch for negative | FileNotFoundError |

---

## 15. Data Types Reference

```python
@dataclass SliceResult:
    output_dir: Path
    piece_paths: list[Path]
    skipped_slices: int
    axis: SliceAxis

@dataclass ReduceResult:
    output_path: Path
    original_triangles: int
    reduced_triangles: int
    reduction_percent: float

@dataclass BoundaryResult:
    output_dir: Path
    boundary_paths: list[Path]
    skipped_slices: int

@dataclass BoundaryV2PlacementStats:
    total_points: int = 0
    valid_points: int = 0

@dataclass BoundaryV2Result(BoundaryResult):
    axis: SliceAxis
    previous_layer_count: int
    placement_stats: BoundaryV2PlacementStats

@dataclass StitchResult:
    output_path: Path
    layers_stitched: int
    ring_count: int
    triangle_count: int

@dataclass CombineResult:
    output_path: Path
    axis_paths: dict[str, Path]
    triangle_count: int

@dataclass NegativeResult:
    output_path: Path
    margin_mm: float
    box_extents_mm: tuple[float, float, float]
    triangle_count: int
```

---

## 16. Implementation Notes for Code Generation

1. **Single canonical Z algorithm:** All axis generality is achieved exclusively through §3.3 permutations; do not duplicate footprint/stitch logic per axis.

2. **Shapely join_style=1** is mitre; **resolution=32** on buffer controls curve segmentation.

3. **PyVista face array format:** `[3, i, j, k, 3, i, j, k, ...]` — reshape `(-1, 4)[:, 1:4]` for triangle indices.

4. **Integer division** for `previous_layer_count` — floor toward zero; minimum 1.

5. **Slice numbering** uses arange start positions; last slice may be thinner than `slice_height_mm` when extent not an exact multiple.

6. **Multi-body slices:** convex hull wrap before offset ensures one continuous outer boundary; may over-estimate concave regions.

7. **Stitch interface rings:** 50/50 linear blend between aligned consecutive layer loops at each slice top Z.

8. **Combine vs negative:** Combine concatenates three axis shells; negative boolean-subtracts only Z stitch from a box.

9. **Dependencies:** `manifold3d` required for mold negative; earcut required for cap triangulation.

10. **File naming:** Only purely numeric stems participate in ordering; exclude `stitched`, `stitched_v2`, `negative` from boundary iteration.

---

## 17. Revision History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2026-06-28 | Initial spec: multi-axis slice, boundary v2, stitch, combine, mold negative |
