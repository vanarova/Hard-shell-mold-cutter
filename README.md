# Hard-shell Mold Cutter

Desktop tools for STL viewing and hard-shell mold workflows: layered slicing, 2.5D mold making, and 3D plane wrapping with a deformable end plane.

## Tools

| Launcher | Entry point | Purpose |
| --- | --- | --- |
| `Launch_3D_Mold_Wrapper.bat` | `src/main_3d_mold_wrapper.py` | Start/end oriented planes, end-plane deform, Backend / Back-start planes, Wrap Plane and Wrap Back |
| `launch_25D-MoldMaker.bat` | `src/main_25d_moldmaker.py` | 2.5D mold maker workflow |
| `launch.bat` | `src/main.py` | Classic STL viewer and Z-slice / boundary / mold-negative pipeline |

On first run, each launcher creates a local `.venv` (if missing) and installs dependencies from `requirements.txt`.

## Requirements

- Python 3.10+
- Windows (launchers are `.bat`; source also runs on macOS/Linux with a venv)

## Setup (manual)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**3D Mold Wrapper (recommended for plane wrap workflows):**

```bat
Launch_3D_Mold_Wrapper.bat
```

or:

```bash
.venv\Scripts\python.exe src\main_3d_mold_wrapper.py
```

**2.5D Mold Maker:**

```bat
launch_25D-MoldMaker.bat
```

**Classic Mold Liner:**

```bat
launch.bat
```

## 3D Mold Wrapper

1. Open an STL and optionally reduce triangles.
2. Set **Start Plane** and **End Plane** (X/Y/Z, rotate X/Y/Z, pick on model, confirm).
3. Use **Match End to Start** inside the End Plane section when needed.
4. Deform the end plane with control points (select from the list or click a point in the 3D view, then pull/push).
5. **Copy Backend Plane** freezes the curved end surface; optionally show/hide it.
6. Set **Back-start Plane** (or **Apply Backend Position to Back-start**) and its N×N sample density (up to ~100,000 points; points are not drawn).
7. **Wrap Plane** extrudes Start → End (stops on model or deformed end).
8. **Wrap Back** extrudes Back-start → Backend (stops on model or Backend surface).

Outputs are written next to the model STL in a folder named after the file (for example `wrap_plane.stl`, `wrap_back_plane.stl`).

## Classic Mold Liner (brief)

1. Open STL → optional triangle reduction.
2. Slice by Z height into `1.stl`, `2.stl`, …
3. Create equidistant boundaries → `boundaries_v2/`.
4. Stitch boundaries → `stitched.stl`.
5. Create mold negative → `negative.stl`.

See `spec/MOLD_LINER_ALGORITHM_SPEC.md` for algorithm details.

## Project layout

```
├── Launch_3D_Mold_Wrapper.bat
├── launch_25D-MoldMaker.bat
├── launch.bat
├── requirements.txt
├── README.md
├── spec/
│   └── MOLD_LINER_ALGORITHM_SPEC.md
└── src/
    ├── main_3d_mold_wrapper.py
    ├── main_25d_moldmaker.py
    ├── main.py
    └── app/
        ├── mold_wrapper_main_window.py
        ├── plane_deform.py
        ├── plane_wrap.py
        └── …
```

## License / remote

Repository: https://github.com/vanarova/Hard-shell-mold-cutter
