# GammaView Dosimetry

A [3D Slicer](https://www.slicer.org/) scripted module for intraoperative GammaTile placement planning and dosimetry visualization. Developed as part of the IntraOp GammaTile Research project at the University of Minnesota, with input from the radiation oncology and neurosurgery teams.

## What it does

GammaView lets a user load a patient MRI volume in 3D Slicer, place GammaTile collagen implants (and standalone Cs-131 seeds) directly on the brain surface, and visualize the resulting radiation dose as 3D isodose surfaces — all intraoperatively, without leaving Slicer.

Dose is computed from a TG-43U1 2D formalism for the IsoRay Proxcelan CS-1 Rev2 seed, using tabulated radial dose and anisotropy function data digitized from the seed's dosimetry reference tables.

## Workflow

1. Load an MRI volume and click **Create Brain Surface** to generate a surface model from it.
2. Switch to **Placement Mode** and click on the brain surface (in the 3D view or any 2D slice view) to place GammaTiles or standalone seeds. Tiles snap and orient to the local surface normal.
3. Switch to **Cursor Mode** to select placed tiles by clicking them in the 2D slice views.
4. With a tile selected, use the **Rotate**/**Translate** controls to fine-tune its position and orientation.
5. Click **Update & Show Isodoses** to compute the dose volume and render 3D isodose surfaces.
6. Export results:
   - **Export seed and tile positions** — CSV or JSON, with per-seed and per-tile RAS/local coordinates, normals, and seed axes.
   - **Export Dose Volume structure** — DICOM RTDOSE (in Gy) or NRRD.

The current tile count is shown at the bottom of the module panel.

## Key components

| Component | Responsibility |
|---|---|
| `DoseInterpolator` | TG-43U1 dose-rate calculation from tabulated radial dose and anisotropy (F(r,θ)) data for the Proxcelan CS-1 Rev2 seed |
| `GammaTilePlacement` / `SingleSeedPlacement` | Geometry of a placed 4-seed tile or a standalone seed (center, normal, seed positions in RAS/local space) |
| `GammaTileDosimetry` | Scene-level logic: placing/removing tiles and seeds, undo history, dose-volume computation |
| `BrainSurfaceGenerator` | Builds a brain surface model from the loaded volume |
| `IsodoseSurfaceGenerator` | Builds 3D isodose surface models from a computed dose volume |
| `GammaTileModule` / `GammaTileWidget` | 3D Slicer module registration and the interactive UI (placement, selection, drag/rotate/translate, click-to-surface raycasting, export) |

## Requirements

- [3D Slicer](https://www.slicer.org/) (scripted module — depends on `slicer`, `vtk`, `qt`, `ctk`)
- Python packages: `numpy`

## Installation

Add the directory containing `GammaTile.py` as an additional module path in 3D Slicer (**Edit → Application Settings → Modules**), or place it in a Slicer extension's module folder, then restart Slicer. The module appears under the **Surgery** category as **GammaView Dosimetry**.
