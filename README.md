# GammaView Dosimetry

A [3D Slicer](https://www.slicer.org/) scripted module for intraoperative GammaTile placement planning and dosimetry visualization. Developed as part of the IntraOp GammaTile Research project at the University of Minnesota, with input from the radiation oncology and neurosurgery teams.

GammaView lets a user load a patient MRI volume in 3D Slicer, place GammaTile collagen implants (and standalone Cs-131 seeds) directly on the brain surface, and visualize the resulting radiation dose as 3D isodose surfaces, all intraoperatively and without leaving Slicer. Dose is computed with the TG-43U1 2D formalism for the IsoRay Proxcelan CS-1 Rev2 seed using tabulated radial dose and anisotropy data.

The whole module is a single file, `GammaTile.py`. A full-head dose grid at 2 mm spacing computes in well under a second.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow](#workflow)
- [Interaction reference](#interaction-reference)
- [Dosimetry model](#dosimetry-model)
- [Dose grid and isodose surfaces](#dose-grid-and-isodose-surfaces)
- [Export formats](#export-formats)
- [Architecture](#architecture)
- [Performance](#performance)
- [Validation and testing](#validation-and-testing)
- [Repository layout](#repository-layout)
- [Version notes](#version-notes)

## Features

- Brain surface extraction from any loaded scalar volume (automatic intensity threshold, smoothed mesh).
- Tile placement by clicking in the 3D view or in any 2D slice view. Tiles settle onto the surface and orient to the local surface normal; penetration below the surface is capped at 10 mm.
- Standalone Cs-131 seed placement with the seed axis along the surface normal.
- Cursor mode for selecting a placed tile or seed in a 2D view, then rotating (10° steps) or translating (2 mm steps) it in that view's plane.
- Undo of the last tile or last seed.
- Vectorized TG-43U1 dose computation on a 2 mm grid over the whole volume.
- 100%, 90% and 50% isodose surfaces of the prescription, rendered in 3D and as intersections in the slice views, with a colour legend.
- Export of all seed and tile positions to CSV or JSON.
- Export of the dose volume as DICOM RTDOSE (with SlicerRT), a DICOM image series, or NRRD.

## Requirements

- 3D Slicer 5.x (tested with 5.12.1 on macOS). The module depends on Slicer's bundled `slicer`, `vtk`, `qt`, `ctk` and `numpy`.
- Optional: the [SlicerRT](https://slicerrt.github.io/) extension, which enables true DICOM RTDOSE export. Without it the module falls back to a DICOM image series or NRRD.

## Installation

1. Clone or download this repository.
2. In Slicer open **Edit → Application Settings → Modules**, add the folder containing `GammaTile.py` under *Additional module paths*, and restart Slicer.
3. The module appears under the **Surgery** category as **GammaView Dosimetry**.

## Workflow

1. **Load an MRI volume** (DICOM or any format Slicer reads) and select it in the *MRI Volume* selector.
2. Click **Create Brain Surface**. The module thresholds the volume at *mean + 0.5 × standard deviation*, extracts the isosurface, smooths it and adds it to the scene as a translucent model named *BrainSurface*.
3. Set the **Prescription** dose in Gy (default 60 Gy). This drives the isodose levels.
4. Choose **Placement Mode** and click on the brain surface in the 3D view or in the Red, Yellow or Green slice view to place a tile. Each click adds one tile. Choose **Seeds Mode** to place standalone seeds the same way.
5. Choose **Cursor Mode** and click a tile or seed in a slice view to select it. The selection label shows what is selected and from which view.
6. Use **Rotate Clockwise / Counterclockwise** and the **← → ↑ ↓** buttons to fine-tune the selected tile or seed. Rotation is about the normal of the slice view it was selected in; translation is in that view's plane.
7. Click **Update & Show Isodoses** to compute the dose grid and render the isodose surfaces. The status line reports the tile and seed counts, the maximum dose and the number of surfaces created, and a detailed summary is printed to the Python console.
8. Export with **Export seed and tile positions** or **Export Dose Volume structure**.

The tile and seed counts are always shown at the bottom of the panel.

## Interaction reference

| Control | Effect |
|---|---|
| Placement Mode, click in 3D view | Picks the point under the cursor on any visible actor (falls back to the volume centre), takes the surface normal from the brain mesh or from the volume gradient, settles the point onto the surface along the normal, limits penetration to 10 mm and places a tile whose bottom face sits on the surface. |
| Placement Mode, click in a slice view | Casts a ray from the click along the slice normal, finds the first surface (cavity wall preferred, then interior wall, then the brain mesh), settles onto it and places a tile oriented in the slice plane. |
| Seeds Mode, click | Same surface search as tile placement; the seed is placed with its axis along the surface normal and one end on the surface. |
| Cursor Mode, click in a slice view | Selects a tile within 15 mm (in-plane) or a seed within 8 mm of the click. If several match, the most recently placed one is chosen. |
| Rotate Clockwise / Counterclockwise | Rotates the selected tile by ±10° about the normal of the slice view it was selected from. The tile normal is unchanged; the seed axis and seed positions rotate with the tile. |
| ← → ↑ ↓ | Moves the selected tile or seed by 2 mm along the selected slice view's horizontal or vertical axis. |
| Undo Last Tile / Undo Last Seed | Removes the most recently placed tile or seed and its model. |
| Update & Show Isodoses | Recomputes the dose grid and replaces the dose volume and isodose models in the scene. |

## Dosimetry model

Dose is evaluated per seed with the AAPM TG-43U1 2D line-source formalism and summed over all seeds:

```
D(r, θ) = Λ · [G_L(r, θ) / G_L(r₀, θ₀)] · g_L(r) · F(r, θ) · C
```

| Quantity | Value in this module |
|---|---|
| Seed model | IsoRay Proxcelan CS-1 Rev2 (Cs-131) |
| Dose-rate constant Λ | 1.056 cGy h⁻¹ U⁻¹ |
| Active length L | 0.40 cm |
| Reference point | r₀ = 1.0 cm, θ₀ = 90° |
| Geometry function G_L | Line-source form, with the on-axis approximation 1 / (r² − (L/2)²) for θ < 0.01°. The denominator is clamped to a magnitude of at least 1 × 10⁻³ cm² so a point exactly at a source tip yields a finite dose instead of a division error. |
| Radial dose g_L(r) | CLRP v2 polynomial fit, (a₀ r⁻² + a₁ r⁻¹ + a₂ + a₃ r + a₄ r² + a₅ r³) · exp(−a₆ r), evaluated for r clipped to [0.05, 10] cm |
| Anisotropy F(r, θ) | Bilinear interpolation on the published table (32 polar angles from 0° to 90°, 12 radii from 0.10 to 10 cm). Table entries that are undefined at small radii are filled from the nearest tabulated radius, searching outward first. |
| Conversion C | 335.23 × 3.5. This corresponds to integrating the initial dose rate over the complete decay of Cs-131 (mean life ≈ 335.2 h) for an air-kerma strength of 3.5 U per seed. Change `DOSE_CONVERSION_FACTOR` in `DoseInterpolator` for a different seed strength. |

The polar angle is measured from the seed's longitudinal axis and folded into [0°, 90°]. Radius is clipped to [0.041, 10] cm. Points closer than the seed radius (0.41 mm) to a seed centre, or farther than 100 mm from it, receive no dose from that seed. Negative dose rates are clamped to zero.

**Tile geometry.** A GammaTile is modelled as a 20 × 20 × 4 mm slab with four seeds at (±5, ±5, 0) mm in the tile's local frame, i.e. 5 mm in from each edge at mid-thickness. All four seeds are parallel to the tile's local +X axis, which lies in the tile plane. The tile's local Z axis is the surface normal; the tile centre sits 2 mm above the surface so the bottom face rests on it.

**Standalone seed geometry.** A standalone seed is 4.5 mm long and 0.82 mm in diameter, with its axis along the surface normal. Its centre is offset half a seed length below the settled surface point so one end touches the surface.

## Dose grid and isodose surfaces

- The grid covers the RAS bounding box of the selected volume at 2 mm isotropic spacing.
- Only voxels within 50 mm of at least one tile or seed centre are evaluated; the rest are zero.
- The result is stored in the scene as a scalar volume named *Dose Volume* (units cGy) and tagged as a dose volume for Subject Hierarchy.
- Isodose surfaces are extracted at 100% (red), 90% (orange) and 50% (yellow) of the prescription with `vtkFlyingEdges3D`, and shown in 3D and as slice intersections. Surfaces whose level exceeds the maximum dose are skipped.
- Each update removes the previous dose volume and isodose models before adding new ones.

## Export formats

### Seed and tile positions

**CSV** contains three blocks separated by blank lines:

1. One row per tile seed: `tile_index, seed_index, x_ras_mm, y_ras_mm, z_ras_mm, x_local_mm, y_local_mm, z_local_mm`.
2. One row per tile: `tile_index, center_x_ras, center_y_ras, center_z_ras, normal_x, normal_y, normal_z, seed_axis_x, seed_axis_y, seed_axis_z`.
3. One row per standalone seed: `standalone_seed_index, x_ras_mm, y_ras_mm, z_ras_mm, normal_x, normal_y, normal_z`.

After the dose has been computed, a fourth block lists the RAS bounding box of each isodose surface.

**JSON** contains `tiles` (each with `tile_index`, `center_ras_mm`, `normal_ras`, `seed_axis_ras` and a `seeds` list of `seed_index`, `position_ras_mm`, `position_local_mm`), `standalone_seeds` (`seed_index`, `center_ras_mm`, `normal_ras`) and `isodose_surface_bounds` (name, `dose_level_cGy` and per-axis `bounds_ras_mm`).

### Dose volume

The export tries three writers in order and reports which one succeeded in the status line:

| Writer | Output | Units |
|---|---|---|
| SlicerRT (if installed) | DICOM RTDOSE study | Gy |
| Slicer's *Create a DICOM Series* CLI | DICOM Secondary Capture series (`DOSE0001.dcm` …), one file per slice, 32-bit signed integer pixels, series description *GammaTile dose (cGy)* | cGy, exact to 1 cGy |
| NRRD | `GammaTile_dose.nrrd`, 32-bit float | Gy |

The DICOM image-series writer stores integer centigray rather than gray because this Slicer version does not preserve a non-unit RescaleSlope for these modalities, and integer gray would lose all sub-gray resolution.

## Architecture

| Component | Responsibility |
|---|---|
| `DoseInterpolator` | Vectorized TG-43U1 dose for arrays of (angle, radius); holds the seed constants, the g_L fit and the anisotropy table |
| `GammaTilePlacement` | Geometry of one placed tile: centre, normal, local frame, four seed positions and the shared seed axis; vectorized dose from its four seeds |
| `SingleSeedPlacement` | Geometry and dose of one standalone seed |
| `GammaTileDosimetry` | Scene-independent logic: lists of tiles and seeds, undo, and the masked, vectorized dose-grid computation |
| `BrainSurfaceGenerator` | Threshold, isosurface, smoothing, and creation of the RAS-space brain model |
| `IsodoseSurfaceGenerator` | Isosurface extraction at a dose level and model creation |
| `GammaTileModule` / `GammaTile` | Slicer module registration (title, category, contributors) |
| `GammaTileWidget` | The panel UI and all interaction: view observers, click-to-surface ray casting, placement, selection, drag, rotate, translate, dose update, legend and exports |
| `GammaTileLogic` | Slicer logic entry point |

Internally the widget keeps two caches: a NumPy copy of the selected volume with its RAS-to-IJK matrix (invalidated whenever the pixel data or the node geometry changes) for ray marching and surface settling, and a static cell locator for the current brain mesh (only the most recent mesh is retained).

## Performance

| Operation | Method |
|---|---|
| Dose grid | All voxel centres within range are evaluated in a handful of NumPy array operations per seed rather than a per-voxel Python loop. A 96³ grid with three tiles and a seed computes in about 0.5 s inside Slicer. |
| Brain and isodose surfaces | `vtkFlyingEdges3D` (multithreaded), which produces the same mesh as marching cubes |
| Surface normals and closest points | VTK static point and cell locators |
| Ray casting and surface settling | Whole rays are sampled at once from the cached NumPy volume |

## Validation and testing

Two independent checks were run on this version.

**Numerical equivalence with the previous per-voxel implementation.** Both implementations were imported side by side and compared on 20,000 random (angle, radius) pairs, 5,000 random points around a tile and a seed, and a full 41 × 41 × 37 dose grid with two tiles and one seed. Doses agree to a relative difference of 1.2 × 10⁻⁷ or better everywhere (float32 rounding), the set of non-zero voxels is identical, and flying-edges surfaces are vertex-for-vertex identical to marching cubes at three isodose levels. The only intentional difference is within 0.025 mm of a seed tip on the seed axis, where the old code divided by zero.

**Headless feature test inside Slicer.** `tests/slicer_feature_test.py` loads the module through Slicer's module factory, builds a synthetic brain volume, and exercises every feature: brain surface creation, tile placement through the 3D interactor and through slice-view ray casting, seed placement, selection, rotation, translation, dose update (twice), isodose surfaces and legend, CSV and JSON export, the Gy conversion, DICOM series export with a pydicom read-back, NRRD export, undo, volume ray casting, cache invalidation and cleanup. It prints one `[TEST] PASS/FAIL` line per check (40 checks) and exits with code 0 when all pass:

```bash
/Applications/Slicer.app/Contents/MacOS/Slicer --testing --no-splash --python-script tests/slicer_feature_test.py
```

Set `GAMMATILE_TEST_OUT` to keep the exported files and `results.json`.

## Repository layout

| Path | Purpose |
|---|---|
| `GammaTile.py` | The Slicer module |
| `tests/slicer_feature_test.py` | Headless end-to-end test, run inside Slicer |
| `README.md` | This document |

## Version notes

This version merges the previously separate performance-optimized build into the single `GammaTile.py` module. The per-voxel implementation has been retired. Fixes made while consolidating:

- The brain surface mesh is now stored in RAS coordinates. Previously it was kept in voxel coordinates under a display transform, so every RAS-space query against it (surface normal lookup, closest-point search, ray intersection, penetration limiting and tile deformation) silently missed the mesh and fell back to volume-based estimates.
- The Gy-scaled dose copy used by the RTDOSE and NRRD exports failed to build because of an incorrect matrix accessor call, so exports were always in cGy.
- DICOM series export always failed (the CLI rejects modality "OT") and silently fell back to NRRD; it now writes a Secondary Capture series with 32-bit integer pixels so near-seed doses cannot overflow.
- The volume sampling cache is invalidated when the volume node's geometry changes, not only when its pixel data changes.
- The mesh locator cache keeps only the current brain surface instead of every surface ever created.
- A point exactly at a seed tip on the seed axis no longer raises a division-by-zero error during dose computation.

## Contributors

Jacob Lee, UMN Brachytherapy Research Lab. Clara Ferreira, UMN Department of Radiation Oncology.
