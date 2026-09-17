"""
GammaView Dosimetry - 3D Slicer Module
Works with DICOM brain volumes for intraoperative GammaTile placement

The dose grid is evaluated with vectorized NumPy (no per-voxel Python loops),
isodose/brain surfaces use vtkFlyingEdges3D, and volume ray casting samples a
cached NumPy array instead of per-voxel VTK calls.
"""

import json
import csv
import os
import uuid
import numpy as np
import vtk
from vtk.util import numpy_support
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import ctk
import logging

logger = logging.getLogger(__name__)


class DoseInterpolator:
    """TG-43U1 2D formalism for IsoRay Proxcelan CS-1 Rev2 (L=0.40 cm).

    Two dosimetry datasets are built in; DATASET selects which one is used:

    * "TG43U1S2" (default): AAPM/GEC-ESTRO consensus data from TG-43U1S2,
      Rivard et al., Med. Phys. 44(9):e297-e338 (2017), doi:10.1002/mp.12430.
      Lambda and L from Table AI, g_L(r) from Table AII (line-source column
      "IsoRay CS-1 Rev2"), F(r, theta) from Table AXIII.
    * "CLRP_V2": CLRP TG-43 parameter database v2 (2019) egs_brachy Monte Carlo
      data for the same source (polynomial g_L fit and 32x12 F table).
    """

    DATASET = "TG43U1S2"  # or "CLRP_V2"
    DATASET_NAMES = {
        "TG43U1S2": "TG-43U1S2 consensus (Rivard et al. 2017)",
        "CLRP_V2": "CLRP TG-43 v2 (2019)",
    }

    DOSE_CONVERSION_FACTOR = 335.23 * 3.5  # cGy per air kerma strength; math was derived
    _LAMBDA = 1.056 #Dose rate constant (cGy h^-1 U^-1), TG-43U1S2 Table AI
    _L = 0.40 # Active source length (cm), TG-43U1S2 Table AI

    # ---- TG-43U1S2 consensus dataset ----
    # CONg_L(r), Table AII (line-source approximation), linearly interpolated in r
    _CON_GL_RADII = np.array([
        0.10, 0.15, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00,
        5.00, 6.00, 7.00, 8.00, 9.00, 10.00
    ], dtype=np.float64)
    _CON_GL_VALUES = np.array([
        0.960, 0.971, 0.989, 1.006, 1.009, 1.000, 0.962, 0.908, 0.777, 0.642,
        0.518, 0.411, 0.323, 0.251, 0.1931, 0.1481
    ], dtype=np.float64)
    # CONF(r, theta), Table AXIII. Entries at r = 0.1 cm are not tabulated for
    # theta <= 20 deg (np.nan, filled from the nearest tabulated radius). A row at
    # 90 deg (F = 1 by definition) anchors interpolation up to the polar-angle clip.
    _CON_F_THETAS = np.array([
        0, 2, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90
    ], dtype=np.float64)
    _CON_F_RADII = np.array([
        0.10, 0.25, 0.50, 0.70, 1.00, 2.00, 3.00, 5.00, 7.00, 10.00
    ], dtype=np.float64)
    _CON_F_TABLE = np.array([
        #  0.10    0.25   0.50   0.70   1.00   2.00   3.00   5.00   7.00  10.00
        [np.nan, 0.622, 0.829, 0.851, 0.845, 0.837, 0.838, 0.843, 0.853, 0.833],  # 0
        [np.nan, 0.670, 0.796, 0.806, 0.818, 0.839, 0.842, 0.846, 0.845, 0.847],  # 2
        [np.nan, 0.731, 0.781, 0.761, 0.750, 0.764, 0.780, 0.800, 0.810, 0.819],  # 5
        [np.nan, 0.730, 0.718, 0.706, 0.713, 0.746, 0.769, 0.794, 0.808, 0.816],  # 7
        [np.nan, 0.707, 0.678, 0.690, 0.709, 0.752, 0.776, 0.801, 0.815, 0.826],  # 10
        [np.nan, 0.759, 0.720, 0.734, 0.753, 0.791, 0.811, 0.831, 0.840, 0.847],  # 15
        [np.nan, 0.843, 0.778, 0.788, 0.803, 0.832, 0.847, 0.862, 0.868, 0.871],  # 20
        [ 1.161, 0.887, 0.828, 0.834, 0.845, 0.866, 0.876, 0.886, 0.891, 0.896],  # 25
        [ 1.113, 0.915, 0.868, 0.870, 0.878, 0.894, 0.902, 0.910, 0.912, 0.916],  # 30
        [ 1.031, 0.949, 0.923, 0.923, 0.926, 0.935, 0.939, 0.943, 0.944, 0.944],  # 40
        [ 1.009, 0.971, 0.958, 0.957, 0.959, 0.963, 0.964, 0.966, 0.967, 0.966],  # 50
        [ 1.002, 0.985, 0.978, 0.978, 0.980, 0.981, 0.982, 0.982, 0.982, 0.983],  # 60
        [ 1.000, 0.993, 0.990, 0.991, 0.992, 0.993, 0.993, 0.993, 0.993, 0.992],  # 70
        [ 1.000, 0.998, 0.998, 0.997, 0.998, 0.999, 0.999, 0.999, 0.999, 0.998],  # 80
        [ 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 90
    ], dtype=np.float64)

    # ---- CLRP TG-43 v2 (2019) dataset ----
    # g_L(r) = (a0*r^-2 + a1*r^-1 + a2 + a3*r + a4*r^2 + a5*r^3) * exp(-a6*r), fit valid 0.05-10 cm
    _CLRP_GR_COEFFS = np.array([7.38e-04, -1.198e-02, 9.991e-01, 4.979e-01,
                                1.07e-02, 1.39e-03, 4.055e-01])
    # Anisotropy table F(r, theta) — V2 (2019) Proxcelan CS-1 Rev2
    _CLRP_F_THETAS = np.array([
        0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 73, 75, 78, 80, 82, 84, 85, 86, 87, 88, 89, 90
    ], dtype=np.float64)

    _CLRP_F_RADII = np.array([
        0.10, 0.15, 0.25, 0.50, 0.75, 1.00, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00
    ], dtype=np.float64)
    _CLRP_F_TABLE = np.array([
        #  0.10   0.15   0.25   0.50   0.75   1.00   2.00   3.00   4.00   5.00   7.50  10.00
        [  np.nan, np.nan, 0.617, 0.883, 0.903, 0.900, 0.889, 0.884, 0.882, 0.882, 0.880, 0.874],  # 0
        [  np.nan, np.nan, 0.648, 0.864, 0.890, 0.893, 0.891, 0.889, 0.887, 0.885, 0.879, 0.875],  # 1
        [  np.nan, np.nan, 0.679, 0.847, 0.868, 0.878, 0.877, 0.882, 0.878, 0.875, 0.869, 0.866],  # 2
        [  np.nan, np.nan, 0.709, 0.837, 0.860, 0.867, 0.850, 0.847, 0.848, 0.849, 0.850, 0.852],  # 3
        [  np.nan, np.nan, 0.762, 0.822, 0.793, 0.779, 0.787, 0.798, 0.807, 0.814, 0.826, 0.832],  # 5
        [  np.nan, np.nan, 0.786, 0.742, 0.726, 0.730, 0.761, 0.781, 0.794, 0.803, 0.818, 0.827],  # 7
        [  np.nan, np.nan, 0.731, 0.688, 0.702, 0.718, 0.760, 0.783, 0.797, 0.807, 0.822, 0.830],  # 10
        [  np.nan, np.nan, 0.696, 0.694, 0.714, 0.730, 0.772, 0.794, 0.807, 0.817, 0.830, 0.838],  # 12
        [  np.nan, np.nan, 0.761, 0.724, 0.743, 0.758, 0.795, 0.814, 0.826, 0.834, 0.845, 0.852],  # 15
        [  np.nan, 1.141, 0.845, 0.780, 0.794, 0.806, 0.835, 0.849, 0.858, 0.863, 0.871, 0.875],  # 20
        [  1.143, 1.051, 0.889, 0.829, 0.838, 0.847, 0.868, 0.879, 0.885, 0.888, 0.894, 0.897],  # 25
        [  1.131, 1.016, 0.916, 0.868, 0.873, 0.880, 0.895, 0.903, 0.907, 0.910, 0.914, 0.915],  # 30
        [  1.074, 1.001, 0.935, 0.899, 0.902, 0.906, 0.918, 0.923, 0.926, 0.928, 0.930, 0.932],  # 35
        [  1.040, 0.994, 0.950, 0.923, 0.924, 0.927, 0.936, 0.940, 0.942, 0.943, 0.944, 0.946],  # 40
        [  1.021, 0.993, 0.962, 0.942, 0.942, 0.944, 0.951, 0.953, 0.955, 0.956, 0.956, 0.957],  # 45
        [  1.013, 0.993, 0.972, 0.958, 0.957, 0.959, 0.963, 0.965, 0.966, 0.966, 0.967, 0.968],  # 50
        [  1.007, 0.994, 0.979, 0.969, 0.970, 0.970, 0.973, 0.974, 0.975, 0.975, 0.975, 0.976],  # 55
        [  1.004, 0.995, 0.985, 0.978, 0.979, 0.980, 0.981, 0.982, 0.982, 0.982, 0.982, 0.983],  # 60
        [  1.002, 0.996, 0.990, 0.985, 0.986, 0.987, 0.988, 0.988, 0.988, 0.988, 0.988, 0.989],  # 65
        [  1.001, 0.998, 0.994, 0.990, 0.991, 0.992, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993],  # 70
        [  1.001, 0.998, 0.996, 0.993, 0.993, 0.994, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995],  # 73
        [  1.001, 0.999, 0.997, 0.995, 0.995, 0.995, 0.996, 0.996, 0.996, 0.996, 0.996, 0.997],  # 75
        [  1.000, 0.999, 0.998, 0.997, 0.997, 0.997, 0.998, 0.998, 0.998, 0.998, 0.998, 0.998],  # 78
        [  1.000, 0.999, 0.999, 0.998, 0.998, 0.998, 0.998, 0.999, 0.999, 0.999, 0.999, 0.998],  # 80
        [  1.000, 1.000, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999],  # 82
        [  1.000, 1.000, 0.999, 0.999, 0.999, 0.999, 0.999, 1.000, 1.000, 0.999, 1.000, 1.000],  # 84
        [  1.000, 1.000, 1.000, 0.999, 1.000, 1.000, 0.999, 1.000, 1.000, 1.000, 1.000, 1.000],  # 85
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 86
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 87
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 88
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 89
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 90
    ], dtype=np.float64)

    def __init__(self, dataset=None):
        self.dataset = dataset or self.DATASET
        if self.dataset not in self.DATASET_NAMES:
            raise ValueError(f"Unknown dosimetry dataset {self.dataset!r}; expected one of {list(self.DATASET_NAMES)}")
        self.dataset_name = self.DATASET_NAMES[self.dataset]
        if self.dataset == "TG43U1S2":
            self._F_THETAS = self._CON_F_THETAS
            self._F_RADII = self._CON_F_RADII
            self._F_TABLE = self._CON_F_TABLE
        else:
            self._F_THETAS = self._CLRP_F_THETAS
            self._F_RADII = self._CLRP_F_RADII
            self._F_TABLE = self._CLRP_F_TABLE
        # Precompute reference geometry factor G_L(r0=1.0, theta0=90°)
        self._GL_ref = self._geometry_factor(1.0, 90.0)
        # NaN-free copy of the anisotropy table so lookups can be vectorized.
        # Fill rule matches the original scalar lookup: nearest valid value
        # searching increasing radius first, then decreasing radius.
        table = self._F_TABLE.copy()
        n_r = table.shape[1]
        for ti in range(table.shape[0]):
            for ri in range(n_r):
                if np.isnan(self._F_TABLE[ti, ri]):
                    filled = 1.0
                    right = np.where(~np.isnan(self._F_TABLE[ti, ri + 1:]))[0]
                    if right.size:
                        filled = self._F_TABLE[ti, ri + 1 + right[0]]
                    else:
                        left = np.where(~np.isnan(self._F_TABLE[ti, :ri]))[0]
                        if left.size:
                            filled = self._F_TABLE[ti, left[-1]]
                    table[ti, ri] = filled
        self._F_TABLE_FILLED = table

    def _geometry_factor(self, r, theta_deg):
        """Line-source geometry function G_L(r, theta)."""
        L = self._L
        theta_rad = np.radians(theta_deg)
        if abs(theta_deg) < 0.01:
            # On-axis approximation
            return 1.0 / (r * r - (L / 2.0) ** 2)
        sin_t = np.sin(theta_rad)
        cos_t = np.cos(theta_rad)
        r_sin = r * sin_t
        r_cos = r * cos_t
        beta = np.arctan2(r_sin, r_cos - L / 2.0) - np.arctan2(r_sin, r_cos + L / 2.0)
        return beta / (L * r * sin_t)

    def _geometry_factor_vec(self, r, theta_deg):
        """Vectorized line-source geometry function G_L(r, theta) for arrays."""
        L = self._L
        theta_rad = np.radians(theta_deg)
        sin_t = np.sin(theta_rad)
        cos_t = np.cos(theta_rad)
        r_sin = r * sin_t
        r_cos = r * cos_t
        beta = np.arctan2(r_sin, r_cos - L / 2.0) - np.arctan2(r_sin, r_cos + L / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            G = beta / (L * r * sin_t)
        # On-axis approximation where sin(theta) ~ 0. Guard the singularity at
        # r = L/2 (point exactly at the source tip): clamp the denominator so the
        # dose stays finite, at a magnitude comparable to the closest allowed
        # perpendicular distance (the original scalar code returns inf here).
        denom = r * r - (L / 2.0) ** 2
        denom = np.where(np.abs(denom) < 1e-3, np.copysign(1e-3, denom + 1e-300), denom)
        return np.where(np.abs(theta_deg) < 0.01, 1.0 / denom, G)

    def _radial_dose(self, r):
        """Radial dose function g_L(r) for arrays of r (cm)."""
        if self.dataset == "TG43U1S2":
            # Linear interpolation on the tabulated consensus values; held constant
            # below 0.10 cm and above 10 cm (np.interp clamps to the end values).
            return np.interp(np.asarray(r, dtype=np.float64), self._CON_GL_RADII, self._CON_GL_VALUES)
        # CLRP v2 modified-polynomial fit (vectorized), valid 0.05-10 cm
        r = np.clip(r, 0.05, 10.0)
        a = self._CLRP_GR_COEFFS
        return (a[0] * r**-2 + a[1] * r**-1 + a[2] + a[3] * r
                + a[4] * r**2 + a[5] * r**3) * np.exp(-a[6] * r)

    def _anisotropy(self, r, theta_deg):
        """Anisotropy function F(r, theta): vectorized bilinear interpolation on the (NaN-filled) table."""
        thetas = self._F_THETAS
        radii = self._F_RADII
        table = self._F_TABLE_FILLED

        theta_c = np.clip(theta_deg, thetas[0], thetas[-1])
        r_c = np.clip(r, radii[0], radii[-1])
        it = np.clip(np.searchsorted(thetas, theta_c, side='right') - 1, 0, len(thetas) - 2)
        ir = np.clip(np.searchsorted(radii, r_c, side='right') - 1, 0, len(radii) - 2)

        v00 = table[it, ir]
        v01 = table[it, ir + 1]
        v10 = table[it + 1, ir]
        v11 = table[it + 1, ir + 1]

        # Interpolation fractions
        dr = radii[ir + 1] - radii[ir]
        t_r = np.where(dr > 1e-12, (r_c - radii[ir]) / dr, 0.0)
        dt = thetas[it + 1] - thetas[it]
        t_t = np.where(dt > 1e-12, (theta_c - thetas[it]) / dt, 0.0)

        # Bilinear
        top = v00 * (1.0 - t_r) + v01 * t_r
        bot = v10 * (1.0 - t_r) + v11 * t_r
        return top * (1.0 - t_t) + bot * t_t

    def dose_at_points(self, angle_deg, radius_cm):
        """Vectorized TG-43U1 dose for arrays of (angle_deg, radius_cm); returns cGy array."""
        theta = np.mod(np.asarray(angle_deg, dtype=np.float64), 360.0)
        theta = np.where(theta > 180.0, 360.0 - theta, theta)
        theta = np.where(theta > 90.0, 180.0 - theta, theta)
        theta = np.clip(theta, 0.0, 90.0)
        r = np.clip(np.asarray(radius_cm, dtype=np.float64), 0.041, 10.0)

        GL = self._geometry_factor_vec(r, theta)
        gL = self._radial_dose(r)
        F = self._anisotropy(r, theta)
        rate = self._LAMBDA * (GL / self._GL_ref) * gL * F
        return self.DOSE_CONVERSION_FACTOR * np.maximum(0.0, rate)

    def dose_at_point(self, angle_deg, radius_cm):
        """TG-43U1 dose at a single point, returns cGy."""
        return float(self.dose_at_points(np.array([angle_deg], dtype=np.float64),
                                         np.array([radius_cm], dtype=np.float64))[0])


class GammaTilePlacement:
    """Single tile placement with accurate dimensions."""
    TILE_LENGTH_MM = 20.0
    TILE_WIDTH_MM = 20.0
    TILE_THICKNESS_MM = 4.0
    SOURCE_DEPTH_MM = 2.0
    SEEDS_PER_TILE = 4
    SEED_EDGE_OFFSET_MM = 5.0
    SEED_SPACING_MM = 10.0
    MAX_PENETRATION_MM = 10.0
    SEED_LENGTH_MM = 4.5
    SEED_DIAMETER_MM = 0.82
    SEED_RADIUS_MM = SEED_DIAMETER_MM / 2.0
    
    def __init__(self, center_ras, normal_vector, spacing_mm=1.0):
        self.center = np.array(center_ras, dtype=np.float32)
        self.normal = np.array(normal_vector, dtype=np.float32)
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-6)
        self.spacing_mm = spacing_mm
        
        # Create local coordinate frame
        z = self.normal
        if abs(z[0]) < 0.9:
            x_candidate = np.array([1, 0, 0], dtype=np.float32)
        else:
            x_candidate = np.array([0, 1, 0], dtype=np.float32)
        
        x = np.cross(z, x_candidate)
        x = x / (np.linalg.norm(x) + 1e-6)
        y = np.cross(z, x)
        
        self.frame = np.array([x, y, z]).T
        
        # Four seeds in square, 5mm from tile edges (tile is 20mm, edges at ±10mm)
        tile_half = self.TILE_LENGTH_MM / 2.0
        seed_edge_offset = self.SEED_EDGE_OFFSET_MM
        seed_x_pos = tile_half - seed_edge_offset
        seed_z_position = 0.0  # Midpoint of 4mm thickness (2mm from each edge)
        self.seed_positions_local = [
            np.array([-seed_x_pos, -seed_x_pos, seed_z_position], dtype=np.float32),
            np.array([+seed_x_pos, -seed_x_pos, seed_z_position], dtype=np.float32),
            np.array([-seed_x_pos, +seed_x_pos, seed_z_position], dtype=np.float32),  
            np.array([+seed_x_pos, +seed_x_pos, seed_z_position], dtype=np.float32)
        ]
        self.seed_positions_ras = []
        for seed_local in self.seed_positions_local:
            seed_ras = self.center + self.frame @ seed_local
            self.seed_positions_ras.append(seed_ras)
        self.seed_axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # +X in tile local coords (in-plane, parallel to tile surface)
        self.seed_axis_ras = self.frame @ self.seed_axis_local  # Transform to RAS
        self.seed_axis_ras = self.seed_axis_ras / (np.linalg.norm(self.seed_axis_ras) + 1e-6)  # Normalize
    
    def dose_at_points(self, points_ras, interpolator):
        """Vectorized total dose from all 4 seeds at an (N, 3) array of RAS points; returns (N,) cGy."""
        points = np.asarray(points_ras, dtype=np.float64)
        total_dose = np.zeros(points.shape[0], dtype=np.float64)
        seed_axis = np.asarray(self.seed_axis_ras, dtype=np.float64)

        for seed_pos in self.seed_positions_ras:
            relative = points - np.asarray(seed_pos, dtype=np.float64)
            r_total = np.linalg.norm(relative, axis=1)
            valid = (r_total >= self.SEED_DIAMETER_MM / 2.0) & (r_total <= 100)
            if not np.any(valid):
                continue
            rel = relative[valid]
            r_parallel = rel @ seed_axis
            r_perp = np.linalg.norm(rel - r_parallel[:, None] * seed_axis, axis=1)
            theta_deg = np.degrees(np.arctan2(r_perp, np.abs(r_parallel)))
            r_cm = r_total[valid] / 10.0
            total_dose[valid] += interpolator.dose_at_points(theta_deg, r_cm)

        return total_dose

    def dose_at_point(self, point_ras, interpolator):
        """Total dose from all 4 seeds, TG-43 polar angle from seed axis."""
        return float(self.dose_at_points(np.asarray(point_ras, dtype=np.float64).reshape(1, 3),
                                         interpolator)[0])


class SingleSeedPlacement:
    """Single Cs-131 seed placement (standalone, not part of a tile). Same TG-43 dosimetry as one seed in a tile."""
    SEED_LENGTH_MM = GammaTilePlacement.SEED_LENGTH_MM
    SEED_DIAMETER_MM = GammaTilePlacement.SEED_DIAMETER_MM

    def __init__(self, center_ras, normal_vector):
        self.center = np.array(center_ras, dtype=np.float32)
        self.normal = np.array(normal_vector, dtype=np.float32)
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-6)
        # Seed longitudinal axis = normal (seed oriented along surface normal)
        self.seed_axis_ras = self.normal.copy()

    def dose_at_points(self, points_ras, interpolator):
        """Vectorized dose from this single seed at an (N, 3) array of RAS points; returns (N,) cGy."""
        points = np.asarray(points_ras, dtype=np.float64)
        total_dose = np.zeros(points.shape[0], dtype=np.float64)
        seed_axis = np.asarray(self.seed_axis_ras, dtype=np.float64)

        relative = points - np.asarray(self.center, dtype=np.float64)
        r_total = np.linalg.norm(relative, axis=1)
        valid = (r_total >= GammaTilePlacement.SEED_DIAMETER_MM / 2.0) & (r_total <= 100)
        if np.any(valid):
            rel = relative[valid]
            r_parallel = rel @ seed_axis
            r_perp = np.linalg.norm(rel - r_parallel[:, None] * seed_axis, axis=1)
            theta_deg = np.degrees(np.arctan2(r_perp, np.abs(r_parallel)))
            r_cm = r_total[valid] / 10.0
            total_dose[valid] = interpolator.dose_at_points(theta_deg, r_cm)

        return total_dose

    def dose_at_point(self, point_ras, interpolator):
        """Dose from this single seed at point_ras (TG-43)."""
        return float(self.dose_at_points(np.asarray(point_ras, dtype=np.float64).reshape(1, 3),
                                         interpolator)[0])


class GammaTileDosimetry:
    """Main coordinator for tile placement and dosimetry."""

    def __init__(self):
        self.interpolator = DoseInterpolator()
        self.tiles = []
        self.seeds = []  # Standalone SingleSeedPlacement instances
        self.dose_volume = None
        self.prescription_gy = 60
        self.fiducial_nodes = []
        self.seed_fiducial_nodes = []  # Model nodes for standalone seeds
        self.brain_surface_model = None
        self.isodose_nodes_3d = []
        self.dose_volume_node = None

    def place_tile(self, click_point_ras, normal, spacing_mm=1.0):
        """Place tile at clicked point with proper normal."""
        point = np.array(click_point_ras)
        normal = np.array(normal, dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-6)

        tile = GammaTilePlacement(point, normal, spacing_mm)
        self.tiles.append(tile)

        return point, normal

    def place_seed(self, center_ras, normal):
        """Place a single standalone seed at center_ras with axis along normal."""
        center = np.array(center_ras)
        normal = np.array(normal, dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        seed = SingleSeedPlacement(center, normal)
        self.seeds.append(seed)
        return center, normal
    
    def compute_dose_volume_optimized(self, bounds_ras, spacing_mm=2.0, max_distance_mm=75.0):
        """Compute dose grid; only voxels within max_distance_mm of any tile/seed are evaluated."""
        dims = [int((bounds_ras[i][1] - bounds_ras[i][0]) / spacing_mm) + 1 for i in range(3)]
        volume = np.zeros(dims, dtype=np.float32)

        if not self.tiles and not self.seeds:
            return volume, spacing_mm, dims, bounds_ras

        # Create coordinate grids
        x_coords = np.arange(bounds_ras[0][0], bounds_ras[0][1] + spacing_mm, spacing_mm)[:dims[0]]
        y_coords = np.arange(bounds_ras[1][0], bounds_ras[1][1] + spacing_mm, spacing_mm)[:dims[1]]
        z_coords = np.arange(bounds_ras[2][0], bounds_ras[2][1] + spacing_mm, spacing_mm)[:dims[2]]

        # All voxel centers as one (N, 3) array, in volume[i, j, k] order
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # Mask: voxels within max_distance_mm of any tile/seed center
        max_dist_sq = max_distance_mm ** 2
        min_dist_sq = np.full(points.shape[0], np.inf)
        for source in list(self.tiles) + list(self.seeds):
            center = np.asarray(source.center, dtype=np.float64)
            dist_sq = np.sum((points - center) ** 2, axis=1)
            np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
        mask = min_dist_sq <= max_dist_sq

        if np.any(mask):
            active_points = points[mask]
            dose = np.zeros(active_points.shape[0], dtype=np.float64)
            for tile in self.tiles:
                dose += tile.dose_at_points(active_points, self.interpolator)
            for seed in self.seeds:
                dose += seed.dose_at_points(active_points, self.interpolator)
            flat = np.zeros(points.shape[0], dtype=np.float32)
            flat[mask] = dose.astype(np.float32)
            volume = flat.reshape(dims)

        self.dose_volume = volume
        return volume, spacing_mm, dims, bounds_ras
    
    def remove_tile_at_index(self, index):
        if index < 0 or index >= len(self.tiles):
            return
        self.tiles.pop(index)
        if index < len(self.fiducial_nodes):
            node = self.fiducial_nodes.pop(index)
            if node is not None and node.GetScene():
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except RuntimeError as e:
                    logger.warning("Could not remove tile node from scene: %s", e)
        if self.isodose_nodes_3d:
            for node in self.isodose_nodes_3d:
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
            self.isodose_nodes_3d = []
        if self.dose_volume_node and self.dose_volume_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.dose_volume_node)
            self.dose_volume_node = None

    def remove_seed_at_index(self, index):
        if index < 0 or index >= len(self.seeds):
            return
        self.seeds.pop(index)
        if index < len(self.seed_fiducial_nodes):
            node = self.seed_fiducial_nodes.pop(index)
            if node is not None and node.GetScene():
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except RuntimeError as e:
                    logger.warning("Could not remove seed node from scene: %s", e)
        if self.isodose_nodes_3d:
            for node in self.isodose_nodes_3d:
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
            self.isodose_nodes_3d = []
        if self.dose_volume_node and self.dose_volume_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.dose_volume_node)
            self.dose_volume_node = None

    def undo_tile(self):
        if self.tiles:
            self.tiles.pop()
            if self.fiducial_nodes:
                node = self.fiducial_nodes.pop()
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
        
        if self.isodose_nodes_3d:
            for node in self.isodose_nodes_3d:
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
            self.isodose_nodes_3d = []
        
        if self.dose_volume_node and self.dose_volume_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.dose_volume_node)
            self.dose_volume_node = None

    def undo_seed(self):
        if self.seeds:
            self.seeds.pop()
            if self.seed_fiducial_nodes:
                node = self.seed_fiducial_nodes.pop()
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
        if self.isodose_nodes_3d:
            for node in self.isodose_nodes_3d:
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
            self.isodose_nodes_3d = []
        if self.dose_volume_node and self.dose_volume_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.dose_volume_node)
            self.dose_volume_node = None

    def clear(self):
        self.tiles = []
        self.seeds = []
        for node in self.seed_fiducial_nodes:
            if node and node.GetScene():
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except RuntimeError as e:
                    logger.warning("Could not remove seed node from scene: %s", e)
        self.seed_fiducial_nodes = []
        if self.brain_surface_model:
            slicer.mrmlScene.RemoveNode(self.brain_surface_model)
            self.brain_surface_model = None


class BrainSurfaceGenerator:
    """Generate brain surface model from volume."""
    
    @staticmethod
    def create_brain_surface(volumeNode, threshold=None):
        imageData = volumeNode.GetImageData()
        if not imageData:
            return None

        scalars = imageData.GetPointData().GetScalars()
        if not scalars:
            return None

        dims = imageData.GetDimensions()
        np_array = numpy_support.vtk_to_numpy(scalars)
        np_array = np_array.reshape(dims[::-1])

        if threshold is None:
            threshold = np.mean(np_array) + 0.5 * np.std(np_array)

        binary = (np_array > threshold).astype(np.uint8)

        binary_vtk = numpy_support.numpy_to_vtk(
            binary.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )

        binary_image = vtk.vtkImageData()
        binary_image.SetDimensions(dims)
        binary_image.SetSpacing(imageData.GetSpacing())
        binary_image.SetOrigin(imageData.GetOrigin())
        binary_image.GetPointData().SetScalars(binary_vtk)

        # vtkFlyingEdges3D produces the same isosurface as vtkMarchingCubes
        # but is dramatically faster (parallel edge-based algorithm)
        marching = vtk.vtkFlyingEdges3D() if hasattr(vtk, 'vtkFlyingEdges3D') else vtk.vtkMarchingCubes()
        marching.SetInputData(binary_image)
        marching.SetValue(0, 0.5)
        marching.Update()

        surface = marching.GetOutput()

        if surface.GetNumberOfPoints() == 0:
            return None

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(surface)
        smoother.SetNumberOfIterations(10)
        smoother.SetPassBand(0.1)
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        return smoother.GetOutput()
    
    @staticmethod
    def add_surface_to_scene(surface, volumeNode, name="BrainSurface"):
        # Bake the volume's IJK->RAS transform into the mesh itself. The surface is
        # extracted in voxel (IJK) space; every downstream query against it
        # (normal lookup, closest point, ray intersection, penetration limiting,
        # tile deformation) is made with RAS coordinates, so the stored polydata
        # must be in RAS rather than displayed through a transform node.
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        transform = vtk.vtkTransform()
        transform.SetMatrix(ijkToRas)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(surface)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        surface_ras = transformFilter.GetOutput()
        if ijkToRas.Determinant() < 0:
            # A mirroring transform (e.g. LPS->RAS axis flips) reverses triangle
            # winding; restore it so cell normals keep pointing outward.
            reverse = vtk.vtkReverseSense()
            reverse.SetInputData(surface_ras)
            reverse.ReverseCellsOn()
            reverse.ReverseNormalsOff()
            reverse.Update()
            surface_ras = reverse.GetOutput()

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        modelNode.SetAndObservePolyData(surface_ras)
        
        displayNode = modelNode.GetDisplayNode()
        if displayNode is None:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        
        displayNode.SetColor(0.8, 0.8, 0.9)  # Light blue-gray
        displayNode.SetOpacity(0.6)
        displayNode.SetBackfaceCulling(False)
        displayNode.SetVisibility(True)
        
        return modelNode


class IsodoseSurfaceGenerator:
    """Generate isodose surfaces for visualization."""
    
    @staticmethod
    def create_isodose_surface(dose_volume, dose_level, spacing, origin):
        max_dose = np.max(dose_volume)
        if max_dose < dose_level:
            return None

        if dose_volume.size == 0:
            return None

        dose_flat = np.ascontiguousarray(dose_volume.ravel(order='F'), dtype=np.float32)
        volume_vtk = numpy_support.numpy_to_vtk(dose_flat, deep=True, array_type=vtk.VTK_FLOAT)
        
        image = vtk.vtkImageData()
        image.SetDimensions(dose_volume.shape)
        image.SetSpacing(spacing, spacing, spacing)
        image.SetOrigin(0, 0, 0)
        image.GetPointData().SetScalars(volume_vtk)

        contour = vtk.vtkFlyingEdges3D() if hasattr(vtk, 'vtkFlyingEdges3D') else vtk.vtkMarchingCubes()
        contour.SetInputData(image)
        contour.SetValue(0, float(dose_level))
        # Enable all regions to ensure we get surfaces even for large/low dose regions
        contour.SetComputeNormals(True)
        contour.SetComputeGradients(False)
        contour.SetComputeScalars(False)
        contour.Update()

        surface = contour.GetOutput()

        if surface is None or surface.GetNumberOfPoints() == 0:
            return None

        transform = vtk.vtkTransform()
        transform.Translate(origin[0], origin[1], origin[2])
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(surface)
        transformFilter.Update()
        surface = transformFilter.GetOutput()

        if surface.GetNumberOfPoints() > 0:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(surface)
            smoother.SetNumberOfIterations(5)  # Reduced for performance
            smoother.SetPassBand(0.1)
            smoother.NormalizeCoordinatesOff()
            smoother.Update()
            return smoother.GetOutput()
        
        return surface
    
    @staticmethod
    def add_isodose_to_scene(surface, dose_level, color, name_prefix="Isodose"):
        if surface is None or surface.GetNumberOfPoints() == 0:
            return None
        
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_{dose_level:.0f}cGy")
        modelNode.SetAndObservePolyData(surface)
        modelNode.SetAttribute("DoseLevel_cGy", f"{dose_level:.1f}")
        
        displayNode = modelNode.GetDisplayNode()
        if displayNode is None:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        
        displayNode.SetColor(color[0], color[1], color[2])
        displayNode.SetOpacity(0.8)  # More opaque for better visibility in 2D
        displayNode.SetBackfaceCulling(False)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)  # Show in slice views
        displayNode.SetSliceIntersectionThickness(3)  # Make lines thicker in slice views
        displayNode.SetSliceIntersectionOpacity(0.8)  # Ensure opacity in slice views
        
        slicer.app.processEvents()
        
        return modelNode


class GammaTileModule(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "GammaView Dosimetry"
        self.parent.contributors = ["Jacob Lee, UMN Brachytherapy Research Lab", "Clara Ferreira, UMN Dept of Radio Oncology"]
        self.parent.categories = ["Surgery"]
        parent.helpText = (
        "This module assists with intraoperative GammaTile placement and dosimetry visualization.\n\n"
        "Usage:\n"
        "1. Load an MRI volume and click 'Create Brain Surface' to generate the surface model.\n"
        "2. Switch to Placement Mode and click on the brain surface (3D view or 2D slice views) to place GammaTiles.\n"
        "3. Switch to Cursor Mode to select placed tiles: click tiles in 2D slice views to select them.\n"
        "4. With a tile selected, use Rotate or Translate (arrow buttons) to adjust positions.\n"
        "5. Click 'Update & Show Isodoses' to compute dose volume and display 3D isodose surfaces.\n"
        "6. Export: use 'Export seed and tile positions (CSV)' for coordinates, or 'Export Dose Volume structure (RTDOSE)' for DICOM RTDOSE (units in Gy) or NRRD.\n\n"
        "The tile count is shown at the bottom of the module panel.\n\n"
        "GammaView - Intraoperative GammaTile Dosimetry Visualization")
        parent.acknowledgementText = (
            "This module was developed as part of the IntraOp GammaTile Research project at the University of Minnesota, "
            "with input from the radiation oncology and neurosurgery teams."
        )


class GammaTileWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = GammaTileDosimetry()
        self._volume_array_cache = {}  # volumeNode ID -> ((image MTime, node MTime), (numpy array, dims, rasToIjk))
        self.placing = False
        self.placing_seeds = False
        self.brain_surface_model = None
        self.sliceViewObserverTags = {'Red': None, 'Green': None, 'Yellow': None}  # Track slice view observers
        self.selected_tile_index = None  # Index of currently selected tile
        self.selected_seed_index = None  # Index of currently selected standalone seed
        self.selected_from_slice_view = None  # Which slice view the tile/seed was selected from
        self.is_dragging = False  # Track if we're currently dragging a tile
        self.drag_start_pos = None  # Starting position for drag operation
        self.drag_start_tile_center = None  # Tile center when drag started
        self.is_dragging_seed = False  # Track if we're currently dragging a seed
        self.drag_start_seed_center = None  # Seed center when drag started
        
        # Input Data Section
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Input Data"
        self.layout.addWidget(inputCollapsible)
        inputFormLayout = qt.QFormLayout(inputCollapsible)
        
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.currentNodeChanged.connect(self.onVolumeChanged)
        inputFormLayout.addRow("MRI Volume:", self.volumeSelector)
        
        self.createSurfaceButton = qt.QPushButton("Create Brain Surface")
        self.createSurfaceButton.clicked.connect(self.onCreateBrainSurface)
        inputFormLayout.addWidget(self.createSurfaceButton)
        
        # Dosimetry Section
        dosCollapsible = ctk.ctkCollapsibleButton()
        dosCollapsible.text = "Dosimetry"
        self.layout.addWidget(dosCollapsible)
        dosFormLayout = qt.QFormLayout(dosCollapsible)
        
        self.prescriptionSpinBox = qt.QDoubleSpinBox()
        self.prescriptionSpinBox.value = 60
        self.prescriptionSpinBox.suffix = " Gy"
        dosFormLayout.addRow("Prescription:", self.prescriptionSpinBox)
        
        # Mode selector
        modeGroupBox = qt.QGroupBox("Interaction Mode")
        modeLayout = qt.QVBoxLayout()
        modeGroupBox.setLayout(modeLayout)
        
        self.modeComboBox = qt.QComboBox()
        self.modeComboBox.addItem("Cursor Mode")
        self.modeComboBox.addItem("Placement Mode")
        self.modeComboBox.addItem("Seeds Mode")
        self.modeComboBox.currentIndexChanged.connect(self.onModeChanged)
        modeLayout.addWidget(self.modeComboBox)
        
        dosFormLayout.addRow(modeGroupBox)
        
        self.undoButton = qt.QPushButton("Undo Last Tile")
        self.undoButton.clicked.connect(self.onUndo)
        dosFormLayout.addWidget(self.undoButton)
        self.undoSeedButton = qt.QPushButton("Undo Last Seed")
        self.undoSeedButton.clicked.connect(self.onUndoSeed)
        dosFormLayout.addWidget(self.undoSeedButton)
        
        selectionGroupBox = qt.QGroupBox("Tile / Seed Selection")
        selectionLayout = qt.QVBoxLayout()
        selectionGroupBox.setLayout(selectionLayout)
        
        self.selectionLabel = qt.QLabel("No tile selected")
        self.selectionLabel.setStyleSheet("font-weight: bold; color: #666666;")
        selectionLayout.addWidget(self.selectionLabel)
        
        # Rotation buttons
        rotationLayout = qt.QHBoxLayout()
        self.rotateClockwiseButton = qt.QPushButton("Rotate Clockwise")
        self.rotateClockwiseButton.clicked.connect(lambda: self.onRotateTile(clockwise=True))
        rotationLayout.addWidget(self.rotateClockwiseButton)
        
        self.rotateCounterClockwiseButton = qt.QPushButton("Rotate Counterclockwise")
        self.rotateCounterClockwiseButton.clicked.connect(lambda: self.onRotateTile(clockwise=False))
        rotationLayout.addWidget(self.rotateCounterClockwiseButton)
        selectionLayout.addLayout(rotationLayout)
        
        # Translation buttons
        translationLayout = qt.QHBoxLayout()
        
        self.translateLeftButton = qt.QPushButton("←")
        self.translateLeftButton.setMaximumWidth(50)
        self.translateLeftButton.clicked.connect(lambda: self.onTranslateTile(direction='left'))
        translationLayout.addWidget(self.translateLeftButton)
        
        self.translateRightButton = qt.QPushButton("→")
        self.translateRightButton.setMaximumWidth(50)
        self.translateRightButton.clicked.connect(lambda: self.onTranslateTile(direction='right'))
        translationLayout.addWidget(self.translateRightButton)
        
        self.translateUpButton = qt.QPushButton("↑")
        self.translateUpButton.setMaximumWidth(50)
        self.translateUpButton.clicked.connect(lambda: self.onTranslateTile(direction='up'))
        translationLayout.addWidget(self.translateUpButton)
        
        self.translateDownButton = qt.QPushButton("↓")
        self.translateDownButton.setMaximumWidth(50)
        self.translateDownButton.clicked.connect(lambda: self.onTranslateTile(direction='down'))
        translationLayout.addWidget(self.translateDownButton)
        selectionLayout.addLayout(translationLayout)
        
        dosFormLayout.addRow(selectionGroupBox)
        
        # Visualization Section
        visCollapsible = ctk.ctkCollapsibleButton()
        visCollapsible.text = "Visualization"
        self.layout.addWidget(visCollapsible)
        visFormLayout = qt.QFormLayout(visCollapsible)
        
        self.updateButton = qt.QPushButton("Update & Show Isodoses")
        self.updateButton.clicked.connect(self.onUpdate)
        visFormLayout.addWidget(self.updateButton)
        
        # Dose color legend
        legendLabel = qt.QLabel("Isodose Levels:")
        visFormLayout.addWidget(legendLabel)
        
        self.legendWidget = qt.QWidget()
        legendLayout = qt.QVBoxLayout()
        legendLayout.setContentsMargins(5, 5, 5, 5)
        self.legendWidget.setLayout(legendLayout)
        
        self.legendItems = []
        visFormLayout.addWidget(self.legendWidget)
        
        self.statusLabel = qt.QLabel("Ready")
        visFormLayout.addWidget(self.statusLabel)
        
        # Export Section
        exportCollapsible = ctk.ctkCollapsibleButton()
        exportCollapsible.text = "Export"
        self.layout.addWidget(exportCollapsible)
        exportFormLayout = qt.QFormLayout(exportCollapsible)
        
        self.exportPositionsButton = qt.QPushButton("Export seed and tile positions (CSV)")
        self.exportPositionsButton.clicked.connect(self.onExportPositions)
        exportFormLayout.addWidget(self.exportPositionsButton)
        
        self.exportDoseVolumeButton = qt.QPushButton("Export Dose Volume structure (RTDOSE)")
        self.exportDoseVolumeButton.clicked.connect(self.onExportDoseVolume)
        exportFormLayout.addWidget(self.exportDoseVolumeButton)

        # Tile counter (bottom of panel) - show both tiles and seeds
        tileCounterFrame = qt.QFrame()
        tileCounterLayout = qt.QHBoxLayout(tileCounterFrame)
        tileCounterLayout.setContentsMargins(8, 8, 8, 8)
        self.tileCounterLabel = qt.QLabel("Tiles: 0  |  Seeds: 0")
        self.tileCounterLabel.setStyleSheet("font-weight: bold; font-size: 14px;")
        tileCounterLayout.addWidget(self.tileCounterLabel)
        tileCounterLayout.addStretch()
        self.layout.addWidget(tileCounterFrame)
        
        self.layout.addStretch(1)
        
        # Set up 3D view click handler
        self.interactorObserverTag = None
        
        # Initialize to cursor mode
        self.modeComboBox.setCurrentIndex(0)
    
    def installSliceViewObservers(self):
        self.removeSliceViewObservers()
        
        try:
            layoutManager = slicer.app.layoutManager()
            if not layoutManager:
                return
            
            for sliceViewName in ['Red', 'Green', 'Yellow']:
                try:
                    sliceWidget = layoutManager.sliceWidget(sliceViewName)
                    if not sliceWidget:
                        continue

                    sliceView = sliceWidget.sliceView()
                    if not sliceView:
                        continue

                    interactor = sliceView.interactor()
                    if not interactor:
                        continue
                    
                    # Create closures to capture sliceViewName and interactor
                    def makeClickHandler(name, inter):
                        def pressHandler(caller, event):
                            click_pos = inter.GetEventPosition()
                            
                            # Check mode
                            current_idx = self.modeComboBox.currentIndex
                            
                            # In cursor mode (index 0): try tile selection and start drag
                            if current_idx == 0:
                                try:
                                    result = self.onSliceViewTileSelection(name, click_pos[0], click_pos[1])
                                    # If tile was selected, start drag operation
                                    if result and self.selected_tile_index is not None:
                                        self.is_dragging = True
                                        self.drag_start_pos = np.array(click_pos)
                                        tile = self.logic.tiles[self.selected_tile_index]
                                        self.drag_start_tile_center = tile.center.copy()
                                        return
                                    # Try seed selection if no tile selected; start seed drag if seed selected
                                    if not result:
                                        self.onSliceViewSeedSelection(name, click_pos[0], click_pos[1])
                                        if self.selected_seed_index is not None and self.selected_seed_index < len(self.logic.seeds):
                                            self.is_dragging_seed = True
                                            self.drag_start_pos = np.array(click_pos)
                                            seed = self.logic.seeds[self.selected_seed_index]
                                            self.drag_start_seed_center = seed.center.copy()
                                except Exception as e:
                                    logger.error(f"Error in tile selection: {e}")

                            # In placement mode (index 1): handle tile placement
                            if current_idx == 1 and self.placing and not self.placing_seeds:
                                self.onSliceViewClick(name, click_pos[0], click_pos[1])

                            # In seeds mode (index 2): handle seed placement
                            if current_idx == 2 and self.placing and self.placing_seeds:
                                self.onSliceViewSeedClick(name, click_pos[0], click_pos[1])
                        
                        def moveHandler(caller, event):
                            # Only handle drag in cursor mode
                            current_idx = self.modeComboBox.currentIndex
                            if current_idx == 0 and self.is_dragging and self.selected_tile_index is not None:
                                current_pos = inter.GetEventPosition()
                                self.onTileDrag(name, self.drag_start_pos, current_pos)
                            elif current_idx == 0 and self.is_dragging_seed and self.selected_seed_index is not None:
                                current_pos = inter.GetEventPosition()
                                self.onSeedDrag(name, self.drag_start_pos, current_pos)
                        
                        def releaseHandler(caller, event):
                            if self.is_dragging:
                                self.is_dragging = False
                                self.drag_start_pos = None
                                self.drag_start_tile_center = None
                            if self.is_dragging_seed:
                                self.is_dragging_seed = False
                                self.drag_start_pos = None
                                self.drag_start_seed_center = None
                        
                        return pressHandler, moveHandler, releaseHandler
                    
                    # Create handlers
                    pressHandler, moveHandler, releaseHandler = makeClickHandler(sliceViewName, interactor)
                    
                    tag_press = interactor.AddObserver(
                        vtk.vtkCommand.LeftButtonPressEvent,
                        pressHandler,
                        1.0
                    )
                    tag_move = interactor.AddObserver(
                        vtk.vtkCommand.MouseMoveEvent,
                        moveHandler,
                        1.0
                    )
                    tag_release = interactor.AddObserver(
                        vtk.vtkCommand.LeftButtonReleaseEvent,
                        releaseHandler,
                        1.0
                    )
                    self.sliceViewObserverTags[sliceViewName] = (tag_press, tag_move, tag_release)
                except Exception as e:
                    logger.warning(f"Error installing observer for {sliceViewName} slice view: {e}")
        except Exception as e:
            logger.warning(f"Error installing slice view observers: {e}")
    
    def removeSliceViewObservers(self):
        try:
            layoutManager = slicer.app.layoutManager()
            if not layoutManager:
                return
            
            for sliceViewName in ['Red', 'Green', 'Yellow']:
                tags = self.sliceViewObserverTags.get(sliceViewName)
                if tags is not None:
                    try:
                        sliceWidget = layoutManager.sliceWidget(sliceViewName)
                        if sliceWidget:
                            sliceView = sliceWidget.sliceView()
                            if sliceView:
                                interactor = sliceView.interactor()
                                if interactor:
                                    # Handle both old format (single tag) and new format (tuple of tags)
                                    if isinstance(tags, tuple):
                                        for tag in tags:
                                            if tag is not None:
                                                interactor.RemoveObserver(tag)
                                    else:
                                        interactor.RemoveObserver(tags)
                    except Exception as e:
                        logger.warning(f"Error removing observer for {sliceViewName}: {e}")
                    finally:
                        self.sliceViewObserverTags[sliceViewName] = None
        except Exception as e:
            logger.warning(f"Error removing slice view observers: {e}")
    
    def convertSliceClickToRay(self, sliceViewName, x, y):
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return None, None
        
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        if not sliceWidget:
            return None, None

        sliceLogic = sliceWidget.sliceLogic()
        if not sliceLogic:
            return None, None

        sliceNode = sliceLogic.GetSliceNode()
        if not sliceNode:
            return None, None

        sliceView = sliceWidget.sliceView()
        if not sliceView:
            return None, None
        
        xyToRas = sliceNode.GetXYToRAS()

        if not xyToRas:
            sliceOffset = sliceNode.GetSliceOffset()
            normal = sliceNode.GetNormal()

            sliceView = sliceWidget.sliceView()
            if sliceView:
                viewSize = sliceView.size
                viewWidth = viewSize.width()
                viewHeight = viewSize.height()

                sliceBounds = sliceNode.GetXYBounds()
                if sliceBounds:
                    xMin, xMax, yMin, yMax = sliceBounds
                    x_mm = xMin + (x / viewWidth) * (xMax - xMin)
                    y_mm = yMin + ((viewHeight - y) / viewHeight) * (yMax - yMin)

                    ray_origin_ras = np.array(sliceNode.GetXYZOrigin(), dtype=np.float32)
                    ray_direction_ras = np.array(normal, dtype=np.float32)
                    return ray_origin_ras, ray_direction_ras

            return None, None
        slice_coords = [x, y, 0, 1.0]
        ras_coords = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_coords, ras_coords)
        
        ray_origin_ras = np.array([ras_coords[0], ras_coords[1], ras_coords[2]], dtype=np.float32)

        slice_normal = [0.0, 0.0, 1.0, 0.0]
        ras_normal = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_normal, ras_normal)
        
        ray_direction_ras = np.array([ras_normal[0], ras_normal[1], ras_normal[2]], dtype=np.float32)
        norm = np.linalg.norm(ray_direction_ras)
        if norm > 1e-6:
            ray_direction_ras = ray_direction_ras / norm
        else:
            normal = sliceNode.GetNormal()
            ray_direction_ras = np.array(normal, dtype=np.float32)
        
        return ray_origin_ras, ray_direction_ras
    
    def cleanup(self):
        self.removeClickObserver()
        self.removeSliceViewObservers()
        super().cleanup()
    
    def onVolumeChanged(self):
        vol = self.volumeSelector.currentNode()
        if vol:
            self.statusLabel.setText(f"Volume selected: {vol.GetName()}")
    
    def onCreateBrainSurface(self):
        vol = self.volumeSelector.currentNode()
        if not vol:
            slicer.util.errorDisplay("Select MRI volume first")
            return
        
        try:
            self.statusLabel.setText("Creating brain surface...")
            slicer.app.processEvents()
            
            surface = BrainSurfaceGenerator.create_brain_surface(vol)
            if surface is None or surface.GetNumberOfPoints() == 0:
                self.statusLabel.setText("Surface creation failed. Try adjusting threshold.")
                return
            
            # Remove old surface if exists
            if self.brain_surface_model:
                slicer.mrmlScene.RemoveNode(self.brain_surface_model)
            
            self.brain_surface_model = BrainSurfaceGenerator.add_surface_to_scene(surface, vol)
            self.logic.brain_surface_model = self.brain_surface_model
            
            self.statusLabel.setText("Brain surface created successfully!")
        except Exception as e:
            error_msg = f"Error creating brain surface: {str(e)}"
            self.statusLabel.setText(error_msg)
            logger.error(error_msg, exc_info=True)
    
    def onModeChanged(self, index):
        if index == 0:  # Cursor Mode
            self.setPlacementMode(False, seeds_mode=False)
        elif index == 1:  # Placement Mode (tiles)
            vol = self.volumeSelector.currentNode()
            if not vol:
                slicer.util.errorDisplay("Select MRI volume first")
                self.modeComboBox.blockSignals(True)
                self.modeComboBox.setCurrentIndex(0)
                self.modeComboBox.blockSignals(False)
                return
            self.setPlacementMode(True, seeds_mode=False)
        elif index == 2:  # Seeds Mode
            vol = self.volumeSelector.currentNode()
            if not vol:
                slicer.util.errorDisplay("Select MRI volume first")
                self.modeComboBox.blockSignals(True)
                self.modeComboBox.setCurrentIndex(0)
                self.modeComboBox.blockSignals(False)
                return
            self.setPlacementMode(True, seeds_mode=True)

    def setPlacementMode(self, enable, seeds_mode=False):
        self.placing = bool(enable)
        self.placing_seeds = bool(seeds_mode)
        if enable:
            self.logic.prescription_gy = self.prescriptionSpinBox.value
            if seeds_mode:
                self.statusLabel.setText("SEEDS MODE: Click on brain surface in 3D or 2D slice views to place a seed.")
            else:
                self.statusLabel.setText("PLACEMENT MODE: Click on brain surface in 3D view or 2D slice views to place a tile.")
            self.installClickObserver()
            self.installSliceViewObservers()
        else:
            self.placing_seeds = False
            self.statusLabel.setText("CURSOR MODE: Click on tiles or seeds in 2D views to select them.")
            self.removeClickObserver()
            if not any(self.sliceViewObserverTags.values()):
                self.installSliceViewObservers()
    
    def installClickObserver(self):
        self.removeClickObserver()
        
        try:
            threeDWidget = slicer.app.layoutManager().threeDWidget(0)
            if not threeDWidget:
                return

            threeDView = threeDWidget.threeDView()
            if not threeDView:
                return

            interactorStyle = threeDView.interactorStyle()
            if not interactorStyle:
                return

            interactor = interactorStyle.GetInteractor()
            if not interactor:
                return
            
            self.interactorObserverTag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                self.onMouseClick,
                1.0
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Error setting up placement mode: {str(e)}")
    
    def removeClickObserver(self):
        if self.interactorObserverTag is None:
            return
        try:
            threeDWidget = slicer.app.layoutManager().threeDWidget(0)
            if threeDWidget:
                threeDView = threeDWidget.threeDView()
                if threeDView:
                    interactorStyle = threeDView.interactorStyle()
                    if interactorStyle:
                        interactor = interactorStyle.GetInteractor()
                        if interactor:
                            try:
                                interactor.RemoveObserver(self.interactorObserverTag)
                            except (RuntimeError, TypeError) as e:
                                logger.debug("RemoveObserver (style): %s", e)
                    renderWindow = threeDView.renderWindow()
                    if renderWindow:
                        rwi = renderWindow.GetInteractor()
                        if rwi:
                            try:
                                rwi.RemoveObserver(self.interactorObserverTag)
                            except (RuntimeError, TypeError) as e:
                                logger.debug("RemoveObserver (render window): %s", e)
        except (AttributeError, RuntimeError) as e:
            logger.warning("Error removing click observer: %s", e)
        finally:
            self.interactorObserverTag = None
    
    def onMouseClick(self, caller, event):
        if not getattr(self, 'placing', False) or self.interactorObserverTag is None:
            return
        try:
            mode_combo = getattr(self, 'modeComboBox', None)
            if mode_combo is None or mode_combo.currentIndex not in (1, 2):
                return
        except (AttributeError, RuntimeError):
            return
        try:
            interactor = caller
            if not hasattr(interactor, 'GetEventPosition'):
                threeDWidget = slicer.app.layoutManager().threeDWidget(0)
                if threeDWidget:
                    threeDView = threeDWidget.threeDView()
                    if threeDView:
                        interactorStyle = threeDView.interactorStyle()
                        if interactorStyle:
                            interactor = interactorStyle.GetInteractor()
            
            click_x = interactor.GetEventPosition()[0]
            click_y = interactor.GetEventPosition()[1]

            vol = self.volumeSelector.currentNode()
            if not vol:
                self.statusLabel.setText("No volume selected.")
                self.switchToCursorMode()
                return
            
            threeDWidget = slicer.app.layoutManager().threeDWidget(0)
            if not threeDWidget:
                self.statusLabel.setText("3D view not available.")
                self.switchToCursorMode()
                return
            
            threeDView = threeDWidget.threeDView()
            renderWindow = threeDView.renderWindow()
            if not renderWindow:
                self.statusLabel.setText("Could not get render window")
                self.switchToCursorMode()
                return
            
            renderers = renderWindow.GetRenderers()
            if not renderers or renderers.GetNumberOfItems() == 0:
                self.statusLabel.setText("No renderers available")
                self.switchToCursorMode()
                return
        
            ren = renderers.GetFirstRenderer()
            if not ren:
                renderers.InitTraversal()
                ren = renderers.GetNextItem()
            if not ren:
                self.statusLabel.setText("Could not get renderer")
                self.switchToCursorMode()
                return
            
            picker = vtk.vtkPropPicker()
            picked_pos = None
            picked_actor = None
            if picker.Pick(click_x, click_y, 0, ren):
                picked_pos = picker.GetPickPosition()
                picked_actor = picker.GetActor()
            if picked_pos is None:
                imageData = vol.GetImageData()
                if imageData:
                    bounds = imageData.GetBounds()
                    picked_pos = np.array([
                        (bounds[0] + bounds[1]) / 2,
                        (bounds[2] + bounds[3]) / 2,
                        (bounds[4] + bounds[5]) / 2
                    ])
            if picked_pos is None:
                self.statusLabel.setText("Could not determine click position. Try clicking on the volume or surface.")
                logger.warning("Picking failed (no geometry hit, no volume bounds)")
                self.switchToCursorMode()
                return
            
            picked_pos = np.array(picked_pos)

            imageData = vol.GetImageData()
            if not imageData:
                self.switchToCursorMode()
                return
            
            spacing = imageData.GetSpacing()
            spacing_mm = np.mean(spacing)  # Average spacing
            
            click_pos_ras = np.array(picked_pos)

            normal = None
            if picked_actor and self.brain_surface_model:
                normal = self.getNormalFromModel(self.brain_surface_model, click_pos_ras)

            if normal is None:
                normal = self.computeSurfaceNormal(vol, click_pos_ras)
            
            if normal is None:
                normal = np.array([0, 0, 1], dtype=np.float32)
                self.statusLabel.setText("Using default normal. Click on surface for better orientation.")
            
            settled_pos = self.settleToSurface(vol, click_pos_ras, normal)

            adjusted_pos, penetration_depth = self.checkAndLimitPenetration(vol, settled_pos, normal)
            if penetration_depth > GammaTilePlacement.MAX_PENETRATION_MM:
                self.statusLabel.setText(
                    f"Warning: Tile penetration limited to {GammaTilePlacement.MAX_PENETRATION_MM}mm "
                    f"(was {penetration_depth:.1f}mm). Tile adjusted."
                )

            if self.placing_seeds:
                # Place standalone seed: offset center so one end of seed is on surface (stick like tiles)
                seed_half_len = SingleSeedPlacement.SEED_LENGTH_MM / 2.0
                seed_center = adjusted_pos - seed_half_len * normal
                self.logic.place_seed(seed_center, normal)
                try:
                    seedModel = self.createSeedModel(seed_center, normal)
                    if seedModel:
                        self.logic.seed_fiducial_nodes.append(seedModel)
                        self.updateTileCounter()
                        threeDView.scheduleRender()
                        self.statusLabel.setText(f"Seed {len(self.logic.seeds)} placed. Click to place another seed.")
                    else:
                        self.logic.seeds.pop()
                        self.statusLabel.setText("Error: Seed model creation failed.")
                except Exception as e:
                    if self.logic.seeds:
                        self.logic.seeds.pop()
                    self.statusLabel.setText(f"Error: {str(e)}")
            else:
                final_pos, _ = self.logic.place_tile(adjusted_pos, normal, spacing_mm)
                try:
                    tileModel = self.createTileModel(final_pos, normal, spacing_mm)
                    if tileModel:
                        self.logic.fiducial_nodes.append(tileModel)
                        self.updateTileCounter()
                        threeDView.scheduleRender()
                        self.statusLabel.setText(f"Tile {len(self.logic.tiles)} placed. Click to place another tile.")
                    else:
                        self.statusLabel.setText("Error: Tile model creation failed.")
                except Exception as e:
                    self.statusLabel.setText(f"Error: {str(e)}")
            
        except Exception as e:
            self.statusLabel.setText(f"Error: {str(e)}")
    
    def switchToCursorMode(self):
        self.modeComboBox.blockSignals(True)
        self.modeComboBox.setCurrentIndex(0)  # Cursor Mode
        self.modeComboBox.blockSignals(False)
    
    def onSliceViewTileSelection(self, sliceViewName, x, y):
        if not self.logic.tiles or not self.logic.fiducial_nodes:
            return False

        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        if not sliceWidget:
            return False
        
        sliceNode = sliceWidget.sliceLogic().GetSliceNode()
        if not sliceNode:
            return False
        
        xyToRas = sliceNode.GetXYToRAS()
        if not xyToRas:
            return False
        
        rasToSlice = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xyToRas, rasToSlice)

        ray_origin, ray_direction = self.convertSliceClickToRay(sliceViewName, x, y)
        if ray_origin is None:
            return False
        
        click_pos_ras = np.array(ray_origin)

        click_pos_homog = [click_pos_ras[0], click_pos_ras[1], click_pos_ras[2], 1.0]
        click_slice_coords = [0.0, 0.0, 0.0, 0.0]
        rasToSlice.MultiplyPoint(click_pos_homog, click_slice_coords)
        click_slice_pos = np.array([click_slice_coords[0], click_slice_coords[1], click_slice_coords[2]])

        matching_tiles = []
        
        for idx, tile_node in enumerate(self.logic.fiducial_nodes):
            if idx >= len(self.logic.tiles):
                continue
            
            if not tile_node:
                continue
            
            tile_center = self.logic.tiles[idx].center

            tile_center_homog = [tile_center[0], tile_center[1], tile_center[2], 1.0]
            tile_slice_coords = [0.0, 0.0, 0.0, 0.0]
            rasToSlice.MultiplyPoint(tile_center_homog, tile_slice_coords)
            tile_slice_pos = np.array([tile_slice_coords[0], tile_slice_coords[1], tile_slice_coords[2]])

            dx = tile_slice_pos[0] - click_slice_pos[0]
            dy = tile_slice_pos[1] - click_slice_pos[1]
            distance_2d = np.sqrt(dx*dx + dy*dy)

            if distance_2d < 15.0:
                matching_tiles.append((idx, distance_2d))
        
        if not matching_tiles:
            return False

        matching_tiles.sort(key=lambda x: x[0], reverse=True)
        selected_idx, selected_distance = matching_tiles[0]

        self.selected_tile_index = selected_idx
        self.selected_seed_index = None
        self.selected_from_slice_view = sliceViewName
        self.updateSelectionDisplay()
        return True

    def onSliceViewSeedSelection(self, sliceViewName, x, y):
        if not self.logic.seeds or not self.logic.seed_fiducial_nodes:
            return
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        if not sliceWidget:
            return
        sliceNode = sliceWidget.sliceLogic().GetSliceNode()
        if not sliceNode:
            return
        rasToSlice = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(sliceNode.GetXYToRAS(), rasToSlice)
        ray_origin, ray_direction = self.convertSliceClickToRay(sliceViewName, x, y)
        if ray_origin is None:
            return
        click_pos_ras = np.array(ray_origin)
        click_pos_homog = [click_pos_ras[0], click_pos_ras[1], click_pos_ras[2], 1.0]
        click_slice_coords = [0.0, 0.0, 0.0, 0.0]
        rasToSlice.MultiplyPoint(click_pos_homog, click_slice_coords)
        click_slice_pos = np.array([click_slice_coords[0], click_slice_coords[1], click_slice_coords[2]])
        matching_seeds = []
        for idx, seed in enumerate(self.logic.seeds):
            if idx >= len(self.logic.seed_fiducial_nodes):
                continue
            center = seed.center
            center_homog = [center[0], center[1], center[2], 1.0]
            slice_coords = [0.0, 0.0, 0.0, 0.0]
            rasToSlice.MultiplyPoint(center_homog, slice_coords)
            dx = slice_coords[0] - click_slice_pos[0]
            dy = slice_coords[1] - click_slice_pos[1]
            distance_2d = np.sqrt(dx*dx + dy*dy)
            if distance_2d < 8.0:
                matching_seeds.append((idx, distance_2d))
        if not matching_seeds:
            return
        matching_seeds.sort(key=lambda t: t[0], reverse=True)
        selected_idx, _ = matching_seeds[0]
        self.selected_seed_index = selected_idx
        self.selected_tile_index = None
        self.selected_from_slice_view = sliceViewName
        self.updateSelectionDisplay()
    
    def updateSelectionDisplay(self):
        if self.selected_tile_index is not None and self.selected_tile_index < len(self.logic.tiles):
            selection_text = f"Selected: Tile {self.selected_tile_index + 1} (from {self.selected_from_slice_view} view)"
            self.selectionLabel.setText(selection_text)
            self.selectionLabel.setStyleSheet("font-weight: bold; color: #0066cc;")
            self.statusLabel.setText(selection_text)
            self.selected_seed_index = None
        elif self.selected_seed_index is not None and self.selected_seed_index < len(self.logic.seeds):
            selection_text = f"Selected: Seed {self.selected_seed_index + 1} (from {self.selected_from_slice_view} view)"
            self.selectionLabel.setText(selection_text)
            self.selectionLabel.setStyleSheet("font-weight: bold; color: #0066cc;")
            self.statusLabel.setText(selection_text)
            self.selected_tile_index = None
        else:
            self.selected_tile_index = None
            self.selected_seed_index = None
            self.selected_from_slice_view = None
            self.selectionLabel.setText("No tile or seed selected")
            self.selectionLabel.setStyleSheet("font-weight: bold; color: #666666;")
            self.statusLabel.setText("Ready")
        self.updateTileVisualAppearance()
        self.updateTileCounter()
    
    def updateTileCounter(self):
        tiles_count = len(self.logic.tiles)
        seeds_count = len(self.logic.seeds)
        self.tileCounterLabel.setText(f"Tiles: {tiles_count}  |  Seeds: {seeds_count}")
    
    def updateTileVisualAppearance(self):
        for idx, tile_node in enumerate(self.logic.fiducial_nodes):
            if idx >= len(self.logic.tiles) or not tile_node:
                continue
            displayNode = tile_node.GetDisplayNode()
            if not displayNode:
                continue
            if idx == self.selected_tile_index:
                displayNode.SetColor(1.0, 0.84, 0.0)
                displayNode.SetAmbient(1.0)
                displayNode.SetDiffuse(1.0)
                displayNode.SetSpecular(1.0)
                displayNode.SetOpacity(1.0)
                displayNode.SetPower(50.0)
                displayNode.SetSliceIntersectionThickness(3)
            else:
                displayNode.SetColor(1.0, 0.8, 0.0)
                displayNode.SetAmbient(0.5)
                displayNode.SetDiffuse(0.8)
                displayNode.SetSpecular(0.3)
                displayNode.SetOpacity(1.0)
                displayNode.SetSliceIntersectionThickness(1)
            displayNode.Modified()
        for idx, seed_node in enumerate(self.logic.seed_fiducial_nodes):
            if idx >= len(self.logic.seeds) or not seed_node:
                continue
            displayNode = seed_node.GetDisplayNode()
            if not displayNode:
                continue
            if idx == self.selected_seed_index:
                displayNode.SetColor(0.0, 1.0, 0.5)  # Bright green when selected
                displayNode.SetSliceIntersectionThickness(3)
            else:
                displayNode.SetColor(0.2, 0.8, 0.2)
                displayNode.SetSliceIntersectionThickness(2)
            displayNode.Modified()
        if not self.logic.fiducial_nodes and not self.logic.seed_fiducial_nodes:
            return
        try:
            layoutManager = slicer.app.layoutManager()
            for viewName in ['Red', 'Green', 'Yellow']:
                sliceWidget = layoutManager.sliceWidget(viewName)
                if sliceWidget:
                    sliceView = sliceWidget.sliceView()
                    if sliceView:
                        sliceView.scheduleRender()
            
            threeDWidget = layoutManager.threeDWidget(0)
            if threeDWidget:
                threeDView = threeDWidget.threeDView()
                if threeDView:
                    threeDView.scheduleRender()
        except:
            pass
    
    def onTileDrag(self, sliceViewName, start_pos, current_pos):
        if self.selected_tile_index is None or self.selected_tile_index >= len(self.logic.tiles):
            return

        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        if not sliceWidget:
            return
        
        sliceNode = sliceWidget.sliceLogic().GetSliceNode()
        if not sliceNode:
            return
        
        xyToRas = sliceNode.GetXYToRAS()
        if not xyToRas:
            return
        
        start_ray, _ = self.convertSliceClickToRay(sliceViewName, start_pos[0], start_pos[1])
        current_ray, _ = self.convertSliceClickToRay(sliceViewName, current_pos[0], current_pos[1])

        if start_ray is None or current_ray is None:
            return

        movement_delta = np.array(current_ray) - np.array(start_ray)

        slice_normal_slice = [0.0, 0.0, 1.0, 0.0]
        slice_normal_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_normal_slice, slice_normal_ras)
        slice_normal = np.array([slice_normal_ras[0], slice_normal_ras[1], slice_normal_ras[2]])
        slice_normal = slice_normal / np.linalg.norm(slice_normal)

        perpendicular_component = np.dot(movement_delta, slice_normal) * slice_normal
        movement_in_plane = movement_delta - perpendicular_component

        tile = self.logic.tiles[self.selected_tile_index]
        new_center = self.drag_start_tile_center + movement_in_plane
        tile.center = new_center

        tile.seed_positions_ras = []
        for seed_local in tile.seed_positions_local:
            seed_ras = tile.center + tile.frame @ seed_local
            tile.seed_positions_ras.append(seed_ras)

        self.updateTileModelPosition(self.selected_tile_index)

        self.drag_start_pos = np.array(current_pos)
        self.drag_start_tile_center = tile.center.copy()

    def onSeedDrag(self, sliceViewName, start_pos, current_pos):
        if self.selected_seed_index is None or self.selected_seed_index >= len(self.logic.seeds):
            return
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        if not sliceWidget:
            return
        sliceNode = sliceWidget.sliceLogic().GetSliceNode()
        if not sliceNode:
            return
        xyToRas = sliceNode.GetXYToRAS()
        if not xyToRas:
            return
        start_ray, _ = self.convertSliceClickToRay(sliceViewName, start_pos[0], start_pos[1])
        current_ray, _ = self.convertSliceClickToRay(sliceViewName, current_pos[0], current_pos[1])
        if start_ray is None or current_ray is None:
            return
        movement_delta = np.array(current_ray) - np.array(start_ray)
        slice_normal_slice = [0.0, 0.0, 1.0, 0.0]
        slice_normal_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_normal_slice, slice_normal_ras)
        slice_normal = np.array([slice_normal_ras[0], slice_normal_ras[1], slice_normal_ras[2]])
        slice_normal = slice_normal / np.linalg.norm(slice_normal)
        perpendicular_component = np.dot(movement_delta, slice_normal) * slice_normal
        movement_in_plane = movement_delta - perpendicular_component
        seed = self.logic.seeds[self.selected_seed_index]
        new_center = self.drag_start_seed_center + movement_in_plane
        seed.center = new_center
        self.updateSeedModelPosition(self.selected_seed_index)
        self.drag_start_pos = np.array(current_pos)
        self.drag_start_seed_center = seed.center.copy()

    def updateTileModelPosition(self, tile_index):
        if tile_index < 0 or tile_index >= len(self.logic.tiles) or tile_index >= len(self.logic.fiducial_nodes):
            return
        tile = self.logic.tiles[tile_index]
        tile_node = self.logic.fiducial_nodes[tile_index]
        if not tile_node:
            return
        polydata = tile_node.GetPolyData()
        if not polydata:
            return

        current_points = polydata.GetPoints()
        if current_points and current_points.GetNumberOfPoints() > 0:
            current_center = numpy_support.vtk_to_numpy(current_points.GetData()).mean(axis=0)

            translation = tile.center - current_center

            transform = vtk.vtkTransform()
            transform.Translate(translation[0], translation[1], translation[2])

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transform)
            transformFilter.SetInputData(polydata)
            transformFilter.Update()
            
            new_polydata = transformFilter.GetOutput()
            tile_node.SetAndObservePolyData(new_polydata)

            self.updateTileVisualAppearance()

            try:
                layoutManager = slicer.app.layoutManager()
                for viewName in ['Red', 'Green', 'Yellow']:
                    sliceWidget = layoutManager.sliceWidget(viewName)
                    if sliceWidget:
                        sliceView = sliceWidget.sliceView()
                        if sliceView:
                            sliceView.scheduleRender()
                
                threeDWidget = layoutManager.threeDWidget(0)
                if threeDWidget:
                    threeDView = threeDWidget.threeDView()
                    if threeDView:
                        threeDView.scheduleRender()
            except:
                pass
        else:
            vol = self.volumeSelector.currentNode()
            if not vol:
                return
            
            imageData = vol.GetImageData()
            if not imageData:
                return
            
            spacing = imageData.GetSpacing()
            spacing_mm = np.mean(spacing)

            if tile_node.GetScene():
                slicer.mrmlScene.RemoveNode(tile_node)

            new_model = self.createTileModel(tile.center, tile.normal, spacing_mm)
            if new_model:
                self.logic.fiducial_nodes[tile_index] = new_model
                self.updateTileVisualAppearance()

                try:
                    layoutManager = slicer.app.layoutManager()
                    for viewName in ['Red', 'Green', 'Yellow']:
                        sliceWidget = layoutManager.sliceWidget(viewName)
                        if sliceWidget:
                            sliceView = sliceWidget.sliceView()
                            if sliceView:
                                sliceView.scheduleRender()
                    
                    threeDWidget = layoutManager.threeDWidget(0)
                    if threeDWidget:
                        threeDView = threeDWidget.threeDView()
                        if threeDView:
                            threeDView.scheduleRender()
                except:
                    pass
    
    def updateTileModelRotation(self, tile_index, slice_normal, angle_rad):
        if tile_index < 0 or tile_index >= len(self.logic.tiles) or tile_index >= len(self.logic.fiducial_nodes):
            return
        tile = self.logic.tiles[tile_index]
        tile_node = self.logic.fiducial_nodes[tile_index]
        if not tile_node:
            return
        polydata = tile_node.GetPolyData()
        if not polydata:
            return

        points = polydata.GetPoints()
        if not points or points.GetNumberOfPoints() == 0:
            return
        
        geometric_center = numpy_support.vtk_to_numpy(points.GetData()).mean(axis=0)
        
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(-geometric_center[0], -geometric_center[1], -geometric_center[2])
        angle_deg = np.degrees(angle_rad)
        transform.RotateWXYZ(angle_deg, slice_normal[0], slice_normal[1], slice_normal[2])
        
        transform.Translate(geometric_center[0], geometric_center[1], geometric_center[2])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(polydata)
        transformFilter.Update()
        
        new_polydata = transformFilter.GetOutput()

        new_points = new_polydata.GetPoints()
        new_center = numpy_support.vtk_to_numpy(new_points.GetData()).mean(axis=0)

        drift = tile.center - new_center
        if np.linalg.norm(drift) > 1e-6:
            correction_transform = vtk.vtkTransform()
            correction_transform.Translate(drift[0], drift[1], drift[2])
            correction_filter = vtk.vtkTransformPolyDataFilter()
            correction_filter.SetTransform(correction_transform)
            correction_filter.SetInputData(new_polydata)
            correction_filter.Update()
            new_polydata = correction_filter.GetOutput()
        
        tile_node.SetAndObservePolyData(new_polydata)

        self.updateTileVisualAppearance()

        try:
            layoutManager = slicer.app.layoutManager()
            for viewName in ['Red', 'Green', 'Yellow']:
                sliceWidget = layoutManager.sliceWidget(viewName)
                if sliceWidget:
                    sliceView = sliceWidget.sliceView()
                    if sliceView:
                        sliceView.scheduleRender()

            threeDWidget = layoutManager.threeDWidget(0)
            if threeDWidget:
                threeDView = threeDWidget.threeDView()
                if threeDView:
                    threeDView.scheduleRender()
        except:
            pass

    def onRotateTile(self, clockwise=True):
        if self.selected_seed_index is not None:
            return
        if self.selected_tile_index is None or self.selected_tile_index >= len(self.logic.tiles):
            self.statusLabel.setText("No tile is selected")
            return
        
        if self.selected_from_slice_view is None:
            self.statusLabel.setText("No tile is selected")
            return

        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(self.selected_from_slice_view)
        if not sliceWidget:
            self.statusLabel.setText("Error: Could not access slice view")
            return
        
        sliceNode = sliceWidget.sliceLogic().GetSliceNode()
        if not sliceNode:
            self.statusLabel.setText("Error: Could not access slice node")
            return
        
        xyToRas = sliceNode.GetXYToRAS()
        if not xyToRas:
            self.statusLabel.setText("Error: Could not get slice transform")
            return
        
        slice_normal_slice = [0.0, 0.0, 1.0, 0.0]
        slice_normal_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_normal_slice, slice_normal_ras)
        slice_normal = np.array([slice_normal_ras[0], slice_normal_ras[1], slice_normal_ras[2]])
        slice_normal = slice_normal / np.linalg.norm(slice_normal)

        slice_x_slice = [1.0, 0.0, 0.0, 0.0]
        slice_x_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_x_slice, slice_x_ras)
        slice_x = np.array([slice_x_ras[0], slice_x_ras[1], slice_x_ras[2]])
        slice_x = slice_x / np.linalg.norm(slice_x)
        
        slice_y_slice = [0.0, 1.0, 0.0, 0.0]
        slice_y_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_y_slice, slice_y_ras)
        slice_y = np.array([slice_y_ras[0], slice_y_ras[1], slice_y_ras[2]])
        slice_y = slice_y / np.linalg.norm(slice_y)

        tile = self.logic.tiles[self.selected_tile_index]

        angle_deg = 10.0 if clockwise else -10.0
        angle_rad = np.radians(angle_deg)

        tile_x = tile.frame[:, 0]
        tile_y = tile.frame[:, 1]
        tile_z = tile.frame[:, 2]

        rotation_matrix = self.rotation_matrix_around_axis(slice_normal, angle_rad)

        tile_x_rotated = rotation_matrix @ tile_x
        tile_y_rotated = rotation_matrix @ tile_y

        tile_z_unchanged = tile_z.copy()

        tile_x_rotated = tile_x_rotated / (np.linalg.norm(tile_x_rotated) + 1e-6)
        tile_y_rotated = tile_y_rotated / (np.linalg.norm(tile_y_rotated) + 1e-6)

        new_frame = np.zeros((3, 3))
        new_frame[:, 2] = tile_z_unchanged
        new_frame[:, 0] = tile_x_rotated

        new_frame[:, 1] = np.cross(tile_z_unchanged, tile_x_rotated)
        new_frame[:, 1] = new_frame[:, 1] / (np.linalg.norm(new_frame[:, 1]) + 1e-6)

        tile.frame = new_frame

        tile.normal = tile.frame[:, 2].copy()

        tile.seed_axis_ras = tile.frame @ tile.seed_axis_local
        tile.seed_axis_ras = tile.seed_axis_ras / (np.linalg.norm(tile.seed_axis_ras) + 1e-6)

        tile.seed_positions_ras = []
        for seed_local in tile.seed_positions_local:
            seed_ras = tile.center + tile.frame @ seed_local
            tile.seed_positions_ras.append(seed_ras)

        self.updateTileModelRotation(self.selected_tile_index, slice_normal, angle_rad)
        
        self.statusLabel.setText(f"Tile {self.selected_tile_index + 1} rotated {angle_deg}° {'clockwise' if clockwise else 'counterclockwise'}")
    
    def rotation_matrix_around_axis(self, axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        one_minus_cos = 1 - cos_a

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = cos_a * np.eye(3) + sin_a * K + one_minus_cos * np.outer(axis, axis)
        return R
    
    def onTranslateTile(self, direction):
        """Translate the selected tile or seed by 2mm in the slice plane."""
        if self.selected_from_slice_view is None:
            self.statusLabel.setText("Select a tile or seed in a 2D view first")
            return
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(self.selected_from_slice_view)
        if not sliceWidget or not sliceWidget.sliceLogic().GetSliceNode():
            self.statusLabel.setText("Error: Could not access slice view")
            return
        sliceNode = sliceWidget.sliceLogic().GetSliceNode()
        xyToRas = sliceNode.GetXYToRAS()
        if not xyToRas:
            return
        slice_x_slice = [1.0, 0.0, 0.0, 0.0]
        slice_x_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_x_slice, slice_x_ras)
        slice_x = np.array([slice_x_ras[0], slice_x_ras[1], slice_x_ras[2]])
        slice_x = slice_x / np.linalg.norm(slice_x)
        slice_y_slice = [0.0, 1.0, 0.0, 0.0]
        slice_y_ras = [0.0, 0.0, 0.0, 0.0]
        xyToRas.MultiplyPoint(slice_y_slice, slice_y_ras)
        slice_y = np.array([slice_y_ras[0], slice_y_ras[1], slice_y_ras[2]])
        slice_y = slice_y / np.linalg.norm(slice_y)
        translation_distance = 2.0
        if direction == 'left':
            translation_vector = -slice_x * translation_distance
            direction_name = "left"
        elif direction == 'right':
            translation_vector = slice_x * translation_distance
            direction_name = "right"
        elif direction == 'up':
            translation_vector = slice_y * translation_distance
            direction_name = "up"
        elif direction == 'down':
            translation_vector = -slice_y * translation_distance
            direction_name = "down"
        else:
            return
        if self.selected_seed_index is not None and self.selected_seed_index < len(self.logic.seeds):
            seed = self.logic.seeds[self.selected_seed_index]
            seed.center = seed.center + translation_vector
            self.updateSeedModelPosition(self.selected_seed_index)
            self.statusLabel.setText(f"Seed {self.selected_seed_index + 1} translated {direction_name} by {translation_distance}mm")
            return
        if self.selected_tile_index is None or self.selected_tile_index >= len(self.logic.tiles):
            self.statusLabel.setText("No tile is selected")
            return
        tile = self.logic.tiles[self.selected_tile_index]
        tile.center = tile.center + translation_vector
        tile.seed_positions_ras = []
        for seed_local in tile.seed_positions_local:
            seed_ras = tile.center + tile.frame @ seed_local
            tile.seed_positions_ras.append(seed_ras)
        self.updateTileModelPosition(self.selected_tile_index)
        self.statusLabel.setText(f"Tile {self.selected_tile_index + 1} translated {direction_name} by {translation_distance}mm")
    
    def onSliceViewClick(self, sliceViewName, x, y):
        if not self.placing or self.modeComboBox.currentIndex != 1:
            return
        try:
            vol = self.volumeSelector.currentNode()
            if not vol:
                self.statusLabel.setText("Select MRI volume first")
                return
            self.statusLabel.setText(f"Casting ray from {sliceViewName} slice view click...")
            slicer.app.processEvents()
            ray_origin, ray_direction = self.convertSliceClickToRay(sliceViewName, x, y)
            if ray_origin is None or ray_direction is None:
                self.statusLabel.setText(f"Error: Could not convert {sliceViewName} slice click to ray")
                return
            slice_plane_normal = ray_direction / np.linalg.norm(ray_direction)
            surface_result = self.findNearestSurfaceFromRay(ray_origin, ray_direction, search_radius=100.0)
            if surface_result is None:
                self.statusLabel.setText(f"No surface found along ray from {sliceViewName} slice view")
                return
            surface_point, surface_normal, surface_type, distance = surface_result
            to_ray_origin = ray_origin - surface_point
            dot_product = np.dot(to_ray_origin, surface_normal)
            if dot_product < 0:
                surface_normal = -surface_normal
            positioning_normal = surface_normal.copy()
            layoutManager = slicer.app.layoutManager()
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            slice_x_axis = None
            slice_y_axis = None
            if sliceWidget:
                sliceNode = sliceWidget.sliceLogic().GetSliceNode()
                if sliceNode:
                    xyToRas = sliceNode.GetXYToRAS()
                    if xyToRas:
                        slice_x_homog = [1.0, 0.0, 0.0, 0.0]
                        slice_y_homog = [0.0, 1.0, 0.0, 0.0]
                        ras_x = [0.0, 0.0, 0.0, 0.0]
                        ras_y = [0.0, 0.0, 0.0, 0.0]
                        xyToRas.MultiplyPoint(slice_x_homog, ras_x)
                        xyToRas.MultiplyPoint(slice_y_homog, ras_y)
                        slice_x_axis = np.array([ras_x[0], ras_x[1], ras_x[2]], dtype=np.float32)
                        slice_y_axis = np.array([ras_y[0], ras_y[1], ras_y[2]], dtype=np.float32)
                        slice_x_axis = slice_x_axis / np.linalg.norm(slice_x_axis)
                        slice_y_axis = slice_y_axis / np.linalg.norm(slice_y_axis)
            
            surface_normal_component_along_slice = np.dot(surface_normal, slice_plane_normal)
            in_plane_normal = surface_normal - surface_normal_component_along_slice * slice_plane_normal

            in_plane_norm = np.linalg.norm(in_plane_normal)
            if in_plane_norm > 1e-6:
                tile_orientation_normal = in_plane_normal / in_plane_norm

                original_in_plane = surface_normal - surface_normal_component_along_slice * slice_plane_normal
                if np.linalg.norm(original_in_plane) > 1e-6:
                    original_in_plane = original_in_plane / np.linalg.norm(original_in_plane)
                    if np.dot(tile_orientation_normal, original_in_plane) < 0:
                        tile_orientation_normal = -tile_orientation_normal
            else:
                if slice_x_axis is not None:
                    tile_orientation_normal = slice_x_axis.copy()
                else:
                    tile_orientation_normal = np.array([1, 0, 0], dtype=np.float32)

            surface_type_names = {
                'cavity': 'cavity wall',
                'interior': 'interior wall',
                'exterior': 'exterior wall'
            }
            surface_name = surface_type_names.get(surface_type, 'surface')
            self.statusLabel.setText(f"Found {surface_name} at depth {distance:.1f} mm from {sliceViewName} slice")

            imageData = vol.GetImageData()
            if not imageData:
                return
            
            spacing = imageData.GetSpacing()
            spacing_mm = np.mean(spacing)

            click_pos_ras = surface_point

            settled_pos = self.settleToSurface(vol, click_pos_ras, positioning_normal)

            adjusted_pos, penetration_depth = self.checkAndLimitPenetration(vol, settled_pos, positioning_normal)
            if penetration_depth > GammaTilePlacement.MAX_PENETRATION_MM:
                self.statusLabel.setText(
                    f"Warning: Tile penetration limited to {GammaTilePlacement.MAX_PENETRATION_MM}mm "
                    f"(was {penetration_depth:.1f}mm). Tile adjusted."
                )
            
            final_pos, _ = self.logic.place_tile(adjusted_pos, tile_orientation_normal, spacing_mm)

            try:
                tileModel = self.createTileModel(final_pos, tile_orientation_normal, spacing_mm)
                if tileModel:
                    self.logic.fiducial_nodes.append(tileModel)

                    try:
                        threeDWidget = slicer.app.layoutManager().threeDWidget(0)
                        if threeDWidget:
                            threeDView = threeDWidget.threeDView()
                            if threeDView:
                                threeDView.scheduleRender()

                        layoutManager = slicer.app.layoutManager()
                        for svName in ['Red', 'Yellow', 'Green']:
                            sliceWidget = layoutManager.sliceWidget(svName)
                            if sliceWidget:
                                sliceView = sliceWidget.sliceView()
                                if sliceView:
                                    sliceView.scheduleRender()
                    except:
                        pass
                    
                    self.updateTileCounter()
                    self.statusLabel.setText(
                        f"Tile {len(self.logic.tiles)} placed on {surface_name} from {sliceViewName} slice. "
                        f"Click to place another tile."
                    )
                else:
                    self.statusLabel.setText("Error: Tile model creation failed.")
            except Exception as e:
                error_msg = f"Error creating tile model: {str(e)}"
                self.statusLabel.setText(error_msg)
                logger.error(error_msg, exc_info=True)
            
        except Exception as e:
            error_msg = f"Error in slice view click handler: {str(e)}"
            self.statusLabel.setText(error_msg)
            logger.error(error_msg, exc_info=True)

    def onSliceViewSeedClick(self, sliceViewName, x, y):
        if not self.placing_seeds or self.modeComboBox.currentIndex != 2:
            return
        try:
            vol = self.volumeSelector.currentNode()
            if not vol:
                return
            ray_origin, ray_direction = self.convertSliceClickToRay(sliceViewName, x, y)
            if ray_origin is None or ray_direction is None:
                return
            surface_result = self.findNearestSurfaceFromRay(ray_origin, ray_direction, search_radius=100.0)
            if surface_result is None:
                self.statusLabel.setText(f"No surface found from {sliceViewName} slice.")
                return
            surface_point, surface_normal, surface_type, distance = surface_result
            to_ray_origin = ray_origin - surface_point
            if np.dot(to_ray_origin, surface_normal) < 0:
                surface_normal = -surface_normal
            positioning_normal = surface_normal.copy()
            click_pos_ras = surface_point
            settled_pos = self.settleToSurface(vol, click_pos_ras, positioning_normal)
            adjusted_pos, penetration_depth = self.checkAndLimitPenetration(vol, settled_pos, positioning_normal)
            # Offset seed center so one end of seed is on surface (stick like tiles in 2D slices)
            seed_half_len = SingleSeedPlacement.SEED_LENGTH_MM / 2.0
            seed_center = adjusted_pos - seed_half_len * positioning_normal
            self.logic.place_seed(seed_center, positioning_normal)
            try:
                seedModel = self.createSeedModel(seed_center, positioning_normal)
                if seedModel:
                    self.logic.seed_fiducial_nodes.append(seedModel)
                    self.updateTileCounter()
                    layoutManager = slicer.app.layoutManager()
                    for svName in ['Red', 'Yellow', 'Green']:
                        sw = layoutManager.sliceWidget(svName)
                        if sw and sw.sliceView():
                            sw.sliceView().scheduleRender()
                    self.statusLabel.setText(f"Seed {len(self.logic.seeds)} placed from {sliceViewName} slice. Click to place another.")
                else:
                    self.logic.seeds.pop()
            except Exception as e:
                if self.logic.seeds:
                    self.logic.seeds.pop()
                self.statusLabel.setText(f"Error: {str(e)}")
        except Exception as e:
            self.statusLabel.setText(f"Error: {str(e)}")
    
    def _getVolumeSamplingData(self, volumeNode):
        """Cached (numpy scalars indexed [k, j, i], dims, rasToIjk 4x4 numpy) for fast volume sampling."""
        imageData = volumeNode.GetImageData()
        if not imageData:
            return None
        scalars = imageData.GetPointData().GetScalars()
        if not scalars:
            return None
        # Key on both the pixel data and the node itself so geometry changes
        # (origin/spacing/IJKToRAS edits, hardened transforms) invalidate the cache.
        cache_key = (imageData.GetMTime(), volumeNode.GetMTime())
        cached = self._volume_array_cache.get(volumeNode.GetID())
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        dims = imageData.GetDimensions()
        np_array = numpy_support.vtk_to_numpy(scalars).reshape(dims[::-1])
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        rasToIjk_vtk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk_vtk)
        rasToIjk = np.array([[rasToIjk_vtk.GetElement(r, c) for c in range(4)] for r in range(4)])
        data = (np_array, dims, rasToIjk)
        self._volume_array_cache[volumeNode.GetID()] = (cache_key, data)
        return data

    def _sampleVolumeAtRAS(self, sampling_data, points_ras):
        """Nearest-neighbor voxel intensities at an (N, 3) array of RAS points.
        Returns (values, in_bounds_mask); out-of-bounds values are 0."""
        np_array, dims, rasToIjk = sampling_data
        pts = np.asarray(points_ras, dtype=np.float64).reshape(-1, 3)
        homog = np.column_stack([pts, np.ones(pts.shape[0])])
        ijk = np.rint(homog @ rasToIjk.T[:, :3]).astype(np.int64)
        in_bounds = ((ijk[:, 0] >= 0) & (ijk[:, 0] < dims[0]) &
                     (ijk[:, 1] >= 0) & (ijk[:, 1] < dims[1]) &
                     (ijk[:, 2] >= 0) & (ijk[:, 2] < dims[2]))
        values = np.zeros(pts.shape[0], dtype=np.float64)
        ib = ijk[in_bounds]
        if ib.size:
            values[in_bounds] = np_array[ib[:, 2], ib[:, 1], ib[:, 0]]
        return values, in_bounds

    def _buildPointLocator(self, polyData):
        locator = vtk.vtkStaticPointLocator() if hasattr(vtk, 'vtkStaticPointLocator') else vtk.vtkPointLocator()
        locator.SetDataSet(polyData)
        locator.BuildLocator()
        return locator

    def _getCellLocator(self, polyData):
        """Cached static cell locator for a polydata (rebuilt only when the mesh changes)."""
        if not hasattr(self, '_cell_locator_cache'):
            self._cell_locator_cache = {}
        key = id(polyData)
        cached = self._cell_locator_cache.get(key)
        if cached is not None and cached[0] == polyData.GetMTime():
            return cached[1]
        locator = vtk.vtkStaticCellLocator() if hasattr(vtk, 'vtkStaticCellLocator') else vtk.vtkCellLocator()
        locator.SetDataSet(polyData)
        locator.BuildLocator()
        # Keep a reference to polyData so its id() is not reused while cached.
        # Only the most recent mesh is retained so regenerated surfaces are not leaked.
        self._cell_locator_cache = {key: (polyData.GetMTime(), locator, polyData)}
        return locator

    def getNormalFromModel(self, modelNode, pos_ras, search_radius=5.0):
        try:
            polyData = modelNode.GetPolyData()
            if not polyData or polyData.GetNumberOfPoints() == 0:
                return None

            locator = self._buildPointLocator(polyData)
            nearest_idx = locator.FindClosestPoint([float(pos_ras[0]), float(pos_ras[1]), float(pos_ras[2])])
            if nearest_idx < 0:
                return None
            nearest_pt = np.array(polyData.GetPoint(nearest_idx))
            if np.linalg.norm(nearest_pt - pos_ras) > search_radius:
                return None

            normals = polyData.GetPointData().GetNormals()
            if normals:
                normal = normals.GetTuple(nearest_idx)
                return np.array(normal, dtype=np.float32)

            return self.computeNormalFromNeighbors(polyData, nearest_idx)
        except:
            return None

    def computeNormalFromNeighbors(self, polyData, point_idx, radius=2.0):
        try:
            points = polyData.GetPoints()
            center = np.array(points.GetPoint(point_idx))

            locator = self._buildPointLocator(polyData)
            neighbor_ids = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, [float(center[0]), float(center[1]), float(center[2])], neighbor_ids)
            if neighbor_ids.GetNumberOfIds() < 3:
                return None

            all_points = numpy_support.vtk_to_numpy(points.GetData())
            ids = [neighbor_ids.GetId(i) for i in range(neighbor_ids.GetNumberOfIds())]
            neighbors = np.asarray(all_points[ids], dtype=np.float64)

            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid

            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]

            if np.dot(normal, center) < 0:
                normal = -normal

            return normal.astype(np.float32)
        except:
            return None
    
    def computeSurfaceNormal(self, volumeNode, pos_ras):
        image = volumeNode.GetImageData()
        if not image:
            return None

        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        rasToIjk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)
        
        pos_homog = [pos_ras[0], pos_ras[1], pos_ras[2], 1.0]
        ijk = [0, 0, 0, 0]
        rasToIjk.MultiplyPoint(pos_homog, ijk)
        
        i, j, k = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))
        
        scalars = image.GetPointData().GetScalars()
        dims = image.GetDimensions()

        i = max(1, min(i, dims[0]-2))
        j = max(1, min(j, dims[1]-2))
        k = max(1, min(k, dims[2]-2))

        def getVoxel(ii, jj, kk):
            if 0 <= ii < dims[0] and 0 <= jj < dims[1] and 0 <= kk < dims[2]:
                idx = ii + jj*dims[0] + kk*dims[0]*dims[1]
                return scalars.GetValue(idx)
            return 0
        
        gx = (getVoxel(i+1,j,k) - getVoxel(i-1,j,k)) * 2.0
        gx += getVoxel(i+1,j+1,k) - getVoxel(i-1,j+1,k)
        gx += getVoxel(i+1,j-1,k) - getVoxel(i-1,j-1,k)
        gx += getVoxel(i+1,j,k+1) - getVoxel(i-1,j,k+1)
        gx += getVoxel(i+1,j,k-1) - getVoxel(i-1,j,k-1)
        
        gy = (getVoxel(i,j+1,k) - getVoxel(i,j-1,k)) * 2.0
        gy += getVoxel(i+1,j+1,k) - getVoxel(i+1,j-1,k)
        gy += getVoxel(i-1,j+1,k) - getVoxel(i-1,j-1,k)
        gy += getVoxel(i,j+1,k+1) - getVoxel(i,j-1,k+1)
        gy += getVoxel(i,j+1,k-1) - getVoxel(i,j-1,k-1)
        
        gz = (getVoxel(i,j,k+1) - getVoxel(i,j,k-1)) * 2.0
        gz += getVoxel(i+1,j,k+1) - getVoxel(i+1,j,k-1)
        gz += getVoxel(i-1,j,k+1) - getVoxel(i-1,j,k-1)
        gz += getVoxel(i,j+1,k+1) - getVoxel(i,j+1,k-1)
        gz += getVoxel(i,j-1,k+1) - getVoxel(i,j-1,k-1)

        grad_ijk = np.array([gx, gy, gz])
        spacing = np.array(image.GetSpacing())
        grad_ijk = grad_ijk / (spacing + 1e-6)

        rot = np.zeros((3, 3))
        for i_idx in range(3):
            for j_idx in range(3):
                rot[i_idx, j_idx] = ijkToRas.GetElement(i_idx, j_idx)
        
        grad_ras = rot @ grad_ijk
        norm = np.linalg.norm(grad_ras)
        if norm < 1e-6:
            return None
        
        normal = grad_ras / norm
        return normal.astype(np.float32)
    
    def settleToSurface(self, volumeNode, pos_ras, normal, max_trace=20.0, step=0.2):
        """Settle onto brain surface along normal; returns tile center so bottom face sits on surface."""
        if self.brain_surface_model:
            surface_pos = self.findNearestPointOnSurface(self.brain_surface_model, pos_ras)
            if surface_pos is not None:
                # Position tile so its bottom face (2mm below center) sits on surface
                tile_thickness_half = GammaTilePlacement.TILE_THICKNESS_MM / 2.0
                # Move tile center up by half thickness so bottom sits on surface
                tile_center = surface_pos + normal * tile_thickness_half
                return tile_center
        
        # Fallback: volume-based surface detection (vectorized ray march on cached array)
        sampling_data = self._getVolumeSamplingData(volumeNode)
        if sampling_data is None:
            return np.array(pos_ras, dtype=np.float32)

        pos = np.asarray(pos_ras, dtype=np.float64)
        normal64 = np.asarray(normal, dtype=np.float64)
        max_steps = int(max_trace / step)
        tile_thickness_half = GammaTilePlacement.TILE_THICKNESS_MM / 2.0

        # Get initial intensity to determine threshold
        init_val, init_in = self._sampleVolumeAtRAS(sampling_data, pos)
        threshold = max(30, init_val[0] * 0.4) if init_in[0] else 50  # Adaptive threshold

        # Trace inward (opposite to normal) to find brain surface
        t = np.arange(max_steps, dtype=np.float64) * step
        samples = pos[None, :] - normal64[None, :] * t[:, None]
        intensities, in_bounds = self._sampleVolumeAtRAS(sampling_data, samples)

        # Only the contiguous in-bounds prefix is traced (march stops at first out-of-bounds voxel)
        oob = np.where(~in_bounds)[0]
        limit = int(oob[0]) if oob.size else max_steps
        below = np.where(intensities[:limit] < threshold)[0]

        if below.size:
            # Move back to surface boundary, then position tile so bottom face sits on surface
            surface_pos = samples[below[0]] + normal64 * (step * 0.5)
            return (surface_pos + normal64 * tile_thickness_half).astype(np.float32)
        # If surface not found, use original position but adjust for tile thickness
        return pos_ras + normal * tile_thickness_half
    
    def findNearestPointOnSurface(self, modelNode, pos_ras, search_radius=10.0):
        try:
            polyData = modelNode.GetPolyData()
            if not polyData or polyData.GetNumberOfPoints() == 0:
                return None

            locator = self._getCellLocator(polyData)

            closestPoint = [0.0, 0.0, 0.0]
            cellId = [0]
            subId = [0]
            dist2 = [0.0]
            
            locator.FindClosestPoint(
                [pos_ras[0], pos_ras[1], pos_ras[2]],
                closestPoint,
                cellId,
                subId,
                dist2
            )
            
            if dist2[0] < search_radius ** 2:
                return np.array(closestPoint, dtype=np.float32)
            
            return None
        except Exception as e:
            return None
    
    def castRayThroughVolume(self, ray_origin, ray_direction, volumeNode, max_distance=100.0):
        """Cast ray through volume; detect surfaces by intensity transitions. Returns [(dist, point, normal, type), ...]."""
        intersections = []

        try:
            sampling_data = self._getVolumeSamplingData(volumeNode)
            if sampling_data is None:
                return intersections

            ray_origin = np.asarray(ray_origin, dtype=np.float64)
            ray_direction = np.asarray(ray_direction, dtype=np.float64)

            step_size = 1.0
            num_steps = int(max_distance / step_size)
            if num_steps < 1:
                return intersections

            # Sample the whole ray in one vectorized pass on the cached array
            t = np.arange(num_steps, dtype=np.float64) * step_size
            points = ray_origin[None, :] + ray_direction[None, :] * t[:, None]
            values, in_bounds = self._sampleVolumeAtRAS(sampling_data, points)

            # Thresholds from the first few (in-bounds) samples along the ray
            n_lead = min(10, num_steps)
            lead_vals = values[:n_lead][in_bounds[:n_lead]]
            if lead_vals.size == 0:
                return intersections
            mean_intensity = np.mean(lead_vals)
            std_intensity = np.std(lead_vals) if lead_vals.size > 1 else 1.0

            cavity_threshold = mean_intensity - 0.5 * std_intensity
            brain_threshold = mean_intensity + 0.3 * std_intensity
            bone_threshold = mean_intensity + 2.0 * std_intensity

            # March stops at the first out-of-bounds sample
            oob = np.where(~in_bounds)[0]
            limit = int(oob[0]) if oob.size else num_steps
            if limit < 2:
                return intersections
            vals = values[:limit]

            prev = vals[:-1]
            curr = vals[1:]
            # Transition tests in the same priority order as the original elif chain
            c1 = (prev >= cavity_threshold) & (curr < cavity_threshold)
            c2 = ~c1 & (prev < cavity_threshold) & (curr >= brain_threshold)
            c3 = ~c1 & ~c2 & (prev < brain_threshold) & (curr >= brain_threshold)
            c4 = ~c1 & ~c2 & ~c3 & (prev >= brain_threshold) & (curr < brain_threshold)
            c5 = ~c1 & ~c2 & ~c3 & ~c4 & (curr >= bone_threshold)

            for k in np.where(c1 | c2 | c3 | c4 | c5)[0]:
                t_hit = t[k + 1]
                if c1[k]:
                    intersections.append((t_hit, points[k].copy(), -ray_direction.copy(), 'cavity'))
                elif c2[k]:
                    intersections.append((t_hit, points[k + 1].copy(), ray_direction.copy(), 'cavity'))
                elif c3[k]:
                    intersections.append((t_hit, points[k + 1].copy(), ray_direction.copy(), 'interior'))
                elif c4[k]:
                    intersections.append((t_hit, points[k].copy(), -ray_direction.copy(), 'interior'))
                elif c5[k]:
                    # A barrier transition terminates the march
                    intersections.append((t_hit, points[k + 1].copy(), ray_direction.copy(), 'barrier'))
                    break

            return intersections

        except Exception as e:
            logger.warning(f"Error casting ray through volume: {e}")
            return []
    
    def castRayToSurfaces(self, ray_origin, ray_direction, max_distance=100.0):
        """Cast ray; return surface intersections (model then volume fallback), sorted by distance."""
        intersections = []
        
        try:
            vol = self.volumeSelector.currentNode()
            if not vol:
                return intersections
            
            if self.brain_surface_model:
                try:
                    surfacePolyData = self.brain_surface_model.GetPolyData()
                    if surfacePolyData and surfacePolyData.GetNumberOfPoints() > 0:
                        locator = self._getCellLocator(surfacePolyData)

                        tolerance = 0.001
                        points = vtk.vtkPoints()
                        cellIds = vtk.vtkIdList()
                        
                        ray_end = ray_origin + ray_direction * max_distance
                        locator.IntersectWithLine(
                            [ray_origin[0], ray_origin[1], ray_origin[2]],
                            [ray_end[0], ray_end[1], ray_end[2]],
                            tolerance,
                            points,
                            cellIds
                        )
                        
                        for i in range(points.GetNumberOfPoints()):
                            point_ras = np.array(points.GetPoint(i), dtype=np.float32)
                            distance = np.linalg.norm(point_ras - ray_origin)

                            if cellIds.GetNumberOfIds() > i:
                                cellId = cellIds.GetId(i)
                                cell = surfacePolyData.GetCell(cellId)
                                if cell:
                                    points_cell = cell.GetPoints()
                                    if points_cell.GetNumberOfPoints() >= 3:
                                        p0 = np.array(points_cell.GetPoint(0))
                                        p1 = np.array(points_cell.GetPoint(1))
                                        p2 = np.array(points_cell.GetPoint(2))
                                        v1 = p1 - p0
                                        v2 = p2 - p0
                                        normal = np.cross(v1, v2)
                                        norm = np.linalg.norm(normal)
                                        if norm > 1e-6:
                                            normal = normal / norm
                                        else:
                                            normal = ray_direction.copy()
                                    else:
                                        normal = ray_direction.copy()
                                else:
                                    normal = ray_direction.copy()
                            else:
                                normal = ray_direction.copy()

                            intersections.append((distance, point_ras, normal, 'exterior'))
                except Exception as e:
                    logger.warning(f"Error in model-based ray casting: {e}")
            vol_intersections = self.castRayThroughVolume(ray_origin, ray_direction, vol, max_distance)

            all_intersections = intersections + vol_intersections
            all_intersections.sort(key=lambda x: x[0])
            
            return all_intersections
            
        except Exception as e:
            logger.warning(f"Error casting ray to surfaces: {e}")
            return []
    
    def findNearestSurfaceFromRay(self, ray_origin, ray_direction, search_radius=100.0):
        """Nearest surface along ray (cavity > interior > exterior). Returns (point, normal, type, dist) or None."""
        try:
            intersections = self.castRayToSurfaces(ray_origin, ray_direction, search_radius)
            
            if not intersections:
                return None

            cavity_intersections = [i for i in intersections if i[3] == 'cavity']
            interior_intersections = [i for i in intersections if i[3] == 'interior']
            exterior_intersections = [i for i in intersections if i[3] == 'exterior']

            if cavity_intersections:
                distance, point, normal, surface_type = cavity_intersections[0]
                return point, normal, surface_type, distance
            elif interior_intersections:
                distance, point, normal, surface_type = interior_intersections[0]
                return point, normal, surface_type, distance
            elif exterior_intersections:
                distance, point, normal, surface_type = exterior_intersections[0]
                return point, normal, surface_type, distance
            
            return None
            
        except Exception as e:
            logger.warning(f"Error finding nearest surface from ray: {e}")
            return None
    
    def checkAndLimitPenetration(self, volumeNode, tile_center_pos, normal):
        """Limit tile penetration to MAX_PENETRATION_MM; return (adjusted_center, penetration_depth)."""
        max_penetration = GammaTilePlacement.MAX_PENETRATION_MM
        tile_thickness_half = GammaTilePlacement.TILE_THICKNESS_MM / 2.0

        tile_bottom_pos = tile_center_pos - normal * tile_thickness_half

        if self.brain_surface_model:
            surface_pos = self.findNearestPointOnSurface(self.brain_surface_model, tile_bottom_pos, search_radius=20.0)
        else:
            surface_pos = self.settleToSurface(volumeNode, tile_bottom_pos, normal)
            surface_pos = surface_pos - normal * tile_thickness_half
        
        if surface_pos is None:
            return tile_center_pos, 0.0

        to_surface = surface_pos - tile_bottom_pos
        penetration_depth = np.dot(to_surface, normal)

        if penetration_depth > max_penetration:
            adjustment = (penetration_depth - max_penetration) * normal
            adjusted_center = tile_center_pos + adjustment
            return adjusted_center, max_penetration
        elif penetration_depth < 0:
            adjustment = abs(penetration_depth) * normal
            adjusted_center = tile_center_pos - adjustment
            return adjusted_center, 0.0
        else:
            return tile_center_pos, penetration_depth
    
    def deformTileToSurface(self, polydata, position, normal, volumeNode):
        if not self.brain_surface_model:
            return polydata
        
        try:
            surfacePolyData = self.brain_surface_model.GetPolyData()
            if not surfacePolyData or surfacePolyData.GetNumberOfPoints() == 0:
                return polydata

            locator = self._getCellLocator(surfacePolyData)
            fallbackPointLocator = None  # built at most once if the cell-locator API is unavailable

            points = polydata.GetPoints()
            numPoints = points.GetNumberOfPoints()

            z = normal / (np.linalg.norm(normal) + 1e-6)
            if abs(z[0]) < 0.9:
                x_candidate = np.array([1, 0, 0], dtype=np.float32)
            else:
                x_candidate = np.array([0, 1, 0], dtype=np.float32)
            x = np.cross(z, x_candidate)
            x = x / (np.linalg.norm(x) + 1e-6)
            y = np.cross(z, x)
            frame = np.array([x, y, z]).T
            
            tile_thickness = GammaTilePlacement.TILE_THICKNESS_MM
            tile_half_size = GammaTilePlacement.TILE_LENGTH_MM / 2.0

            num_samples_x = 8
            num_samples_y = 8
            surface_data = {}
            
            for i in range(num_samples_x):
                for j in range(num_samples_y):
                    local_x = -tile_half_size + (i / (num_samples_x - 1)) * (2 * tile_half_size)
                    local_y = -tile_half_size + (j / (num_samples_y - 1)) * (2 * tile_half_size)
                    local_z = -tile_thickness / 2.0
                    
                    sample_local = np.array([local_x, local_y, local_z], dtype=np.float32)
                    sample_ras = position + frame @ sample_local

                    closestPoint = [0.0, 0.0, 0.0]
                    cell_id_val = 0
                    dist2_val = float('inf')
                    closest_point_id = -1

                    try:
                        if hasattr(vtk, 'reference'):
                            cellId = vtk.reference(0)
                            subId = vtk.reference(0)
                            dist2 = vtk.reference(0.0)
                        else:
                            cellId = [0]
                            subId = [0]
                            dist2 = [0.0]

                        locator.FindClosestPoint(
                            [float(sample_ras[0]), float(sample_ras[1]), float(sample_ras[2])],
                            closestPoint,
                            cellId,
                            subId,
                            dist2
                        )

                        if hasattr(vtk, 'reference'):
                            try:
                                cell_id_val = int(cellId.get())
                                dist2_val = float(dist2.get())
                            except AttributeError:
                                cell_id_val = cellId[0]
                                dist2_val = dist2[0]
                        else:
                            cell_id_val = cellId[0]
                            dist2_val = dist2[0]
                            
                    except (TypeError, AttributeError) as e:
                        try:
                            if fallbackPointLocator is None:
                                fallbackPointLocator = self._buildPointLocator(surfacePolyData)

                            closest_point_id = fallbackPointLocator.FindClosestPoint(
                                float(sample_ras[0]), 
                                float(sample_ras[1]), 
                                float(sample_ras[2])
                            )
                            
                            if closest_point_id >= 0:
                                points = surfacePolyData.GetPoints()
                                closestPoint = list(points.GetPoint(closest_point_id))
                                dist_vec = np.array(closestPoint) - sample_ras
                                dist2_val = float(np.dot(dist_vec, dist_vec))
                                cell_id_val = 0
                            else:
                                continue

                        except Exception as e2:
                            continue
                    
                    if dist2_val < 100.0:
                        try:
                            cell = surfacePolyData.GetCell(cell_id_val)
                        except:
                            cell = None
                            if closest_point_id >= 0:
                                cellArray = surfacePolyData.GetPolys()
                                if cellArray.GetNumberOfCells() > 0:
                                    cell = surfacePolyData.GetCell(0)
                            elif cell_id_val >= 0:
                                try:
                                    cell = surfacePolyData.GetCell(cell_id_val)
                                except:
                                    cell = None
                        
                        if cell:
                            normals = surfacePolyData.GetPointData().GetNormals()
                            if normals:
                                pointIds = cell.GetPointIds()
                                if pointIds.GetNumberOfIds() > 0:
                                    normal_vec = normals.GetTuple(pointIds.GetId(0))
                                    surface_normal = np.array(normal_vec, dtype=np.float32)
                                else:
                                    surface_normal = z
                            else:
                                points_cell = cell.GetPoints()
                                if points_cell.GetNumberOfPoints() >= 3:
                                    p0 = np.array(points_cell.GetPoint(0))
                                    p1 = np.array(points_cell.GetPoint(1))
                                    p2 = np.array(points_cell.GetPoint(2))
                                    v1 = p1 - p0
                                    v2 = p2 - p0
                                    surface_normal = np.cross(v1, v2)
                                    norm = np.linalg.norm(surface_normal)
                                    if norm > 1e-6:
                                        surface_normal = surface_normal / norm
                                    else:
                                        surface_normal = z
                                else:
                                    surface_normal = z
                            
                            surface_data[(i, j)] = {
                                'position': np.array(closestPoint, dtype=np.float32),
                                'normal': surface_normal,
                                'local': sample_local
                            }
            
            newPoints = vtk.vtkPoints()
            newPoints.SetNumberOfPoints(numPoints)
            
            for i in range(numPoints):
                pt = np.array(points.GetPoint(i))

                relative = pt - position
                local = frame.T @ relative

                local_x = local[0]
                local_y = local[1]
                local_z = local[2]

                i_sample = int((local_x + tile_half_size) / (2 * tile_half_size) * (num_samples_x - 1))
                j_sample = int((local_y + tile_half_size) / (2 * tile_half_size) * (num_samples_y - 1))
                i_sample = max(0, min(num_samples_x - 1, i_sample))
                j_sample = max(0, min(num_samples_y - 1, j_sample))

                i0, i1 = max(0, i_sample - 1), min(num_samples_x - 1, i_sample + 1)
                j0, j1 = max(0, j_sample - 1), min(num_samples_y - 1, j_sample + 1)

                samples = []
                for ii in [i0, i1]:
                    for jj in [j0, j1]:
                        if (ii, jj) in surface_data:
                            samples.append((ii, jj, surface_data[(ii, jj)]))
                
                if len(samples) > 0:
                    total_weight = 0.0
                    surface_pos_interp = np.zeros(3, dtype=np.float32)
                    surface_normal_interp = np.zeros(3, dtype=np.float32)
                    
                    for ii, jj, data in samples:
                        sample_local = data['local']
                        dist = np.sqrt((local_x - sample_local[0])**2 + (local_y - sample_local[1])**2)
                        weight = 1.0 / (dist + 0.1)
                        total_weight += weight
                        surface_pos_interp += data['position'] * weight
                        surface_normal_interp += data['normal'] * weight
                    
                    if total_weight > 0:
                        surface_pos_interp /= total_weight
                        surface_normal_interp /= (np.linalg.norm(surface_normal_interp) + 1e-6)
                    else:
                        surface_pos_interp = pt
                        surface_normal_interp = z
                    
                    if local_z < -tile_thickness / 4.0:
                        to_surface = surface_pos_interp - pt
                        proj_dist = np.dot(to_surface, surface_normal_interp)
                        deformed_pt = pt + surface_normal_interp * proj_dist * 0.9
                    else:
                        center_dist = np.sqrt(local_x**2 + local_y**2)
                        center_dist_norm = center_dist / (tile_half_size * np.sqrt(2))

                        bending_factor = center_dist_norm * 0.4

                        to_surface = surface_pos_interp - pt
                        proj_dist = np.dot(to_surface, surface_normal_interp)

                        z_factor = (local_z + tile_thickness / 2.0) / tile_thickness
                        bending_strength = bending_factor * (1.0 - z_factor * 0.5)

                        deformation = surface_normal_interp * proj_dist * bending_strength
                        deformed_pt = pt + deformation
                else:
                    deformed_pt = pt
                
                newPoints.SetPoint(i, deformed_pt[0], deformed_pt[1], deformed_pt[2])
            
            deformedPolyData = vtk.vtkPolyData()
            deformedPolyData.ShallowCopy(polydata)
            deformedPolyData.SetPoints(newPoints)
            deformedPolyData.Modified()
            
            return deformedPolyData
            
        except Exception as e:
            logger.error(f"Error in deformTileToSurface: {e}")
            import traceback
            traceback.print_exc()
            return polydata
    
    def createTileModel(self, position, normal, spacing_mm=1.0):
        tile_length = GammaTilePlacement.TILE_LENGTH_MM
        tile_width = GammaTilePlacement.TILE_WIDTH_MM
        tile_thickness_total = GammaTilePlacement.TILE_THICKNESS_MM

        plane = vtk.vtkCubeSource()
        plane.SetXLength(tile_length)
        plane.SetYLength(tile_width)
        plane.SetZLength(tile_thickness_total)
        plane.SetCenter(0, 0, 0)
        plane.Update()
        
        polydata = plane.GetOutput()

        triangulator = vtk.vtkTriangleFilter()
        triangulator.SetInputData(polydata)
        triangulator.Update()
        polydata = triangulator.GetOutput()

        subdivider = vtk.vtkLinearSubdivisionFilter()
        subdivider.SetInputData(polydata)
        subdivider.SetNumberOfSubdivisions(1)
        subdivider.Update()
        polydata = subdivider.GetOutput()

        transform = vtk.vtkTransform()
        transform.Translate(position[0], position[1], position[2])

        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, normal), -1, 1)))
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            transform.RotateWXYZ(rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(polydata)
        transformFilter.Update()
        
        polydata = transformFilter.GetOutput()

        vol = self.volumeSelector.currentNode()
        if vol:
            polydata = self.deformTileToSurface(polydata, position, normal, vol)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"GammaTile_{len(self.logic.tiles)}")

        if polydata.GetNumberOfPoints() == 0:
            return None
        
        modelNode.SetAndObservePolyData(polydata)

        displayNode = modelNode.GetDisplayNode()
        if displayNode is None:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        
        displayNode.SetColor(1.0, 0.8, 0.0)
        displayNode.SetOpacity(1.0)
        displayNode.SetAmbient(0.5)
        displayNode.SetDiffuse(0.8)
        displayNode.SetSpecular(0.3)
        displayNode.SetBackfaceCulling(False)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)

        if not modelNode.GetScene():
            slicer.mrmlScene.AddNode(modelNode)

        try:
            threeDWidget = slicer.app.layoutManager().threeDWidget(0)
            if threeDWidget:
                threeDView = threeDWidget.threeDView()
                if threeDView:
                    threeDView.scheduleRender()
        except:
            pass
        
        return modelNode

    def createSeedModel(self, position, normal):
        length_mm = SingleSeedPlacement.SEED_LENGTH_MM
        radius_mm = SingleSeedPlacement.SEED_DIAMETER_MM / 2.0
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(radius_mm)
        cylinder.SetHeight(length_mm)
        cylinder.SetCenter(0, 0, 0)
        cylinder.Update()
        polydata = cylinder.GetOutput()
        transform = vtk.vtkTransform()
        transform.Translate(position[0], position[1], position[2])
        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(y_axis, normal)
        n = np.linalg.norm(rotation_axis)
        if n > 1e-6:
            rotation_axis = rotation_axis / n
            angle_deg = np.degrees(np.arccos(np.clip(np.dot(y_axis, normal), -1, 1)))
            transform.RotateWXYZ(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(polydata)
        transformFilter.Update()
        polydata = transformFilter.GetOutput()
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Seed_{len(self.logic.seeds)}")
        modelNode.SetAndObservePolyData(polydata)
        displayNode = modelNode.GetDisplayNode()
        if displayNode is None:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(0.2, 0.8, 0.2)
        displayNode.SetOpacity(1.0)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetSliceIntersectionThickness(2)
        if not modelNode.GetScene():
            slicer.mrmlScene.AddNode(modelNode)
        return modelNode

    def updateSeedModelPosition(self, seed_index):
        if seed_index < 0 or seed_index >= len(self.logic.seeds) or seed_index >= len(self.logic.seed_fiducial_nodes):
            return
        seed = self.logic.seeds[seed_index]
        modelNode = self.logic.seed_fiducial_nodes[seed_index]
        if not modelNode:
            return
        length_mm = SingleSeedPlacement.SEED_LENGTH_MM
        radius_mm = SingleSeedPlacement.SEED_DIAMETER_MM / 2.0
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(radius_mm)
        cylinder.SetHeight(length_mm)
        cylinder.SetCenter(0, 0, 0)
        cylinder.Update()
        polydata = cylinder.GetOutput()
        transform = vtk.vtkTransform()
        transform.Translate(seed.center[0], seed.center[1], seed.center[2])
        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(y_axis, seed.normal)
        n = np.linalg.norm(rotation_axis)
        if n > 1e-6:
            rotation_axis = rotation_axis / n
            angle_deg = np.degrees(np.arccos(np.clip(np.dot(y_axis, seed.normal), -1, 1)))
            transform.RotateWXYZ(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(polydata)
        transformFilter.Update()
        modelNode.SetAndObservePolyData(transformFilter.GetOutput())

    def updateDoseLegend(self, doses_3d, max_dose):
        for item in self.legendItems:
            item.setParent(None)
        self.legendItems = []
        
        layout = self.legendWidget.layout()
        if layout:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        header3D = qt.QLabel("3D Isodoses:")
        header3D.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(header3D)
        self.legendItems.append(header3D)

        for dose_level, color, label in doses_3d:
            if dose_level <= max_dose:
                itemLayout = qt.QHBoxLayout()
                colorLabel = qt.QLabel()
                colorLabel.setFixedSize(20, 20)
                colorLabel.setStyleSheet(f"background-color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}); border: 1px solid black;")
                itemLayout.addWidget(colorLabel)
                textLabel = qt.QLabel(f"{label}: {dose_level:.0f} cGy")
                itemLayout.addWidget(textLabel)
                itemLayout.addStretch()
                layout.addLayout(itemLayout)
        
        self.legendWidget.update()
    
    def onUndo(self):
        self.logic.undo_tile()

        if self.selected_tile_index is not None:
            if self.selected_tile_index >= len(self.logic.tiles):
                self.selected_tile_index = None
                self.selected_from_slice_view = None
                self.updateSelectionDisplay()
        
        self.statusLabel.setText(f"Tiles: {len(self.logic.tiles)}  Seeds: {len(self.logic.seeds)}")

    def onUndoSeed(self):
        self.logic.undo_seed()
        if self.selected_seed_index is not None and self.selected_seed_index >= len(self.logic.seeds):
            self.selected_seed_index = None
            self.selected_from_slice_view = None
            self.updateSelectionDisplay()
        self.statusLabel.setText(f"Tiles: {len(self.logic.tiles)}  Seeds: {len(self.logic.seeds)}")
    
    def printDosageSummary(self, prescription_cgy, max_dose, num_tiles, dims, spacing, bounds_computed, doses_3d, num_seeds=0):
        """Print a clean dosage summary to console."""
        print("\n" + "="*70)
        print("DOSAGE SUMMARY")
        print("="*70)
        print(f"Prescription dose: {prescription_cgy:.1f} cGy ({prescription_cgy/100:.1f} Gy)")
        print(f"Maximum computed dose: {max_dose:.1f} cGy")
        print(f"Number of tiles: {num_tiles}")
        print(f"Number of standalone seeds: {num_seeds}")
        print(f"Dosimetry dataset: {self.logic.interpolator.dataset_name}")
        print(f"Dose volume dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
        print(f"Spacing: {spacing:.2f} mm")
        print(f"Volume bounds (RAS):")
        print(f"  X: [{bounds_computed[0][0]:.2f}, {bounds_computed[0][1]:.2f}] mm")
        print(f"  Y: [{bounds_computed[1][0]:.2f}, {bounds_computed[1][1]:.2f}] mm")
        print(f"  Z: [{bounds_computed[2][0]:.2f}, {bounds_computed[2][1]:.2f}] mm")
        print("\n3D Isodose Levels:")
        for dose_level, color, label in doses_3d:
            if dose_level <= max_dose:
                print(f"  {label:>4s}: {dose_level:>7.1f} cGy - RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
        print("="*70 + "\n")

    def onExportPositions(self):
        if not self.logic.tiles and not self.logic.seeds:
            slicer.util.warningDisplay("No tiles or seeds placed. Place tiles or seeds before exporting positions.")
            return
        result = qt.QFileDialog.getSaveFileName(
            self.parent, "Export seed and tile positions", "", "CSV (*.csv);;JSON (*.json)"
        )
        path = result[0] if isinstance(result, (tuple, list)) else result
        selected_filter = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else ""
        if not path or not str(path).strip():
            return
        path = str(path).strip()
        ext = os.path.splitext(path)[1].lower()
        if "json" in selected_filter or ext == ".json":
            self._writePositionsJSON(path)
        else:
            self._writePositionsCSV(path)
        self.statusLabel.setText(f"Exported positions to {os.path.basename(path)}")
    
    def _isodoseSurfaceBounds(self):
        """Axis-aligned RAS bounding box of each isodose surface.

        The same model nodes drive the 3D surfaces and their 2D slice-view
        contours, so one box per dose level bounds both representations.
        Returns [(name, dose_level_cGy, (xmin, xmax, ymin, ymax, zmin, zmax)), ...].
        """
        rows = []
        for node in self.logic.isodose_nodes_3d:
            if not node or not node.GetScene():
                continue
            polydata = node.GetPolyData()
            if not polydata or polydata.GetNumberOfPoints() == 0:
                continue
            bounds = [0.0] * 6
            try:
                node.GetRASBounds(bounds)  # accounts for any parent transforms
            except (AttributeError, TypeError):
                bounds = list(polydata.GetBounds())
            rows.append((node.GetName(), node.GetAttribute("DoseLevel_cGy") or "", tuple(bounds)))
        return rows

    def _writePositionsCSV(self, path):
        """Write tile and seed positions to CSV (consistent with onUpdate log)."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dosimetry_dataset", self.logic.interpolator.dataset_name])
            w.writerow([])
            w.writerow(["tile_index", "seed_index", "x_ras_mm", "y_ras_mm", "z_ras_mm", "x_local_mm", "y_local_mm", "z_local_mm"])
            for tile_idx, tile in enumerate(self.logic.tiles):
                for seed_idx, seed_pos in enumerate(tile.seed_positions_ras):
                    seed_local = tile.seed_positions_local[seed_idx]
                    w.writerow([
                        tile_idx, seed_idx + 1,
                        f"{seed_pos[0]:.6f}", f"{seed_pos[1]:.6f}", f"{seed_pos[2]:.6f}",
                        f"{seed_local[0]:.6f}", f"{seed_local[1]:.6f}", f"{seed_local[2]:.6f}"
                    ])
            w.writerow([])
            w.writerow(["tile_index", "center_x_ras", "center_y_ras", "center_z_ras", "normal_x", "normal_y", "normal_z", "seed_axis_x", "seed_axis_y", "seed_axis_z"])
            for tile_idx, tile in enumerate(self.logic.tiles):
                c, n, a = tile.center, tile.normal, tile.seed_axis_ras
                w.writerow([
                    tile_idx,
                    f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}",
                    f"{n[0]:.6f}", f"{n[1]:.6f}", f"{n[2]:.6f}",
                    f"{a[0]:.6f}", f"{a[1]:.6f}", f"{a[2]:.6f}"
                ])
            if self.logic.seeds:
                w.writerow([])
                w.writerow(["standalone_seed_index", "x_ras_mm", "y_ras_mm", "z_ras_mm", "normal_x", "normal_y", "normal_z"])
                for seed_idx, seed in enumerate(self.logic.seeds):
                    c, n = seed.center, seed.normal
                    w.writerow([
                        seed_idx,
                        f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}",
                        f"{n[0]:.6f}", f"{n[1]:.6f}", f"{n[2]:.6f}"
                    ])
            isodose_bounds = self._isodoseSurfaceBounds()
            if isodose_bounds:
                w.writerow([])
                w.writerow(["isodose_surface", "dose_level_cGy",
                            "min_x_ras_mm", "max_x_ras_mm",
                            "min_y_ras_mm", "max_y_ras_mm",
                            "min_z_ras_mm", "max_z_ras_mm"])
                for name, level, b in isodose_bounds:
                    w.writerow([
                        name, level,
                        f"{b[0]:.6f}", f"{b[1]:.6f}",
                        f"{b[2]:.6f}", f"{b[3]:.6f}",
                        f"{b[4]:.6f}", f"{b[5]:.6f}"
                    ])

    def _writePositionsJSON(self, path):
        """Write tile and seed positions to JSON (list of tiles with center, normal, seed_axis, seeds)."""
        tiles_data = []
        for tile_idx, tile in enumerate(self.logic.tiles):
            seeds = []
            for seed_idx, seed_pos in enumerate(tile.seed_positions_ras):
                seed_local = tile.seed_positions_local[seed_idx]
                seeds.append({
                    "seed_index": seed_idx + 1,
                    "position_ras_mm": [float(seed_pos[0]), float(seed_pos[1]), float(seed_pos[2])],
                    "position_local_mm": [float(seed_local[0]), float(seed_local[1]), float(seed_local[2])]
                })
            tiles_data.append({
                "tile_index": tile_idx,
                "center_ras_mm": [float(tile.center[0]), float(tile.center[1]), float(tile.center[2])],
                "normal_ras": [float(tile.normal[0]), float(tile.normal[1]), float(tile.normal[2])],
                "seed_axis_ras": [float(tile.seed_axis_ras[0]), float(tile.seed_axis_ras[1]), float(tile.seed_axis_ras[2])],
                "seeds": seeds
            })
        standalone_seeds_data = []
        for seed_idx, seed in enumerate(self.logic.seeds):
            c, n = seed.center, seed.normal
            standalone_seeds_data.append({
                "seed_index": seed_idx,
                "center_ras_mm": [float(c[0]), float(c[1]), float(c[2])],
                "normal_ras": [float(n[0]), float(n[1]), float(n[2])]
            })
        isodose_data = []
        for name, level, b in self._isodoseSurfaceBounds():
            isodose_data.append({
                "name": name,
                "dose_level_cGy": float(level) if level else None,
                "bounds_ras_mm": {
                    "x": [float(b[0]), float(b[1])],
                    "y": [float(b[2]), float(b[3])],
                    "z": [float(b[4]), float(b[5])]
                }
            })
        with open(path, "w") as f:
            json.dump({
                "tiles": tiles_data,
                "standalone_seeds": standalone_seeds_data,
                "dosimetry_dataset": self.logic.interpolator.dataset_name,
                "isodose_surface_bounds": isodose_data
            }, f, indent=2)
    
    def onExportDoseVolume(self):
        doseNode = self.logic.dose_volume_node
        if not doseNode or not doseNode.GetScene():
            slicer.util.errorDisplay("Compute dose first. Click 'Update Show Isodoses'.")
            return
        out_dir = qt.QFileDialog.getExistingDirectory(self.parent, "Choose output directory for dose volume")
        if not out_dir or not str(out_dir).strip():
            return
        out_dir = str(out_dir).strip()
        
        success = False
        doseNodeGy = self._createDoseVolumeInGy(doseNode)
        exportNode = doseNodeGy if doseNodeGy else doseNode

        if hasattr(slicer.modules, 'dicomrtimportexport') and hasattr(slicer.modules, 'dicomPlugins'):
            try:
                rtExportLogic = slicer.modules.dicomrtimportexport.logic()
                shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                if shNode and rtExportLogic:
                    itemID = shNode.GetItemByDataNode(exportNode)
                    if itemID:
                        plugin = slicer.modules.dicomPlugins.get('DicomRtImportExportPlugin')
                        if plugin:
                            exportables = plugin.examineForExport(itemID)
                            if exportables:
                                for exp in exportables:
                                    if hasattr(exp, 'setTag'):
                                        exp.setTag('SeriesDescription', 'GammaTile dose')
                                exportablesCollection = vtk.vtkCollection()
                                for exp in exportables:
                                    vtkExp = slicer.vtkSlicerDICOMExportable()
                                    exp.copyToVtkExportable(vtkExp)
                                    exportablesCollection.AddItem(vtkExp)
                                message = rtExportLogic.ExportDicomRTStudy(exportablesCollection)
                                if message is None or (isinstance(message, str) and 'failed' not in message.lower()):
                                    success = True
                                    self.statusLabel.setText(f"Exported DICOM RTDOSE to {os.path.basename(out_dir)}")
            except Exception as e:
                logger.warning("SlicerRT export failed, falling back to CreateDICOMSeries: %s", e)

        if not success and hasattr(slicer.modules, 'createdicomseries'):
            # Always write the cGy volume here: integer cGy is exact to 0.01 Gy, whereas an
            # integer Gy volume would be truncated to whole gray.
            success = self._exportDoseVolumeAsDICOM(doseNode, out_dir, "DOSE", in_gy=False)
            if success:
                self.statusLabel.setText(f"Exported dose volume (DICOM series, cGy) to {os.path.basename(out_dir)}")

        if not success:
            try:
                dose_path = os.path.join(out_dir, "GammaTile_dose.nrrd")
                slicer.util.saveNode(exportNode, dose_path)
                success = True
                self.statusLabel.setText(f"Exported dose volume (NRRD) to {os.path.basename(out_dir)}")
            except Exception as e:
                logger.exception("NRRD export failed: %s", e)
                slicer.util.errorDisplay(f"Export failed: {e}")
        
        if not success:
            self.statusLabel.setText("Dose volume export failed")
        
        if doseNodeGy and doseNodeGy.GetScene():
            slicer.mrmlScene.RemoveNode(doseNodeGy)
    
    def _createDoseVolumeInGy(self, doseNodeCgy):
        """Create a temporary copy of the dose volume with values in Gy (divide by 100). Returns node or None."""
        try:
            imageData = doseNodeCgy.GetImageData()
            if not imageData:
                return None
            dims = imageData.GetDimensions()
            scalars = imageData.GetPointData().GetScalars()
            if not scalars:
                return None
            arr = numpy_support.vtk_to_numpy(scalars).copy()
            arr = arr.astype(np.float32) / 100.0  # cGy -> Gy
            arr_vtk = numpy_support.numpy_to_vtk(np.ascontiguousarray(arr.ravel(order='F')), deep=True, array_type=vtk.VTK_FLOAT)
            newImage = vtk.vtkImageData()
            newImage.SetDimensions(dims)
            newImage.SetSpacing(imageData.GetSpacing())
            newImage.SetOrigin(imageData.GetOrigin())
            newImage.GetPointData().SetScalars(arr_vtk)
            doseNodeGy = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Dose Volume Gy")
            doseNodeGy.SetAndObserveImageData(newImage)
            ijkToRas = vtk.vtkMatrix4x4()
            doseNodeCgy.GetIJKToRASMatrix(ijkToRas)
            doseNodeGy.SetIJKToRASMatrix(ijkToRas)
            doseNodeGy.SetAttribute("DicomRtImport.DoseVolume", "1")
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            if shNode:
                itemID = shNode.GetItemByDataNode(doseNodeGy)
                if itemID:
                    shNode.SetItemAttribute(itemID, "DoseUnitName", "Gy")
                    shNode.SetItemAttribute(itemID, "DoseUnitValue", "1.0")
            return doseNodeGy
        except Exception as e:
            logger.warning("Could not create Gy-scaled dose volume: %s", e)
            return None
    
    def _exportDoseVolumeAsDICOM(self, volumeNode, directory, filename_prefix, in_gy=False):
        """Write the dose volume as a DICOM Secondary Capture series via the CreateDICOMSeries CLI.
        Returns True on success.

        Pixel values are stored as 32-bit integers with RescaleSlope 1 in the volume's own
        units (cGy unless in_gy is True), so the stored numbers are exact to 1 cGy. The CLI
        in Slicer 5.x divides pixel data by any rescaleSlope it is given but does not write
        the tag back for these modalities, so a non-unit slope must not be used here."""
        try:
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H%M%S")
            uid_root = "1.2.840.113619.6."
            tags = {
                "Patient Name": "GammaTile",
                "Patient ID": "GT001",
                "Patient Birth Date": "",
                "Patient Sex": "",
                "Patient Comments": "",
                "Study ID": "1",
                "Study Date": date_str,
                "Study Time": time_str,
                "Study Description": "GammaTile",
                "Modality": "SC",  # CreateDICOMSeries only accepts CT|MR|NM|US|PT|CR|SC
                "Manufacturer": "GammaTile",
                "Model": "Export",
                "Series Description": "GammaTile dose (Gy)" if in_gy else "GammaTile dose (cGy)",
                "Series Number": "1",
                "Series Date": date_str,
                "Series Time": time_str,
                "Content Date": date_str,
                "Content Time": time_str,
                "Study Instance UID": uid_root + str(uuid.uuid4().int % 10**10),
                "Series Instance UID": uid_root + str(uuid.uuid4().int % 10**10),
                "Frame of Reference UID": uid_root + str(uuid.uuid4().int % 10**10),
            }
            displayNode = volumeNode.GetDisplayNode()
            if displayNode and displayNode.IsA("vtkMRMLScalarVolumeDisplayNode"):
                window_center = str(displayNode.GetLevel())
                window_width = str(displayNode.GetWindow())
            else:
                window_center = "0"
                window_width = "1"
            cliparameters = {
                "patientName": tags["Patient Name"],
                "patientID": tags["Patient ID"],
                "patientBirthDate": tags["Patient Birth Date"],
                "patientSex": tags["Patient Sex"] if tags.get("Patient Sex") in ("M", "F", "O") else "[unknown]",
                "patientComments": tags.get("Patient Comments", ""),
                "studyID": tags["Study ID"],
                "studyDate": tags["Study Date"],
                "studyTime": tags["Study Time"],
                "studyDescription": tags["Study Description"],
                "modality": tags["Modality"],
                "manufacturer": tags["Manufacturer"],
                "model": tags["Model"],
                "seriesDescription": tags["Series Description"],
                "seriesNumber": tags["Series Number"],
                "seriesDate": tags["Series Date"],
                "seriesTime": tags["Series Time"],
                "windowCenter": window_center,
                "windowWidth": window_width,
                "contentDate": tags["Content Date"],
                "contentTime": tags["Content Time"],
                "rescaleSlope": "1.0",
                "rescaleIntercept": "0",
                # 32-bit integer pixels: cGy values near the seeds exceed the 16-bit range.
                "Type": "Int",
                "studyInstanceUID": tags["Study Instance UID"],
                "seriesInstanceUID": tags["Series Instance UID"],
                "frameOfReferenceUID": tags.get("Frame of Reference UID", ""),
                "inputVolume": volumeNode.GetID(),
                "dicomDirectory": directory,
                "dicomPrefix": filename_prefix,
            }
            dicomWrite = slicer.modules.createdicomseries
            cliNode = slicer.cli.run(dicomWrite, None, cliparameters, wait_for_completion=True)
            success = cliNode.GetStatus() == cliNode.Completed
            slicer.mrmlScene.RemoveNode(cliNode)
            return success
        except Exception as e:
            logger.exception("CreateDICOMSeries failed: %s", e)
            slicer.util.errorDisplay(f"DICOM export failed: {e}")
            return False
    
    def _getVolumeRASBounds(self, volumeNode):
        imageData = volumeNode.GetImageData()
        dims = imageData.GetDimensions()
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        
        ras_min = [float('inf')] * 3
        ras_max = [float('-inf')] * 3
        for i in [0, dims[0]-1]:
            for j in [0, dims[1]-1]:
                for k in [0, dims[2]-1]:
                    pt = [i, j, k, 1.0]
                    ras = [0.0, 0.0, 0.0, 0.0]
                    ijkToRas.MultiplyPoint(pt, ras)
                    for axis in range(3):
                        ras_min[axis] = min(ras_min[axis], ras[axis])
                        ras_max[axis] = max(ras_max[axis], ras[axis])
        
        return [[ras_min[0], ras_max[0]], [ras_min[1], ras_max[1]], [ras_min[2], ras_max[2]]]
    
    def onUpdate(self):
        if not self.logic.tiles and not self.logic.seeds:
            slicer.util.warningDisplay("No tiles or seeds placed")
            return
        
        vol = self.volumeSelector.currentNode()
        if not vol:
            slicer.util.errorDisplay("Select volume first")
            return
        
        imageData = vol.GetImageData()
        if not imageData:
            slicer.util.errorDisplay("Volume has no image data")
            return
        
        bounds_ras = self._getVolumeRASBounds(vol)
        
        self.statusLabel.setText("Computing dose volume...")
        slicer.app.processEvents()

        dose_array, spacing, dims, bounds_computed = self.logic.compute_dose_volume_optimized(
            bounds_ras, spacing_mm=2.0, max_distance_mm=50.0
        )
        
        if np.max(dose_array) == 0:
            slicer.util.warningDisplay("No dose computed. Check tile/seed placements.")
            self.statusLabel.setText("No dose computed")
            return

        print("\n" + "="*60)
        print("SEED COORDINATES LOG")
        print("="*60)
        print(f"Total tiles: {len(self.logic.tiles)}")
        print(f"Total tile seeds: {len(self.logic.tiles) * 4}")
        print(f"Standalone seeds: {len(self.logic.seeds)}")
        print("\nTile seed positions (RAS coordinates in mm):")
        print("-" * 60)
        for tile_idx, tile in enumerate(self.logic.tiles):
            print(f"\nTile {tile_idx + 1}:")
            print(f"  Tile center (RAS): [{tile.center[0]:.3f}, {tile.center[1]:.3f}, {tile.center[2]:.3f}]")
            print(f"  Tile normal (RAS): [{tile.normal[0]:.3f}, {tile.normal[1]:.3f}, {tile.normal[2]:.3f}]")
            print(f"  Seed axis (RAS): [{tile.seed_axis_ras[0]:.3f}, {tile.seed_axis_ras[1]:.3f}, {tile.seed_axis_ras[2]:.3f}]")
            for seed_idx, seed_pos in enumerate(tile.seed_positions_ras):
                seed_local = tile.seed_positions_local[seed_idx]
                print(f"  Seed {seed_idx + 1}:")
                print(f"    Local coords: [{seed_local[0]:.3f}, {seed_local[1]:.3f}, {seed_local[2]:.3f}] mm")
                print(f"    RAS coords:   [{seed_pos[0]:.3f}, {seed_pos[1]:.3f}, {seed_pos[2]:.3f}] mm")
        if self.logic.seeds:
            print("\nStandalone seeds (RAS coordinates in mm):")
            for seed_idx, seed in enumerate(self.logic.seeds):
                c, n = seed.center, seed.normal
                print(f"  Seed {seed_idx + 1}: center [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}], normal [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
        print("="*60 + "\n")

        dose_voxel_array = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(dose_array).ravel(order='F'), 
            deep=True, 
            array_type=vtk.VTK_FLOAT
        )

        image = vtk.vtkImageData()
        image.SetDimensions(dims)
        image.SetSpacing(1.0, 1.0, 1.0)
        image.SetOrigin(0.0, 0.0, 0.0)
        image.GetPointData().SetScalars(dose_voxel_array)

        if self.logic.dose_volume_node and self.logic.dose_volume_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.logic.dose_volume_node)
        if self.logic.isodose_nodes_3d:
            for node in self.logic.isodose_nodes_3d:
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
            self.logic.isodose_nodes_3d = []
        doseNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Dose Volume")
        self.logic.dose_volume_node = doseNode

        ijkToRas = vtk.vtkMatrix4x4()
        ijkToRas.SetElement(0, 0, spacing)
        ijkToRas.SetElement(1, 1, spacing)
        ijkToRas.SetElement(2, 2, spacing)
        ijkToRas.SetElement(0, 3, bounds_computed[0][0])
        ijkToRas.SetElement(1, 3, bounds_computed[1][0])
        ijkToRas.SetElement(2, 3, bounds_computed[2][0])
        ijkToRas.SetElement(3, 3, 1.0)

        doseNode.SetIJKToRASMatrix(ijkToRas)
        doseNode.SetAndObserveImageData(image)

        doseNode.SetAttribute("DicomRtImport.DoseVolume", "1")
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if shNode:
            itemID = shNode.GetItemByDataNode(doseNode)
            if itemID:
                shNode.SetItemAttribute(itemID, "DoseUnitName", "Gy")
                shNode.SetItemAttribute(itemID, "DoseUnitValue", "0.01")  # stored cGy * 0.01 = Gy
                shNode.RequestOwnerPluginSearch(itemID)
                shNode.ItemModified(itemID)

        prescription_cgy = self.logic.prescription_gy * 100
        max_dose = np.max(dose_array)

        doses_3d = [
            (prescription_cgy, (1, 0, 0), "100%"),
            (0.9 * prescription_cgy, (1, 0.65, 0), "90%"),
            (0.5 * prescription_cgy, (1, 1, 0), "50%")
        ]
        
        origin = [bounds_computed[0][0], bounds_computed[1][0], bounds_computed[2][0]]

        self.printDosageSummary(prescription_cgy, max_dose, len(self.logic.tiles), dims, spacing, bounds_computed, doses_3d, num_seeds=len(self.logic.seeds))
        
        self.statusLabel.setText("Creating isodose surfaces...")
        slicer.app.processEvents()

        created_surfaces_3d = 0
        for dose_level, color, label in doses_3d:
            if dose_level <= max_dose:
                try:
                    surface = IsodoseSurfaceGenerator.create_isodose_surface(
                        dose_array, dose_level, spacing, origin
                    )
                    if surface:
                        node = IsodoseSurfaceGenerator.add_isodose_to_scene(
                            surface, dose_level, color, f"Isodose3D_{label}"
                        )
                        if node:
                            self.logic.isodose_nodes_3d.append(node)
                            created_surfaces_3d += 1
                except Exception as e:
                    logger.error(f"Error creating 3D isodose {label}: {e}")
        
        # Update legend with 3D isodose levels
        self.updateDoseLegend(doses_3d, max_dose)
        
        # Force render update for all slice views
        try:
            layoutManager = slicer.app.layoutManager()
            for sliceViewName in ['Red', 'Yellow', 'Green']:
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                if sliceWidget:
                    sliceView = sliceWidget.sliceView()
                    if sliceView:
                        sliceView.scheduleRender()
        except Exception as e:
            logger.warning(f"Could not force slice view render: {e}")
        
        self.updateTileCounter()
        self.statusLabel.setText(
            f"Tiles: {len(self.logic.tiles)}, Seeds: {len(self.logic.seeds)}, Max dose: {max_dose:.1f} cGy, "
            f"3D Isodoses: {created_surfaces_3d}"
        )


class GammaTileLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()


class GammaTile(GammaTileModule):
    """Required entry-point class so Slicer can instantiate the module."""
    def __init__(self, parent):
        super().__init__(parent)
