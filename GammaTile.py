"""
GammaView Dosimetry - 3D Slicer Module
Works with DICOM brain volumes for intraoperative GammaTile placement
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
    """TG-43U1 2D formalism for IsoRay Proxcelan CS-1 Rev2 (L=0.40 cm)."""

    DOSE_CONVERSION_FACTOR = 335.23 * 3.5  # cGy per air kerma strength; math was derived
    _LAMBDA = 1.056 #Dose rate constant (cGy h^-1 U^-1)
    _L = 0.40 # Active source length (cm)
    # g_L(r) = (a0*r^-2 + a1*r^-1 + a2 + a3*r + a4*r^2 + a5*r^3) * exp(-a6*r)
    _GR_COEFFS = np.array([7.38e-04, -1.198e-02, 9.991e-01, 4.979e-01,
                           1.07e-02, 1.39e-03, 4.055e-01])
    # Anisotropy table F(r, theta) — V2 (2019) Proxcelan CS-1 Rev2
    _F_THETAS = np.array([
        0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 73, 75, 78, 80, 82, 84, 85, 86, 87, 88, 89, 90
    ], dtype=np.float64)

    _F_RADII = np.array([
        0.10, 0.15, 0.25, 0.50, 0.75, 1.00, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00
    ], dtype=np.float64)
    _F_TABLE = np.array([
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

    def __init__(self):
        self.angles = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90])
        self.radii_cm = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0])
        self.dose_table = np.array([
            [185.4, 29.3, 4.25, 2.04, 0.942, 0.205, 0.0778, 0.01874, 0.00602],
            [185.4, 31.3, 4.08, 1.935, 0.911, 0.206, 0.0782, 0.01881, 0.00596],
            [185.4, 32.8, 4.00, 1.826, 0.836, 0.1876, 0.0724, 0.01777, 0.00572],
            [185.4, 31.5, 3.67, 1.692, 0.794, 0.1831, 0.0714, 0.01765, 0.00570],
            [185.4, 28.3, 3.45, 1.650, 0.789, 0.1846, 0.0720, 0.01781, 0.00575],
            [185.4, 26.6, 3.62, 1.747, 0.836, 0.1940, 0.0752, 0.01848, 0.00593],
            [185.4, 25.8, 3.86, 1.864, 0.889, 0.204, 0.0786, 0.01915, 0.00613],
            [185.4, 24.0, 4.04, 1.957, 0.932, 0.212, 0.0813, 0.01969, 0.00629],
            [145.9, 22.2, 4.15, 2.02, 0.965, 0.219, 0.0836, 0.0202, 0.00643],
            [100.2, 19.29, 4.25, 2.11, 1.009, 0.228, 0.0870, 0.0209, 0.00666],
            [79.3, 17.24, 4.25, 2.14, 1.034, 0.234, 0.0892, 0.0214, 0.00682],
            [67.9, 15.85, 4.20, 2.15, 1.048, 0.238, 0.0907, 0.0218, 0.00692],
            [61.3, 14.95, 4.14, 2.15, 1.055, 0.241, 0.0917, 0.0220, 0.00700]
        ], dtype=np.float32)
        # Precompute reference geometry factor G_L(r0=1.0, theta0=90°)
        self._GL_ref = self._geometry_factor(1.0, 90.0)

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

    def _radial_dose(self, r):
        # g_L(r) — Eq. 3 from TG-43U1, polynomial fit from CLRP v2
        r = float(np.clip(r, 0.05, 10.0))
        a = self._GR_COEFFS
        return (a[0] * r**-2 + a[1] * r**-1 + a[2] + a[3] * r
                + a[4] * r**2 + a[5] * r**3) * np.exp(-a[6] * r)

    def _anisotropy(self, r, theta_deg):
        """Anisotropy function F(r, theta) via bilinear interpolation on hardcoded table."""
        thetas = self._F_THETAS
        radii = self._F_RADII
        table = self._F_TABLE

        theta_c = float(np.clip(theta_deg, thetas[0], thetas[-1]))
        r_c = float(np.clip(r, radii[0], radii[-1]))
        it = int(np.searchsorted(thetas, theta_c, side='right')) - 1
        it = max(0, min(it, len(thetas) - 2))
        it2 = it + 1
        ir = int(np.searchsorted(radii, r_c, side='right')) - 1
        ir = max(0, min(ir, len(radii) - 2))
        ir2 = ir + 1

        def _val(ti, ri):
            v = table[ti, ri]
            if np.isnan(v):
                for k in range(ri + 1, len(radii)):
                    if not np.isnan(table[ti, k]):
                        return table[ti, k]
                for k in range(ri - 1, -1, -1):
                    if not np.isnan(table[ti, k]):
                        return table[ti, k]
                return 1.0
            return v

        v00 = _val(it, ir)
        v01 = _val(it, ir2)
        v10 = _val(it2, ir)
        v11 = _val(it2, ir2)

        # Interpolation fractions
        dr = radii[ir2] - radii[ir]
        t_r = (r_c - radii[ir]) / dr if dr > 1e-12 else 0.0

        dt = thetas[it2] - thetas[it]
        t_t = (theta_c - thetas[it]) / dt if dt > 1e-12 else 0.0

        # Bilinear
        top = v00 * (1.0 - t_r) + v01 * t_r
        bot = v10 * (1.0 - t_r) + v11 * t_r
        return top * (1.0 - t_t) + bot * t_t


    def dose_at_point(self, angle_deg, radius_cm):
        """TG-43U1 dose at a point, returns cGy."""
        angle_deg = angle_deg % 360
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        angle_deg = float(np.clip(angle_deg, 0, 90))
        radius_cm_eff = float(np.clip(radius_cm, 0.041, 10.0))

        rate = self._tg43_dose_rate(angle_deg, radius_cm_eff)
        return self.DOSE_CONVERSION_FACTOR * max(0.0, rate)

    def raw_dose_rate(self, angle_deg, radius_cm):
        """Return dose rate in cGy/h/U (no conversion factor) for TG-43 table validation."""
        angle_deg = angle_deg % 360
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        angle_deg = float(np.clip(angle_deg, 0, 90))
        radius_cm_eff = float(np.clip(radius_cm, 0.041, 10.0))
        return max(0.0, self._tg43_dose_rate(angle_deg, radius_cm_eff))

    def _tg43_dose_rate(self, theta_deg, r):
        """Core TG-43U1 2D calculation: Lambda * [G_L(r,theta)/G_L(r0,theta0)] * g_L(r) * F(r,theta)."""
        GL = self._geometry_factor(r, theta_deg)
        gL = self._radial_dose(r)
        F = self._anisotropy(r, theta_deg)
        return self._LAMBDA * (GL / self._GL_ref) * gL * F


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
    
    def dose_at_point(self, point_ras, interpolator):
        """Total dose from all 4 seeds, TG-43 polar angle from seed axis."""
        total_dose = 0.0
        point = np.array(point_ras)
        
        seed_axis = self.seed_axis_ras
        
        for seed_pos in self.seed_positions_ras:
            # Vector from seed center to calculation point
            relative = point - seed_pos
            r_total = np.linalg.norm(relative)

            if r_total < self.SEED_DIAMETER_MM / 2.0 or r_total > 100:
                continue
            r_parallel = np.dot(relative, seed_axis)
            r_perp_vec = relative - r_parallel * seed_axis
            r_perp = np.linalg.norm(r_perp_vec)
            theta_rad = np.arctan2(r_perp, abs(r_parallel))
            theta_deg = np.degrees(theta_rad)
            r_cm = r_total / 10.0
            dose = interpolator.dose_at_point(theta_deg, r_cm)
            total_dose += dose
        
        return total_dose


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

    def dose_at_point(self, point_ras, interpolator):
        """Dose from this single seed at point_ras (TG-43)."""
        point = np.array(point_ras)
        relative = point - self.center
        r_total = np.linalg.norm(relative)
        if r_total < GammaTilePlacement.SEED_DIAMETER_MM / 2.0 or r_total > 100:
            return 0.0
        r_parallel = np.dot(relative, self.seed_axis_ras)
        r_perp_vec = relative - r_parallel * self.seed_axis_ras
        r_perp = np.linalg.norm(r_perp_vec)
        theta_rad = np.arctan2(r_perp, abs(r_parallel))
        theta_deg = np.degrees(theta_rad)
        r_cm = r_total / 10.0
        return interpolator.dose_at_point(theta_deg, r_cm)


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

        max_dist_sq = max_distance_mm ** 2

        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    ras = np.array([x_coords[i], y_coords[j], z_coords[k]])
                    min_dist_sq = float('inf')
                    for tile in self.tiles:
                        dist_sq = np.sum((ras - tile.center) ** 2)
                        min_dist_sq = min(min_dist_sq, dist_sq)
                    for seed in self.seeds:
                        dist_sq = np.sum((ras - seed.center) ** 2)
                        min_dist_sq = min(min_dist_sq, dist_sq)
                    if min_dist_sq <= max_dist_sq:
                        for tile in self.tiles:
                            volume[i, j, k] += tile.dose_at_point(ras, self.interpolator)
                        for seed in self.seeds:
                            volume[i, j, k] += seed.dose_at_point(ras, self.interpolator)

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

        marching = vtk.vtkMarchingCubes()
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
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        modelNode.SetAndObservePolyData(surface)
        
        # Apply RAS transform from volume
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{name}Transform")
        transformNode.SetMatrixTransformToParent(ijkToRas)
        modelNode.SetAndObserveTransformNodeID(transformNode.GetID())
        
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

        contour = vtk.vtkMarchingCubes()
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


# Reference dose rate (cGy/h/U) along tile normal for a single 4-seed GammaTile.
# Distance in mm from tile center; negative = opposite side of normal.
# Source: dose_along_normal-2.csv
_TILE_NORMAL_REF_DATA = [
    # (distance_mm, superposed_dose_rate_cGy_h_U) from Table XII
    (-100.0, 0.027908),
    (-70.0,  0.086255),
    (-50.0,  0.349537),
    (-30.0,  0.905364),
    (-20.0,  3.362893),
    (-10.0,  6.871183),
    (-5.0,   9.559224),
    (-2.5,  10.835737),
    (-1.0,  11.314545),
    (0.0,   11.413081),
    (1.0,   11.314545),
    (2.5,   10.835737),
    (5.0,    9.559224),
    (10.0,   6.871183),
    (20.0,   3.362893),
    (30.0,   0.905364),
    (50.0,   0.349537),
    (70.0,   0.086255),
    (100.0,  0.027908),
]


class GammaTileWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = GammaTileDosimetry()
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

        self.exportValidationButton = qt.QPushButton("Validation Error CSV")
        self.exportValidationButton.clicked.connect(self.onExportValidationCSV)
        exportFormLayout.addWidget(self.exportValidationButton)

        self.exportTileNormalValidationButton = qt.QPushButton("Tile Normal Superposition Validation CSV")
        self.exportTileNormalValidationButton.clicked.connect(self.onExportTileNormalValidationCSV)
        exportFormLayout.addWidget(self.exportTileNormalValidationButton)
        
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
            current_center = np.zeros(3)
            num_points = current_points.GetNumberOfPoints()
            for i in range(num_points):
                pt = current_points.GetPoint(i)
                current_center += np.array(pt)
            current_center /= num_points

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
        
        geometric_center = np.zeros(3)
        num_points = points.GetNumberOfPoints()
        for i in range(num_points):
            pt = points.GetPoint(i)
            geometric_center += np.array(pt)
        geometric_center /= num_points
        
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
        new_center = np.zeros(3)
        for i in range(num_points):
            pt = new_points.GetPoint(i)
            new_center += np.array(pt)
        new_center /= num_points

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
    
    def getNormalFromModel(self, modelNode, pos_ras, search_radius=5.0):
        try:
            polyData = modelNode.GetPolyData()
            if not polyData or polyData.GetNumberOfPoints() == 0:
                return None

            points = polyData.GetPoints()
            min_dist = float('inf')
            nearest_idx = -1
            
            for i in range(points.GetNumberOfPoints()):
                pt = points.GetPoint(i)
                dist = np.linalg.norm(np.array(pt) - pos_ras)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            if nearest_idx < 0 or min_dist > search_radius:
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
            
            neighbors = []
            for i in range(points.GetNumberOfPoints()):
                pt = np.array(points.GetPoint(i))
                if np.linalg.norm(pt - center) < radius:
                    neighbors.append(pt)
            
            if len(neighbors) < 3:
                return None

            neighbors = np.array(neighbors)
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
        
        # Fallback: use volume-based surface detection
        image = volumeNode.GetImageData()
        if not image:
            return np.array(pos_ras, dtype=np.float32)
        
        scalars = image.GetPointData().GetScalars()
        if not scalars:
            return np.array(pos_ras, dtype=np.float32)
        
        current_pos = np.array(pos_ras, dtype=np.float32)
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        rasToIjk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)
        
        dims = image.GetDimensions()
        max_steps = int(max_trace / step)
        
        # Get initial intensity to determine threshold
        pos_homog = [current_pos[0], current_pos[1], current_pos[2], 1.0]
        ijk = [0, 0, 0, 0]
        rasToIjk.MultiplyPoint(pos_homog, ijk)
        i, j, k = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))
        
        if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
            idx = i + j*dims[0] + k*dims[0]*dims[1]
            initial_intensity = scalars.GetValue(idx)
            threshold = max(30, initial_intensity * 0.4)  # Adaptive threshold
        else:
            threshold = 50
        
        # Trace inward (opposite to normal) to find brain surface
        surface_found = False
        for step_num in range(max_steps):
            pos_homog = [current_pos[0], current_pos[1], current_pos[2], 1.0]
            ijk = [0, 0, 0, 0]
            rasToIjk.MultiplyPoint(pos_homog, ijk)
            
            i, j, k = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))
            
            if not (0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]):
                break
            
            idx = i + j*dims[0] + k*dims[0]*dims[1]
            intensity = scalars.GetValue(idx)
            
            # Stop when entering cavity (low intensity) - found surface boundary
            if intensity < threshold:
                surface_found = True
                # Move back to surface boundary
                current_pos = current_pos + normal * (step * 0.5)
                break
            
            current_pos = current_pos - normal * step
        
        if surface_found:
            # Position tile center so bottom face sits on surface
            tile_thickness_half = GammaTilePlacement.TILE_THICKNESS_MM / 2.0
            tile_center = current_pos + normal * tile_thickness_half
            return tile_center
        else:
            # If surface not found, use original position but adjust for tile thickness
            tile_thickness_half = GammaTilePlacement.TILE_THICKNESS_MM / 2.0
            return pos_ras + normal * tile_thickness_half
    
    def findNearestPointOnSurface(self, modelNode, pos_ras, search_radius=10.0):
        try:
            polyData = modelNode.GetPolyData()
            if not polyData or polyData.GetNumberOfPoints() == 0:
                return None

            locator = vtk.vtkCellLocator()
            locator.SetDataSet(polyData)
            locator.BuildLocator()

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
            imageData = volumeNode.GetImageData()
            if not imageData:
                return intersections
            
            scalars = imageData.GetPointData().GetScalars()
            if not scalars:
                return intersections

            ijkToRas = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRas)
            rasToIjk = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)
            
            dims = imageData.GetDimensions()

            step_size = 1.0
            num_steps = int(max_distance / step_size)
            
            prev_intensity = None
            prev_point = None

            sample_points = []
            for i in range(min(10, num_steps)):
                t = i * step_size
                sample_ras = ray_origin + ray_direction * t
                sample_points.append(sample_ras)
            
            intensities = []
            for sample_ras in sample_points:
                ras_homog = [sample_ras[0], sample_ras[1], sample_ras[2], 1.0]
                ijk = [0, 0, 0, 0]
                rasToIjk.MultiplyPoint(ras_homog, ijk)
                i, j, k = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))
                if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
                    idx = i + j*dims[0] + k*dims[0]*dims[1]
                    intensities.append(scalars.GetValue(idx))
            
            if len(intensities) == 0:
                return intersections

            mean_intensity = np.mean(intensities)
            std_intensity = np.std(intensities) if len(intensities) > 1 else 1.0
            
            cavity_threshold = mean_intensity - 0.5 * std_intensity
            brain_threshold = mean_intensity + 0.3 * std_intensity
            bone_threshold = mean_intensity + 2.0 * std_intensity

            for step in range(num_steps):
                t = step * step_size
                point_ras = ray_origin + ray_direction * t

                ras_homog = [point_ras[0], point_ras[1], point_ras[2], 1.0]
                ijk = [0, 0, 0, 0]
                rasToIjk.MultiplyPoint(ras_homog, ijk)
                
                i, j, k = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))
                
                if not (0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]):
                    break
                
                idx = i + j*dims[0] + k*dims[0]*dims[1]
                intensity = scalars.GetValue(idx)

                if prev_intensity is not None:
                    if prev_intensity >= cavity_threshold and intensity < cavity_threshold:
                        intersections.append((t, prev_point.copy(), -ray_direction.copy(), 'cavity'))

                    elif prev_intensity < cavity_threshold and intensity >= brain_threshold:
                        intersections.append((t, point_ras.copy(), ray_direction.copy(), 'cavity'))

                    elif prev_intensity < brain_threshold and intensity >= brain_threshold:
                        intersections.append((t, point_ras.copy(), ray_direction.copy(), 'interior'))

                    elif prev_intensity >= brain_threshold and intensity < brain_threshold:
                        intersections.append((t, prev_point.copy(), -ray_direction.copy(), 'interior'))

                    elif intensity >= bone_threshold:
                        intersections.append((t, point_ras.copy(), ray_direction.copy(), 'barrier'))
                        break
                
                prev_intensity = intensity
                prev_point = point_ras.copy()
            
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
                        locator = vtk.vtkCellLocator()
                        locator.SetDataSet(surfacePolyData)
                        locator.BuildLocator()

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

            locator = vtk.vtkCellLocator()
            locator.SetDataSet(surfacePolyData)
            locator.BuildLocator()

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
                            pointLocator = vtk.vtkPointLocator()
                            pointLocator.SetDataSet(surfacePolyData)
                            pointLocator.BuildLocator()
                            
                            closest_point_id = pointLocator.FindClosestPoint(
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
    
    def printSeedValidationTables(self):
        """Print TG-43U1 computed dose rate vs Table XII reference with % error."""
        interp = self.logic.interpolator
        angles = interp.angles      # [0,5,10,...,90] deg
        radii  = interp.radii_cm    # [0.1,0.25,...,10] cm
        ref_table = interp.dose_table  # Table XII reference values

        # Cells where Table XII has no value (theta < 40 at r=0.1)
        BLANK_CELLS = {(t_idx, 0) for t_idx, theta in enumerate(angles) if theta < 40}

        sep = "=" * (16 + 9 * len(radii))
        r_header = "  ".join(f"{r:>7.2f}" for r in radii)

        def print_table(label):
            print(f"\n{sep}")
            print(f"  {label}")
            print(f"  Computed vs Table XII (cGy/h/U)")
            print(sep)

            # Computed values
            print(f"\n  Computed (TG-43U1 formula):")
            print(f"  {'theta':>7s}  " + r_header)
            print("-" * (16 + 9 * len(radii)))
            for t_idx, theta in enumerate(angles):
                row = "  ".join(
                    f"{'---':>7s}" if (t_idx, r_idx) in BLANK_CELLS
                    else f"{interp.raw_dose_rate(theta, r):>7.4f}"
                    for r_idx, r in enumerate(radii)
                )
                print(f"  {theta:>7.1f}  {row}")

            # Reference values
            print(f"\n  Reference (Table XII):")
            print(f"  {'theta':>7s}  " + r_header)
            print("-" * (16 + 9 * len(radii)))
            for t_idx, theta in enumerate(angles):
                row = "  ".join(
                    f"{'---':>7s}" if (t_idx, r_idx) in BLANK_CELLS
                    else f"{float(ref_table[t_idx, r_idx]):>7.4f}"
                    for r_idx, r in enumerate(radii)
                )
                print(f"  {theta:>7.1f}  {row}")

            # % Error
            print(f"\n  % Error (|computed - ref| / ref * 100):")
            print(f"  {'theta':>7s}  " + r_header)
            print("-" * (16 + 9 * len(radii)))
            for t_idx, theta in enumerate(angles):
                cells = []
                for r_idx, r in enumerate(radii):
                    if (t_idx, r_idx) in BLANK_CELLS:
                        cells.append(f"{'---':>7s}")
                    else:
                        code_val = interp.raw_dose_rate(theta, r)
                        ref_val = float(ref_table[t_idx, r_idx])
                        pct = abs(code_val - ref_val) / ref_val * 100 if ref_val != 0 else 0.0
                        cells.append(f"{pct:>6.2f}%")
                print(f"  {theta:>7.1f}  " + "  ".join(cells))
            print(sep)

        for tile_idx, tile in enumerate(self.logic.tiles):
            for s_idx in range(len(tile.seed_positions_ras)):
                print_table(f"Tile {tile_idx+1}, Seed {s_idx+1}  (RAS: {tile.seed_positions_ras[s_idx]})")
        for s_idx, seed in enumerate(self.logic.seeds):
            print_table(f"Standalone Seed {s_idx+1}  (RAS: {seed.center})")

    def printDosageSummary(self, prescription_cgy, max_dose, num_tiles, dims, spacing, bounds_computed, doses_3d, num_seeds=0):
        """Print a clean dosage summary to console."""
        print("\n" + "="*70)
        print("DOSAGE SUMMARY")
        print("="*70)
        print(f"Prescription dose: {prescription_cgy:.1f} cGy ({prescription_cgy/100:.1f} Gy)")
        print(f"Maximum computed dose: {max_dose:.1f} cGy")
        print(f"Number of tiles: {num_tiles}")
        print(f"Number of standalone seeds: {num_seeds}")
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
        if self.logic.tiles:
            self.printTileNormalValidation()

    def _compute_tile_normal_dose(self, tile):
        """Return list of (dist_mm, computed_cGy_h_U, ref_cGy_h_U) along tile normal."""
        interp = self.logic.interpolator
        results = []
        for dist_mm, ref_val in _TILE_NORMAL_REF_DATA:
            point = tile.center + dist_mm * tile.normal
            raw_dose = 0.0
            for seed_pos in tile.seed_positions_ras:
                relative = point - seed_pos
                r_total = np.linalg.norm(relative)
                if r_total < GammaTilePlacement.SEED_DIAMETER_MM / 2.0 or r_total > 100:
                    continue
                r_parallel = np.dot(relative, tile.seed_axis_ras)
                r_perp = np.linalg.norm(relative - r_parallel * tile.seed_axis_ras)
                theta_deg = np.degrees(np.arctan2(r_perp, abs(r_parallel)))
                r_cm = r_total / 10.0
                raw_dose += interp.raw_dose_rate(theta_deg, r_cm)
            results.append((dist_mm, raw_dose, ref_val))
        return results

    def printTileNormalValidation(self):
        """Print dose along normal for each tile vs reference data."""
        sep = "=" * 60
        for tile_idx, tile in enumerate(self.logic.tiles):
            results = self._compute_tile_normal_dose(tile)
            print(f"\n{sep}")
            print(f"  Tile {tile_idx+1} — Dose along normal (4-seed superposition)")
            print(f"  Units: cGy/h/U")
            print(sep)
            print(f"  {'dist(mm)':>9s}  {'computed':>11s}  {'reference':>11s}  {'%err':>8s}")
            print("-" * 48)
            for dist_mm, computed, ref in results:
                pct = abs(computed - ref) / ref * 100 if ref != 0 else 0.0
                print(f"  {dist_mm:>9.2f}  {computed:>11.5f}  {ref:>11.6f}  {pct:>7.3f}%")
            print(sep)

    def onExportTileNormalValidationCSV(self):
        if not self.logic.tiles:
            slicer.util.warningDisplay("No tiles placed.")
            return
        result = qt.QFileDialog.getSaveFileName(
            self.parent, "Save Tile Normal Validation CSV", "tile_normal_validation.csv", "CSV (*.csv)"
        )
        path = result[0] if isinstance(result, (tuple, list)) else result
        if not path or not str(path).strip():
            return
        path = str(path).strip()
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tile_index", "distance_mm", "computed_cGy_h_U", "ref_cGy_h_U", "percent_error"])
                for tile_idx, tile in enumerate(self.logic.tiles):
                    for dist_mm, computed, ref in self._compute_tile_normal_dose(tile):
                        pct = abs(computed - ref) / ref * 100 if ref != 0 else 0.0
                        w.writerow([tile_idx + 1, dist_mm, f"{computed:.5f}", f"{ref:.6f}", f"{pct:.3f}"])
            self.statusLabel.setText(f"Tile normal validation saved: {os.path.basename(path)}")
        except Exception as e:
            slicer.util.errorDisplay(f"Could not save CSV: {e}")

    def onExportValidationCSV(self):
        """Export TG-43 table validation: code value vs reference value vs % error at each grid point."""
        result = qt.QFileDialog.getSaveFileName(self.parent, "Save Validation Error CSV", "seed_validation.csv", "CSV (*.csv)")
        path = result[0] if isinstance(result, (tuple, list)) else result
        if not path or not str(path).strip():
            return
        path = str(path).strip()

        interp = self.logic.interpolator
        angles = interp.angles
        radii  = interp.radii_cm
        BLANK_CELLS = {(theta, 0.1) for theta in angles if theta < 40}

        rows = []
        for t_idx, theta in enumerate(angles):
            for r_idx, r in enumerate(radii):
                ref_val = float(interp.dose_table[t_idx, r_idx])
                if (theta, r) in BLANK_CELLS:
                    rows.append([theta, r, "", "", "N/A"])
                else:
                    code_val = interp.raw_dose_rate(theta, r)
                    pct_err = abs(code_val - ref_val) / ref_val * 100 if ref_val != 0 else 0.0
                    rows.append([theta, r, f"{code_val:.5f}", f"{ref_val:.5f}", f"{pct_err:.3f}"])

        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["theta_deg", "r_cm", "code_cGy_h_U", "ref_cGy_h_U", "percent_error"])
                w.writerows(rows)
            self.statusLabel.setText(f"Validation CSV saved: {os.path.basename(path)}")
        except Exception as e:
            slicer.util.errorDisplay(f"Could not save validation CSV: {e}")

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
    
    def _writePositionsCSV(self, path):
        """Write tile and seed positions to CSV (consistent with onUpdate log)."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
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
        with open(path, "w") as f:
            json.dump({"tiles": tiles_data, "standalone_seeds": standalone_seeds_data}, f, indent=2)
    
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
            success = self._exportDoseVolumeAsDICOM(exportNode, out_dir, "DOSE", in_gy=(doseNodeGy is not None))
            if success:
                self.statusLabel.setText(f"Exported dose volume (DICOM) to {os.path.basename(out_dir)}")

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
            doseNodeGy.SetIJKToRASMatrix(doseNodeCgy.GetIJKToRASMatrix())
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
        """Run CreateDICOMSeries CLI for dose volume. Returns True on success.
        in_gy: if True, volume values are in Gy; if False, in cGy (use rescale 0.01 for Gy output)."""
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
                "Modality": "OT",
                "Manufacturer": "GammaTile",
                "Model": "Export",
                "Series Description": "GammaTile dose",
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
                "rescaleSlope": "1.0" if in_gy else "0.01",  # 0.01: cGy->Gy; 1.0: already Gy
                "rescaleIntercept": "0",
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
