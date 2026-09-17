"""
TG-43 QA check of the GammaView dose engine against the published dose-rate table.

The engine takes its consensus data (Lambda, L, g_L(r), F(r,theta)) from
TG-43U1S2, Rivard et al., Med. Phys. 44(9):e297-e338 (2017). Table XII of that
report gives the dose rate per unit source strength, D(r,theta)/S_K in
cGy h^-1 U^-1, for the IsoRay CS-1 Rev2 source and is intended for treatment
planning system commissioning QA. The originally published Table XII was flawed
and was corrected in the erratum, Rivard et al., Med. Phys. 45(2):971-974 (2018),
doi:10.1002/mp.12728. The values below are the corrected ones.

This script evaluates the engine at every tabulated (r, theta), removes the
module's decay-integration and source-strength factor to recover cGy h^-1 U^-1,
and writes a CSV with published, computed and percent difference per point.

Run with plain Python (no Slicer needed):

    python3 tests/tg43_qa_check.py [--dataset TG43U1S2|CLRP_V2] [-o results.csv]

Exit code is 0 when every point agrees within TOLERANCE_PERCENT, 1 otherwise.
"""
import argparse
import csv
import importlib.util
import os
import sys
import types

import numpy as np

TOLERANCE_PERCENT = 2.0  # published values carry 3-4 significant figures

# Erratum Table XII: dose rate (cGy h^-1 U^-1) per unit source strength, IsoRay CS-1 Rev2.
# Columns are r (cm); rows are polar angle theta (deg). Entries at r = 0.1 cm are
# not tabulated for theta <= 20 deg (None).
TABLE_XII_RADII_CM = [0.1, 0.25, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
TABLE_XII = {
    0:  [None,  29.3,  4.25, 2.04,  0.942, 0.205,  0.0778, 0.01874, 0.00602, 0.001321],
    2:  [None,  31.3,  4.08, 1.933, 0.911, 0.206,  0.0782, 0.01881, 0.00596, 0.001343],
    5:  [None,  32.8,  4.00, 1.824, 0.836, 0.1876, 0.0724, 0.01777, 0.00572, 0.001299],
    7:  [None,  31.5,  3.67, 1.689, 0.794, 0.1831, 0.0714, 0.01765, 0.00570, 0.001294],
    10: [None,  28.3,  3.45, 1.648, 0.789, 0.1846, 0.0720, 0.01781, 0.00575, 0.001310],
    15: [None,  26.6,  3.62, 1.745, 0.836, 0.1940, 0.0752, 0.01848, 0.00593, 0.001344],
    20: [None,  25.8,  3.86, 1.861, 0.889, 0.204,  0.0786, 0.01915, 0.00613, 0.001382],
    25: [185.4, 24.0,  4.04, 1.955, 0.932, 0.212,  0.0813, 0.01969, 0.00629, 0.001421],
    30: [145.9, 22.2,  4.15, 2.02,  0.965, 0.219,  0.0836, 0.0202,  0.00643, 0.001452],
    40: [100.2, 19.29, 4.25, 2.10,  1.009, 0.228,  0.0870, 0.0209,  0.00666, 0.001497],
    50: [79.3,  17.24, 4.25, 2.14,  1.034, 0.234,  0.0892, 0.0214,  0.00682, 0.001532],
    60: [67.9,  15.85, 4.20, 2.15,  1.048, 0.238,  0.0907, 0.0218,  0.00692, 0.001558],
    70: [61.3,  14.95, 4.14, 2.15,  1.054, 0.241,  0.0917, 0.0220,  0.00700, 0.001572],
    80: [57.9,  14.45, 4.11, 2.15,  1.056, 0.242,  0.0922, 0.0222,  0.00703, 0.001582],
    90: [56.9,  14.28, 4.10, 2.14,  1.056, 0.242,  0.0923, 0.0222,  0.00705, 0.001585],
}


def load_module(path):
    """Import GammaTile.py outside Slicer by stubbing the Slicer, Qt and CTK modules."""
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    for name in ("slicer", "slicer.util", "qt", "ctk"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = lambda n: _Any()
            sys.modules[name] = mod
    if "slicer.ScriptedLoadableModule" not in sys.modules:
        slm = types.ModuleType("slicer.ScriptedLoadableModule")
        for cls in ("ScriptedLoadableModule", "ScriptedLoadableModuleWidget", "ScriptedLoadableModuleLogic"):
            setattr(slm, cls, type(cls, (), {"__init__": lambda self, *a, **k: None, "setup": lambda self: None}))
        sys.modules["slicer.ScriptedLoadableModule"] = slm
        sys.modules["slicer"].ScriptedLoadableModule = slm
    spec = importlib.util.spec_from_file_location("GammaTile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default=os.path.join(here, "..", "GammaTile.py"), help="path to GammaTile.py")
    parser.add_argument("--dataset", default=None, help="TG43U1S2 (default) or CLRP_V2")
    parser.add_argument("-o", "--output", default=None, help="CSV output path (default: tests/results/tg43_qa_<dataset>.csv)")
    args = parser.parse_args()

    gt = load_module(os.path.abspath(args.module))
    interp = gt.DoseInterpolator(dataset=args.dataset)
    out = args.output or os.path.join(here, "results", f"tg43_qa_{interp.dataset}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    rows = []
    for theta, values in TABLE_XII.items():
        for r, published in zip(TABLE_XII_RADII_CM, values):
            if published is None:
                continue
            dose_cgy = interp.dose_at_point(float(theta), float(r))
            computed = dose_cgy / interp.DOSE_CONVERSION_FACTOR  # -> cGy h^-1 U^-1
            diff_pct = 100.0 * (computed - published) / published
            rows.append((r, theta, published, computed, computed - published, diff_pct))

    diffs = np.array([row[5] for row in rows])
    n_fail = int(np.sum(np.abs(diffs) > TOLERANCE_PERCENT))
    worst = rows[int(np.argmax(np.abs(diffs)))]

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# GammaView TG-43 QA check: engine dose rate per unit source strength vs TG-43U1S2 erratum Table XII (Rivard et al. 2018, doi:10.1002/mp.12728)"])
        w.writerow(["# dosimetry_dataset", interp.dataset_name])
        w.writerow(["# points", len(rows), "mean_abs_diff_percent", f"{np.mean(np.abs(diffs)):.3f}",
                    "max_abs_diff_percent", f"{np.max(np.abs(diffs)):.3f}", "tolerance_percent", TOLERANCE_PERCENT,
                    "points_outside_tolerance", n_fail])
        w.writerow(["r_cm", "theta_deg", "published_cGy_per_h_per_U", "computed_cGy_per_h_per_U",
                    "difference_cGy_per_h_per_U", "difference_percent", "within_tolerance"])
        for r, theta, published, computed, diff, diff_pct in rows:
            w.writerow([r, theta, published, f"{computed:.6g}", f"{diff:+.4g}", f"{diff_pct:+.3f}",
                        "yes" if abs(diff_pct) <= TOLERANCE_PERCENT else "no"])

    print(f"dataset: {interp.dataset_name}")
    print(f"points compared: {len(rows)}")
    print(f"mean |diff|: {np.mean(np.abs(diffs)):.3f}%   max |diff|: {np.max(np.abs(diffs)):.3f}% "
          f"at r={worst[0]} cm, theta={worst[1]} deg (published {worst[2]}, computed {worst[3]:.5g})")
    print(f"points outside +/-{TOLERANCE_PERCENT}%: {n_fail}")
    print(f"wrote {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
