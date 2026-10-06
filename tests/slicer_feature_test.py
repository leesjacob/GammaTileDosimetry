"""
Headless end-to-end feature test for the GammaView Dosimetry Slicer module.

Run from a terminal (Slicer opens, runs every check on a synthetic brain volume, then exits):

    /Applications/Slicer.app/Contents/MacOS/Slicer --testing --no-splash \
        --python-script tests/slicer_feature_test.py

Environment variables:
    GAMMATILE_MODULE   path to GammaTile.py (default: ../GammaTile.py relative to this file,
                       or ./GammaTile.py in the current directory)
    GAMMATILE_TEST_OUT directory for exported files and results.json (default: a temp dir)

Exit code is 0 when every check passes, 1 otherwise. Each check prints a [TEST] PASS/FAIL line.
"""
import os as osmod, json, traceback, time, tempfile
import numpy as np
import slicer, qt, vtk

OUT = osmod.environ.get("GAMMATILE_TEST_OUT") or tempfile.mkdtemp(prefix="gammatile_test_")
osmod.makedirs(OUT, exist_ok=True)
MOD = osmod.environ.get("GAMMATILE_MODULE")
if not MOD:
    here = osmod.path.dirname(osmod.path.abspath(__file__)) if "__file__" in globals() else osmod.getcwd()
    for cand in (osmod.path.join(here, "..", "GammaTile.py"), osmod.path.join(osmod.getcwd(), "GammaTile.py")):
        if osmod.path.exists(cand):
            MOD = osmod.path.abspath(cand); break
print(f"[TEST] module={MOD} output={OUT}", flush=True)
RESULTS = {}
def record(name, ok, detail=""):
    RESULTS[name] = {"ok": bool(ok), "detail": str(detail)}
    print(f"[TEST] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
def finish(code):
    import os as _os, json as _json
    _json.dump(RESULTS, open(_os.path.join(OUT, "results.json"), "w"), indent=2)
    # Exit from the event loop so Slicer's script wrapper is not interrupted mid-script
    qt.QTimer.singleShot(0, lambda: slicer.app.exit(code))

def ras_to_xy(sliceViewName, ras):
    sn = slicer.app.layoutManager().sliceWidget(sliceViewName).sliceLogic().GetSliceNode()
    inv = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Invert(sn.GetXYToRAS(), inv)
    out = [0.0]*4; inv.MultiplyPoint([float(ras[0]), float(ras[1]), float(ras[2]), 1.0], out)
    return int(round(out[0])), int(round(out[1]))

try:
    # ---------- 1. Load module through Slicer's factory ----------
    fm = slicer.app.moduleManager().factoryManager()
    fm.registerModule(qt.QFileInfo(MOD))
    ok = fm.loadModules(["GammaTile"])
    record("module_load", ok and hasattr(slicer.modules, "gammatile"), f"loadModules={ok}")
    record("module_title", slicer.modules.gammatile.title == "GammaView Dosimetry", slicer.modules.gammatile.title)
    slicer.util.selectModule("GammaTile")
    slicer.app.processEvents()
    w = slicer.modules.gammatile.widgetRepresentation().self()
    record("widget_instance", type(w).__name__ == "GammaTileWidget" and hasattr(w, "logic"), type(w).__name__)
    record("logic_class", hasattr(slicer.modules, "GammaTileWidget") and type(slicer.modules.GammaTileWidget).__name__ == "GammaTileWidget", f"slicer.modules.GammaTileWidget={type(getattr(slicer.modules, 'GammaTileWidget', None)).__name__}")

    # ---------- 2. Synthetic brain volume: sphere r=70mm, cavity r=15 at (0,0,55) ----------
    n, sp = 96, 2.0
    c = (n - 1) / 2.0
    kk, jj, ii = np.indices((n, n, n)).astype(np.float64)
    x, y, z = (ii - c) * sp, (jj - c) * sp, (kk - c) * sp
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4, size=(n, n, n)).astype(np.int16)  # faint background noise so ray thresholds are well defined
    arr[np.sqrt(x**2 + y**2 + z**2) <= 70] = 120
    arr[np.sqrt(x**2 + y**2 + (z - 55)**2) <= 15] = 5
    ijkToRas = np.array([[sp,0,0,-c*sp],[0,sp,0,-c*sp],[0,0,sp,-c*sp],[0,0,0,1]])
    vol = slicer.util.addVolumeFromArray(arr, ijkToRAS=ijkToRas, name="SyntheticBrain")
    slicer.util.setSliceViewerLayers(background=vol, fit=True)
    w.volumeSelector.setCurrentNode(vol)
    slicer.app.processEvents()
    record("volume_selected", w.volumeSelector.currentNode() is vol, w.statusLabel.text)

    # ---------- 3. Brain surface ----------
    t0 = time.time(); w.onCreateBrainSurface(); dt = time.time() - t0
    bs = w.brain_surface_model
    npts = bs.GetPolyData().GetNumberOfPoints() if bs else 0
    record("brain_surface", bs is not None and npts > 1000 and w.logic.brain_surface_model is bs, f"{npts} pts in {dt:.2f}s; status='{w.statusLabel.text}'")
    # surface radius sanity: points should sit near r=70
    P = slicer.util.arrayFromModelPoints(bs)
    rr = np.linalg.norm(P, axis=1)
    outer = rr[rr > 40]
    record("brain_surface_geometry", abs(np.median(outer) - 70) < 2.5, f"median outer radius {np.median(outer):.2f} mm (mesh stored in RAS)")
    ntn = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLLinearTransformNode")
    record("brain_surface_no_transform_node", bs.GetParentTransformNode() is None, f"parent transform={bs.GetParentTransformNode()}, linear transform nodes in scene={ntn}")

    # ---------- 4. 3D-click placement path (normal from model -> settle -> penetration -> place -> model) ----------
    click = np.array([0.0, 0.0, -70.0], dtype=np.float32)  # bottom of the sphere, away from the cavity
    normal = w.getNormalFromModel(bs, click)
    record("normal_from_model", normal is not None and abs(abs(np.dot(normal / np.linalg.norm(normal), [0,0,-1])) - 1) < 0.1, f"normal={None if normal is None else np.round(normal,3)}")
    if normal is None: normal = np.array([0,0,-1], dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, click) < 0: normal = -normal  # outward
    settled = w.settleToSurface(vol, click, normal)
    adjusted, pen = w.checkAndLimitPenetration(vol, settled, normal)
    final_pos, _ = w.logic.place_tile(adjusted, normal, sp)
    tileModel = w.createTileModel(final_pos, normal, sp)
    if tileModel: w.logic.fiducial_nodes.append(tileModel)
    w.updateTileCounter()
    record("tile_place_3d_path", len(w.logic.tiles) == 1 and tileModel is not None and tileModel.GetScene() is not None,
           f"center={np.round(w.logic.tiles[0].center,2)} pen={pen:.2f} counter='{w.tileCounterLabel.text}'")
    record("tile_sits_on_surface", abs(np.linalg.norm(w.logic.tiles[0].center) - 70) < 4.0, f"|center|={np.linalg.norm(w.logic.tiles[0].center):.2f} (expect ~70+2)")
    t0_ = w.logic.tiles[0]
    depths = [float(np.dot(np.asarray(sp) - t0_.center, t0_.normal)) + 2.0 for sp in t0_.seed_positions_ras]  # distance above the contact face
    record("seed_plane_3mm_from_contact_face", all(abs(d - 3.0) < 1e-3 for d in depths) and all(abs(float(sl[2]) - 1.0) < 1e-6 for sl in t0_.seed_positions_local),
           f"seed depths above contact face = {np.round(depths, 3)} mm; local z = {[float(sl[2]) for sl in t0_.seed_positions_local]}")
    pd_ = tileModel.GetPolyData(); sides_ = pd_.GetCellData().GetArray("TileSide")
    nvals = set(int(sides_.GetTuple1(i)) for i in range(sides_.GetNumberOfTuples())) if sides_ else set()
    dn_ = tileModel.GetDisplayNode(); cn_ = dn_.GetColorNode()
    record("tile_two_tone_shading", sides_ is not None and nvals == {0, 1} and dn_.GetScalarVisibility() and dn_.GetActiveScalarName() == "TileSide"
           and cn_ is not None and cn_.GetNumberOfColors() == 2,
           f"TileSide values={sorted(nvals)}, scalarVisibility={dn_.GetScalarVisibility()}, activeScalar={dn_.GetActiveScalarName()}, colorNode={cn_.GetName() if cn_ else None}")

    # ---------- 5. Real 3D interactor click through onMouseClick ----------
    try:
        w.modeComboBox.setCurrentIndex(1); slicer.app.processEvents()
        record("placement_mode_flags", w.placing and not w.placing_seeds and w.interactorObserverTag is not None, f"placing={w.placing} tag={w.interactorObserverTag}")
        tdv = slicer.app.layoutManager().threeDWidget(0).threeDView()
        tdv.resetFocalPoint(); tdv.forceRender(); slicer.app.processEvents()
        interactor = tdv.interactorStyle().GetInteractor()
        size = interactor.GetRenderWindow().GetSize()
        interactor.SetEventPosition(size[0] // 2, size[1] // 2)
        before = len(w.logic.tiles)
        w.onMouseClick(interactor, None)
        record("tile_place_3d_click", len(w.logic.tiles) == before + 1, f"tiles {before}->{len(w.logic.tiles)}; status='{w.statusLabel.text}'")
    except Exception as e:
        record("tile_place_3d_click", False, traceback.format_exc())

    # ---------- 6. Slice-view click placement (ray cast through volume/surface) ----------
    lm = slicer.app.layoutManager()
    red = lm.sliceWidget("Red"); red.sliceLogic().SetSliceOffset(0.0); red.sliceView().forceRender(); slicer.app.processEvents()
    w.modeComboBox.setCurrentIndex(1); slicer.app.processEvents()
    before = len(w.logic.tiles)
    for off in (-80.0, 80.0):  # start the ray in background just outside the sphere; slice normal sign decides which works
        red.sliceLogic().SetSliceOffset(off); slicer.app.processEvents()
        xx, yy = ras_to_xy("Red", [0.0, 0.0, off])
        w.onSliceViewClick("Red", xx, yy)
        if len(w.logic.tiles) > before: break
    tl = w.logic.tiles[-1]
    record("tile_place_slice_click", len(w.logic.tiles) == before + 1 and abs(np.linalg.norm(tl.center) - 72) < 4, f"tiles {before}->{len(w.logic.tiles)}; center={np.round(tl.center,1)} normal={np.round(tl.normal,2)}; status='{w.statusLabel.text}'")

    # ---------- 7. Seed placement from slice view ----------
    w.modeComboBox.setCurrentIndex(2); slicer.app.processEvents()
    record("seeds_mode_flags", w.placing and w.placing_seeds, f"placing={w.placing} seeds={w.placing_seeds}")
    before = len(w.logic.seeds)
    for off in (-80.0, 80.0):
        red.sliceLogic().SetSliceOffset(off); slicer.app.processEvents()
        xx, yy = ras_to_xy("Red", [20.0, 10.0, off])
        w.onSliceViewSeedClick("Red", xx, yy)
        if len(w.logic.seeds) > before: break
    record("seed_place_slice_click", len(w.logic.seeds) == before + 1 and len(w.logic.seed_fiducial_nodes) == len(w.logic.seeds), f"seeds {before}->{len(w.logic.seeds)}; status='{w.statusLabel.text}'")

    # ---------- 8. Cursor mode: select tile in slice view, rotate, translate ----------
    w.modeComboBox.setCurrentIndex(0); slicer.app.processEvents()
    record("cursor_mode_flags", not w.placing and not w.placing_seeds, "")
    tsel = w.logic.tiles[-1]
    # Rotation is about the slice normal; pick the view whose normal is least aligned with the seed axis
    views = {"Red": 2, "Yellow": 0, "Green": 1}
    best = min(views, key=lambda v: abs(float(tsel.seed_axis_ras[views[v]])))
    sw = lm.sliceWidget(best); sw.sliceLogic().SetSliceOffset(float(tsel.center[views[best]])); slicer.app.processEvents()
    xx, yy = ras_to_xy(best, tsel.center)
    sel = w.onSliceViewTileSelection(best, xx, yy)
    record("tile_select_slice", bool(sel) and w.selected_tile_index == len(w.logic.tiles) - 1, f"selected={w.selected_tile_index} from {best}; label='{w.selectionLabel.text}'")
    tsel = w.logic.tiles[w.selected_tile_index]
    axis_before = tsel.seed_axis_ras.copy(); center_before = tsel.center.copy(); normal_before = tsel.normal.copy()
    w.onRotateTile(clockwise=True)
    axis_after = tsel.seed_axis_ras
    ang = np.degrees(np.arccos(np.clip(np.dot(axis_before, axis_after), -1, 1)))
    record("tile_rotate", 9.0 < ang < 11.0 and np.allclose(tsel.normal, normal_before, atol=1e-5), f"seed axis rotated {ang:.1f} deg, normal unchanged; status='{w.statusLabel.text}'")
    w.onRotateTile(clockwise=False)
    record("tile_rotate_back", np.allclose(tsel.seed_axis_ras, axis_before, atol=1e-4), "counterclockwise restores axis")
    sel_dn = w.logic.fiducial_nodes[w.selected_tile_index].GetDisplayNode()
    record("selected_tile_uniform_highlight", not sel_dn.GetScalarVisibility(), f"scalarVisibility while selected={sel_dn.GetScalarVisibility()}")
    moved_pd = w.logic.fiducial_nodes[w.selected_tile_index].GetPolyData()
    record("shading_survives_edit", moved_pd.GetCellData().GetArray("TileSide") is not None, "TileSide cell array present after rotate/translate")
    w.onTranslateTile("right")
    moved = np.linalg.norm(w.logic.tiles[w.selected_tile_index].center - center_before)
    record("tile_translate", 0.5 < moved < 5.0, f"moved {moved:.2f} mm; status='{w.statusLabel.text}'")
    # seed selection + translate
    ssel = w.logic.seeds[-1]
    red.sliceLogic().SetSliceOffset(float(ssel.center[2])); slicer.app.processEvents()
    xx, yy = ras_to_xy("Red", ssel.center)
    w.onSliceViewSeedSelection("Red", xx, yy)
    record("seed_select_slice", w.selected_seed_index == len(w.logic.seeds) - 1, f"selected seed={w.selected_seed_index}")
    sc = ssel.center.copy(); w.onTranslateTile("up")
    record("seed_translate", 0.5 < np.linalg.norm(w.logic.seeds[-1].center - sc) < 5.0, f"moved {np.linalg.norm(w.logic.seeds[-1].center - sc):.2f} mm")

    # ---------- 9. Dose update + isodoses ----------
    w.prescriptionSpinBox.value = 60
    w.logic.prescription_gy = w.prescriptionSpinBox.value
    t0 = time.time(); w.onUpdate(); dt = time.time() - t0
    dn = w.logic.dose_volume_node
    darr = slicer.util.arrayFromVolume(dn) if dn else None
    record("dose_update", dn is not None and darr is not None and np.nanmax(darr) > 0 and np.all(np.isfinite(darr)),
           f"max={np.nanmax(darr) if darr is not None else 'n/a'} cGy, dims={darr.shape if darr is not None else 'n/a'} in {dt:.2f}s; status='{w.statusLabel.text}'")
    iso = [nd for nd in w.logic.isodose_nodes_3d if nd and nd.GetScene()]
    record("isodose_surfaces", len(iso) == 3 and all(nd.GetPolyData().GetNumberOfPoints() > 0 for nd in iso),
           f"{len(iso)} isodose nodes: {[ (nd.GetName(), nd.GetPolyData().GetNumberOfPoints()) for nd in iso]}")
    record("dose_legend", w.legendWidget.layout().count() == 4, f"{w.legendWidget.layout().count()} legend layout entries (header + 3 levels)")
    # dose volume geometry: dose node bounds should match the volume RAS bounds
    b = [0.0]*6; dn.GetRASBounds(b); vb = [0.0]*6; vol.GetRASBounds(vb)
    record("dose_volume_geometry", all(abs(b[i]-vb[i]) < 4.0 for i in range(6)), f"dose bounds={np.round(b,1)} vol bounds={np.round(vb,1)}")
    # dose in RAS: max should be near a seed
    t0 = time.time(); w.onUpdate(); dt2 = time.time() - t0  # second update replaces nodes cleanly
    record("dose_update_repeat", w.logic.dose_volume_node.GetScene() is not None and len([nd for nd in w.logic.isodose_nodes_3d if nd.GetScene()]) == 3 and slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode") == 2, f"second update {dt2:.2f}s; scalar volumes in scene={slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLScalarVolumeNode')}")

    # ---------- 10. Exports ----------
    dn = w.logic.dose_volume_node  # the repeat update replaced the node
    darr = slicer.util.arrayFromVolume(dn)
    csvp = osmod.path.join(OUT, "positions.csv"); jsp = osmod.path.join(OUT, "positions.json")
    w._writePositionsCSV(csvp); w._writePositionsJSON(jsp)
    csv_lines = open(csvp).read().splitlines() if osmod.path.exists(csvp) else []
    js = json.load(open(jsp)) if osmod.path.exists(jsp) else {}
    record("export_csv", len(csv_lines) > 1 + 4*len(w.logic.tiles) + len(w.logic.seeds) - 1, f"{len(csv_lines)} lines; header='{csv_lines[0][:80] if csv_lines else ''}'")
    record("export_json", isinstance(js, dict) and js, f"keys={list(js.keys())[:8]}")
    interp = w.logic.interpolator
    record("dosimetry_dataset", interp.dataset == "TG43U1S2" and js.get("dosimetry_dataset") == interp.dataset_name
           and csv_lines[0].startswith("dosimetry_dataset,"), f"dataset='{interp.dataset_name}', in JSON={js.get('dosimetry_dataset')!r}, CSV first line='{csv_lines[0] if csv_lines else ''}'")
    ref = interp.DOSE_CONVERSION_FACTOR * 1.056  # theta=90, r=1 cm: G/G_ref = g_L = F = 1
    record("dose_reference_point", abs(interp.dose_at_point(90.0, 1.0) - ref) < 1e-6 * ref, f"dose(90 deg, 1 cm)={interp.dose_at_point(90.0, 1.0):.4f} cGy, expected {ref:.4f}")
    gy = w._createDoseVolumeInGy(dn)
    gy_arr = slicer.util.arrayFromVolume(gy) if gy else None
    record("dose_gy_conversion", gy is not None and abs(np.nanmax(gy_arr)*100 - np.nanmax(darr)) < 1.0, f"max Gy={np.nanmax(gy_arr) if gy is not None else 'n/a'}")
    dicom_dir = osmod.path.join(OUT, "dicom"); osmod.makedirs(dicom_dir, exist_ok=True)
    has_cds = hasattr(slicer.modules, "createdicomseries")
    if has_cds:
        okd = w._exportDoseVolumeAsDICOM(dn, dicom_dir, "DOSE", in_gy=False)
        files = sorted(f for f in osmod.listdir(dicom_dir) if f.endswith(".dcm"))
        record("export_dicom", okd and len(files) == darr.shape[0], f"ok={okd}, {len(files)} files (expect {darr.shape[0]} slices)")
        try:
            import pydicom
            dsets = [pydicom.dcmread(osmod.path.join(dicom_dir, f)) for f in files]
            pix_max = max(float(d.pixel_array.max()) for d in dsets)
            d0 = dsets[0]
            slope = float(getattr(d0, "RescaleSlope", 1.0))
            record("export_dicom_readback", d0.Modality == "SC" and slope == 1.0 and int(d0.BitsAllocated) == 32
                   and abs(pix_max - float(np.nanmax(darr))) < 1.0 and "cGy" in str(d0.SeriesDescription),
                   f"Modality={d0.Modality} RescaleSlope={slope} Bits={d0.BitsAllocated} SeriesDescription='{d0.SeriesDescription}' pixel max={pix_max} (dose max {np.nanmax(darr):.1f} cGy)")
        except Exception as e:
            record("export_dicom_readback", False, f"pydicom readback failed: {e}")
    else:
        record("export_dicom", False, "CreateDICOMSeries module not available")
    nrrd = osmod.path.join(OUT, "GammaTile_dose.nrrd")
    record("export_nrrd", slicer.util.saveNode(gy if gy else dn, nrrd) and osmod.path.getsize(nrrd) > 1000, f"{osmod.path.getsize(nrrd) if osmod.path.exists(nrrd) else 0} bytes")
    if gy: slicer.mrmlScene.RemoveNode(gy)

    # ---------- 11. Undo ----------
    nt, ns = len(w.logic.tiles), len(w.logic.seeds)
    w.onUndo(); w.onUndoSeed()
    live_tiles = [m for m in w.logic.fiducial_nodes if m and m.GetScene()]
    live_seeds = [m for m in w.logic.seed_fiducial_nodes if m and m.GetScene()]
    record("undo_tile_and_seed", len(w.logic.tiles) == nt-1 and len(w.logic.seeds) == ns-1 and len(live_tiles) == nt-1 and len(live_seeds) == ns-1,
           f"tiles {nt}->{len(w.logic.tiles)} (models {len(live_tiles)}), seeds {ns}->{len(w.logic.seeds)} (models {len(live_seeds)})")

    # ---------- 12. Volume ray cast + geometry-cache invalidation ----------
    hits = w.castRayThroughVolume(np.array([0.0, 0.0, -90.0]), np.array([0.0, 0.0, 1.0]), vol, 180.0)  # start inside the volume, in background
    record("ray_cast_volume", len(hits) > 0, f"{len(hits)} transitions: {[(round(h[0],1), h[3]) for h in hits[:6]]}")
    d1 = w._getVolumeSamplingData(vol)
    origin = list(vol.GetOrigin()); vol.SetOrigin(origin[0] + 10.0, origin[1], origin[2])
    d2 = w._getVolumeSamplingData(vol)
    vol.SetOrigin(*origin)
    d3 = w._getVolumeSamplingData(vol)
    record("sampling_cache_invalidates_on_geometry", (not np.allclose(d1[2], d2[2])) and np.allclose(d1[2], d3[2]), f"rasToIjk changed after origin edit: {not np.allclose(d1[2], d2[2])}")
    d4 = w._getVolumeSamplingData(vol)
    record("sampling_cache_reuses", d4 is d3, "same object returned when nothing changed")
    w.findNearestPointOnSurface(bs, [0,0,60.0]); w.findNearestPointOnSurface(bs, [0,0,-60.0])
    record("cell_locator_cache_single_entry", len(w._cell_locator_cache) == 1, f"{len(w._cell_locator_cache)} cached locators")

    # ---------- 13. Cleanup path ----------
    w.cleanup()
    record("cleanup", True, "cleanup() ran without exception")

except Exception:
    record("UNHANDLED_EXCEPTION", False, traceback.format_exc())

n_fail = sum(1 for r in RESULTS.values() if not r["ok"])
print(f"[TEST] SUMMARY: {len(RESULTS)-n_fail}/{len(RESULTS)} passed", flush=True)
import os, sys  # Slicer's --python-script wrapper expects these names to exist after the script runs
finish(0 if n_fail == 0 else 1)
