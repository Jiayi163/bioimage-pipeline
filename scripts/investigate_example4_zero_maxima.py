#!/usr/bin/env python3
"""Investigate zero-maxima fallback objects in example4 dense ROI (read-only)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io as skio
from skimage.draw import disk
from skimage.feature import peak_local_max
from skimage.measure import find_contours

from bioimage_pipeline.preprocess import gaussian_blur
from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.component_validity import detect_roi_saturation
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.detector_cache import load_peak_table_cache
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.peak_assignment import assign_peaks_to_objects
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate
from bioimage_pipeline.qc import normalize_for_display

BASE = Path(r"C:\Users\Administrator\Desktop\example4")
RUN_DIR = BASE / "phase_b_d"
IMAGE_PATH = BASE / "input" / "MAX_10% BSA block with EV_0007.tif"
MASK_PATH = BASE / "mask" / "MAX_10% BSA block with EV_0007_mask.tif"
CACHE_CSV = RUN_DIR / ".puncta_cache" / "EV_0007_phaseBD_candidates.csv"
OUT_DIR = RUN_DIR / "diagnostics" / "investigation" / "zero_maxima_fallback"

ROI = (258, 198, 552, 490)  # dense cluster bbox from prior investigation


@dataclass
class DetectorTrace:
    response_method: str
    response_max: float
    response_median: float
    threshold_abs: float
    raw_peak_count: int
    filtered_peak_count: int
    raw_before_filter_count: int
    filtered_before_relative_count: int
    min_peak_distance: int
    peak_min_relative_height: float
    peak_relative_prominence: float


@dataclass
class ObjectPeakDiag:
    object_id: int
    mask_label: int
    area: float
    equivalent_diameter: float
    roi_rows: int
    roi_cols: int
    raw_min: float
    raw_max: float
    raw_range: float
    corrected_max: float
    background_level: float
    saturation_present: bool
    saturation_near_clip_fraction: float
    brightest_row: float
    brightest_col: float
    assigned_image_peaks: int
    nearest_image_peak_dist_px: float | None
    nearest_image_peak_on_label: int | None
    image_peak_inside_mask: bool
    object_detector_raw: int
    object_detector_filtered: int
    failure_category: str
    failure_detail: str
    alt_plateau_aware: int
    alt_brightest_seed: int
    alt_min_dist_1: int
    alt_no_smoothing: int
    alt_masked_argmax: int
    alt_light_smoothing: int


def _load_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(skio.imread(IMAGE_PATH), dtype=np.float64)
    mask = np.asarray(skio.imread(MASK_PATH))
    if image.ndim > 2:
        image = image[..., 0]
    if mask.ndim > 2:
        mask = mask[..., 0]
    return image, mask > 0


def _trace_detector(detector: MaximaDetector, patch_raw: np.ndarray, patch_mask: np.ndarray) -> tuple[DetectorTrace, np.ndarray, np.ndarray]:
    cfg = detector.config
    response, method = detector._build_response(patch_raw)  # noqa: SLF001
    threshold = detector._compute_threshold_abs(response, patch_mask)  # noqa: SLF001
    labels = patch_mask.astype(np.int32)

    raw_coords = peak_local_max(
        response,
        labels=labels,
        min_distance=1,
        threshold_abs=threshold,
        exclude_border=False,
    )
    filt_coords = peak_local_max(
        response,
        labels=labels,
        min_distance=max(1, cfg.min_peak_distance),
        threshold_abs=threshold,
        exclude_border=False,
    )
    raw_peaks = detector._coords_to_peaks(raw_coords, response)  # noqa: SLF001
    filt_peaks = detector._apply_relative_filters(  # noqa: SLF001
        detector._coords_to_peaks(filt_coords, response),
        response,
        patch_mask,
    )
    masked = response[patch_mask]
    return (
        DetectorTrace(
            response_method=method,
            response_max=float(masked.max()) if masked.size else 0.0,
            response_median=float(np.median(masked)) if masked.size else 0.0,
            threshold_abs=float(threshold) if threshold is not None else 0.0,
            raw_peak_count=len(raw_peaks),
            filtered_peak_count=len(filt_peaks),
            raw_before_filter_count=len(raw_coords),
            filtered_before_relative_count=len(filt_coords),
            min_peak_distance=cfg.min_peak_distance,
            peak_min_relative_height=cfg.peak_min_relative_height,
            peak_relative_prominence=cfg.peak_relative_prominence,
        ),
        response,
        patch_raw,
    )


def _alt_plateau_aware(response: np.ndarray, patch_mask: np.ndarray, *, min_distance: int) -> int:
    """Plateau-aware: accept regional maxima including flat tops."""
    masked_resp = np.where(patch_mask, response, -np.inf)
    if not patch_mask.any():
        return 0
    peak_val = float(masked_resp.max())
    if peak_val <= 0 or not np.isfinite(peak_val):
        return 0
    # Treat pixels within 99.5% of peak as plateau support; label plateaus then pick centroids.
    plateau = patch_mask & (response >= 0.995 * peak_val)
    labeled, n = ndi.label(plateau)
    if n == 0:
        return 0
    coords = ndi.center_of_mass(plateau, labeled, index=range(1, n + 1))
    return len(coords)


def _alt_brightest_seed(patch_raw: np.ndarray, patch_mask: np.ndarray) -> int:
    if not patch_mask.any():
        return 0
    vals = patch_raw[patch_mask]
    return 1 if float(vals.max()) > float(np.median(vals)) else 0


def _alt_min_dist_1(response: np.ndarray, patch_mask: np.ndarray, threshold: float) -> int:
    coords = peak_local_max(
        response,
        labels=patch_mask.astype(np.int32),
        min_distance=1,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return len(coords)


def _alt_no_smoothing(patch_raw: np.ndarray, patch_mask: np.ndarray, threshold_frac: float = 0.08) -> int:
    response = patch_raw.copy()
    masked = response[patch_mask]
    if masked.size == 0:
        return 0
    bg = float(np.median(masked))
    peak = float(masked.max())
    threshold = bg + threshold_frac * max(peak - bg, 0.0)
    coords = peak_local_max(
        response,
        labels=patch_mask.astype(np.int32),
        min_distance=1,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return len(coords)


def _alt_masked_argmax(patch_raw: np.ndarray, patch_mask: np.ndarray) -> int:
    if not patch_mask.any():
        return 0
    flat_idx = int(np.argmax(np.where(patch_mask, patch_raw, -np.inf)))
    row, col = np.unravel_index(flat_idx, patch_raw.shape)
    # Count as recovery only if not on mask edge exclusively
    return 1


def _alt_light_smoothing(patch_raw: np.ndarray, patch_mask: np.ndarray) -> int:
    response = gaussian_blur(patch_raw, sigma=0.15)
    masked = response[patch_mask]
    bg = float(np.median(masked))
    peak = float(masked.max())
    threshold = bg + 0.08 * max(peak - bg, 0.0)
    coords = peak_local_max(
        response,
        labels=patch_mask.astype(np.int32),
        min_distance=1,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return len(coords)


def _classify_failure(
    *,
    assigned: int,
    obj_raw: int,
    obj_filt: int,
    trace: DetectorTrace,
    saturation_present: bool,
    area: float,
    eq_d: float,
    image_peak_inside: bool,
    nearest_dist: float | None,
    patch_mask: np.ndarray,
    response: np.ndarray,
) -> tuple[str, str]:
    mask_pixels = int(patch_mask.sum())
    if assigned >= 1:
        return "assigned_ok", "image-level peak assigned"

    # Fast path records zero maxima when no IMAGE-LEVEL peak lands on this label.
    # Object-level MaximaDetector replay may still succeed.
    if obj_filt >= 1:
        dist_note = ""
        if not image_peak_inside and nearest_dist is not None:
            dist_note = f"nearest_global_peak_{nearest_dist:.1f}px_on_neighbor_label"
        return "G", (
            f"image_assignment_miss; object_detector_finds_{obj_filt}_filtered_peak(s)"
            + (f"; {dist_note}" if dist_note else "")
        )

    details: list[str] = []

    # Image-level assignment failure modes
    if not image_peak_inside and nearest_dist is not None and nearest_dist <= 3.0:
        details.append(f"nearest_image_peak_{nearest_dist:.1f}px_on_neighbor_label")

    if obj_raw == 0 and trace.response_max <= trace.threshold_abs + 1e-9:
        if saturation_present:
            return "C", "saturated plateau; response max at/below threshold"
        return "D", "detector threshold above response max"

    if obj_raw == 0 and trace.response_max > 0 and trace.raw_before_filter_count == 0:
        if saturation_present:
            return "C", "flat saturated top; no strict local maximum in DoG/LoG response"
        if trace.response_method == "dog":
            details.append("dog_suppressed_peak")
        return "B", ";".join(details) or "smoothing/DoG removed strict local maximum"

    if obj_raw > 0 and obj_filt == 0:
        if mask_pixels <= 9 and trace.min_peak_distance >= 2:
            return "A", f"raw={obj_raw} but filtered=0; area={mask_pixels}px min_dist={trace.min_peak_distance}"
        if trace.filtered_before_relative_count > 0:
            return "D", "relative height / prominence filter removed all peaks"
        return "A", "min_peak_distance removed peaks on tiny object"

    if obj_raw == 0 and area <= 12 and eq_d < 4.5:
        return "A", f"too small for reliable peak (area={area:.0f}, eq_d={eq_d:.1f})"

    if not patch_mask[0, :].any() and not patch_mask[-1, :].any() and not patch_mask[:, 0].any() and not patch_mask[:, -1].any():
        pass
    else:
        # mask touches patch border — can affect peak_local_max neighborhood
        touches = (
            patch_mask[0, :].any() or patch_mask[-1, :].any() or patch_mask[:, 0].any() or patch_mask[:, -1].any()
        )
        if touches and obj_raw == 0:
            details.append("mask_touches_patch_border")

    if saturation_present and obj_raw == 0:
        return "G", "saturation + " + (";".join(details) if details else "no strict peak")

    if float(response[patch_mask].max()) <= 0:
        return "F", "no positive detector response inside mask"

    return "G", ";".join(details) if details else "combination of assignment + local detector failure"


def _nearest_image_peak(
    peaks: list[PeakCandidate],
    labeled: np.ndarray,
    mask_label: int,
    br: float,
    bc: float,
) -> tuple[float | None, int | None, bool]:
    rows, cols = np.where(labeled == mask_label)
    if rows.size == 0:
        return None, None, False
    inside = False
    best_dist = None
    best_label = None
    for peak in peaks:
        pr, pc = int(round(peak.row)), int(round(peak.col))
        if 0 <= pr < labeled.shape[0] and 0 <= pc < labeled.shape[1]:
            if labeled[pr, pc] == mask_label:
                inside = True
            dist = float(np.hypot(peak.row - br, peak.col - bc))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_label = int(labeled[pr, pc]) if 0 <= pr < labeled.shape[0] and 0 <= pc < labeled.shape[1] else None
    return best_dist, best_label, inside


def _save_panel(
    object_id: int,
    patch_raw: np.ndarray,
    patch_mask: np.ndarray,
    response: np.ndarray,
    trace: DetectorTrace,
    *,
    saturated: bool,
) -> None:
    raw_disp = normalize_for_display(patch_raw)
    base = np.stack([raw_disp, raw_disp, raw_disp], axis=-1).astype(np.uint8)
    mask_panel = np.stack([patch_mask * 255] * 3, axis=-1).astype(np.uint8)

    resp = response.copy()
    resp[~patch_mask] = np.nan
    resp_disp = normalize_for_display(np.nan_to_num(resp, nan=0.0))
    resp_rgb = np.stack([resp_disp, resp_disp, resp_disp], axis=-1).astype(np.uint8)

    thresh_panel = resp_rgb.copy()
    thresh_val = trace.threshold_abs
    thresh_mask = patch_mask & (response >= thresh_val)
    thresh_panel[thresh_mask] = [255, 128, 0]

    cand_panel = base.copy()
    labels = patch_mask.astype(np.int32)
    raw_coords = peak_local_max(
        response, labels=labels, min_distance=1, threshold_abs=trace.threshold_abs, exclude_border=False
    )
    for row, col in raw_coords:
        _draw_cross(cand_panel, row, col, (0, 255, 255), size=2)

    final_panel = base.copy()
    filt_coords = peak_local_max(
        response,
        labels=labels,
        min_distance=max(1, trace.min_peak_distance),
        threshold_abs=trace.threshold_abs,
        exclude_border=False,
    )
    for row, col in filt_coords:
        _draw_cross(final_panel, row, col, (0, 255, 0), size=2)

    # Mark brightest pixel
    flat = int(np.argmax(np.where(patch_mask, patch_raw, -np.inf)))
    br, bc = np.unravel_index(flat, patch_raw.shape)
    _draw_cross(cand_panel, br, bc, (255, 0, 255), size=1)
    _draw_cross(final_panel, br, bc, (255, 0, 255), size=1)

    top = np.concatenate([base, mask_panel, resp_rgb], axis=1)
    bottom = np.concatenate([thresh_panel, cand_panel, final_panel], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    # Upscale tiny ROIs for readability (nearest-neighbor).
    scale = max(1, int(np.ceil(80 / max(panel.shape[0], panel.shape[1]))))
    if scale > 1:
        panel = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
    tag = "sat" if saturated else "nonsat"
    skio.imsave(OUT_DIR / "panels" / f"object_{object_id:04d}_{tag}.png", panel)


def _draw_cross(rgb: np.ndarray, row: float, col: float, color: tuple[int, int, int], size: int = 2) -> None:
    r, c = int(round(row)), int(round(col))
    h, w = rgb.shape[:2]
    for dr in range(-size, size + 1):
        for dc in range(-size, size + 1):
            if abs(dr) == size or abs(dc) == size:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    rgb[rr, cc] = color


def _zero_maxima_object_ids(diag: pd.DataFrame, meas: pd.DataFrame) -> list[int]:
    cent = meas.groupby("object_id").first()[["initial_row", "initial_col"]]
    r0, c0, r1, c1 = ROI
    in_roi = cent[
        (cent.initial_row >= r0) & (cent.initial_row < r1) & (cent.initial_col >= c0) & (cent.initial_col < c1)
    ].index
    sub = diag[diag.object_id.isin(in_roi)]
    fb = sub[(sub.path == "fallback") & (sub.n_filtered_local_maxima == 0)]
    return sorted(int(x) for x in fb.object_id.tolist())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "panels").mkdir(exist_ok=True)

    image, mask_bool = _load_image()
    labeled, objects = ConnectedObjectAnalyzer().analyze(mask_bool, image)
    objects_by_label = {o.label: o for o in objects}
    label_to_oid: dict[int, int] = {}

    diag = pd.read_csv(RUN_DIR / "EV_0007_phaseBD_object_diagnostics.csv")
    meas = pd.read_csv(RUN_DIR / "EV_0007_phaseBD_measurements.csv")
    for oid, grp in meas.groupby("object_id"):
        r, c = float(grp.iloc[0]["initial_row"]), float(grp.iloc[0]["initial_col"])
        lbl = int(labeled[int(round(r)), int(round(c))])
        label_to_oid[lbl] = int(oid)

    config = PunctaDeclumpConfig(threshold_method="external_mask")
    peak_table = load_peak_table_cache(CACHE_CSV, "python_log")
    assigned = assign_peaks_to_objects(labeled, objects, peak_table, config)
    detector = MaximaDetector(config)

    target_ids = _zero_maxima_object_ids(diag, meas)
    records: list[ObjectPeakDiag] = []

    for oid in target_ids:
        drow = diag.loc[diag.object_id == oid].iloc[0]
        mrow = meas.loc[meas.object_id == oid].iloc[0]
        br, bc = float(mrow["initial_row"]), float(mrow["initial_col"])
        mask_label = int(labeled[int(round(br)), int(round(bc))])
        obj = objects_by_label[mask_label]
        object_mask = labeled == mask_label
        patch = build_object_patch(image, object_mask, obj, config)
        patch_raw = patch.raw if patch.raw is not None else patch.corrected + patch.background_level

        trace, response, _ = _trace_detector(detector, patch.corrected, patch.object_mask)
        sat = detect_roi_saturation(
            patch,
            near_clip_margin=config.saturation_near_clip_margin,
            near_clip_fraction_threshold=config.saturation_near_clip_fraction,
        )
        assigned_peaks = assigned.get(mask_label, [])
        near_dist, near_label, inside = _nearest_image_peak(
            peak_table.peaks, labeled, mask_label, br, bc
        )

        failure_cat, failure_detail = _classify_failure(
            assigned=len(assigned_peaks),
            obj_raw=trace.raw_peak_count,
            obj_filt=trace.filtered_peak_count,
            trace=trace,
            saturation_present=sat.present,
            area=float(drow["area"]),
            eq_d=float(drow["equivalent_diameter"]),
            image_peak_inside=inside,
            nearest_dist=near_dist,
            patch_mask=patch.object_mask,
            response=response,
        )

        rec = ObjectPeakDiag(
            object_id=oid,
            mask_label=mask_label,
            area=float(drow["area"]),
            equivalent_diameter=float(drow["equivalent_diameter"]),
            roi_rows=patch.corrected.shape[0],
            roi_cols=patch.corrected.shape[1],
            raw_min=float(patch_raw[patch.object_mask].min()),
            raw_max=float(patch_raw[patch.object_mask].max()),
            raw_range=float(patch_raw[patch.object_mask].max() - patch_raw[patch.object_mask].min()),
            corrected_max=float(patch.corrected[patch.object_mask].max()),
            background_level=float(patch.background_level),
            saturation_present=sat.present,
            saturation_near_clip_fraction=sat.near_clip_fraction,
            brightest_row=float(obj.brightest_row),
            brightest_col=float(obj.brightest_col),
            assigned_image_peaks=len(assigned_peaks),
            nearest_image_peak_dist_px=near_dist,
            nearest_image_peak_on_label=near_label,
            image_peak_inside_mask=inside,
            object_detector_raw=trace.raw_peak_count,
            object_detector_filtered=trace.filtered_peak_count,
            failure_category=failure_cat,
            failure_detail=failure_detail,
            alt_plateau_aware=_alt_plateau_aware(response, patch.object_mask, min_distance=1),
            alt_brightest_seed=_alt_brightest_seed(patch_raw, patch.object_mask),
            alt_min_dist_1=_alt_min_dist_1(response, patch.object_mask, trace.threshold_abs),
            alt_no_smoothing=_alt_no_smoothing(patch_raw, patch.object_mask),
            alt_masked_argmax=_alt_masked_argmax(patch_raw, patch.object_mask),
            alt_light_smoothing=_alt_light_smoothing(patch_raw, patch.object_mask),
        )
        records.append(rec)

    table = pd.DataFrame([asdict(r) for r in records])
    table.to_csv(OUT_DIR / "zero_maxima_analysis.csv", index=False)

    # Category counts
    cat_counts = table["failure_category"].value_counts().to_dict()

    # Sample panels: all 36 are saturated; use 5 lowest + 5 highest near-clip fraction.
    sorted_sat = table.sort_values("saturation_near_clip_fraction")
    panel_ids = sorted_sat.head(5)["object_id"].tolist() + sorted_sat.tail(5)["object_id"].tolist()
    panel_ids = list(dict.fromkeys(panel_ids))  # unique preserve order
    for oid in panel_ids:
        mrow = meas.loc[meas.object_id == oid].iloc[0]
        br, bc = float(mrow["initial_row"]), float(mrow["initial_col"])
        mask_label = int(labeled[int(round(br)), int(round(bc))])
        obj = objects_by_label[mask_label]
        patch = build_object_patch(image, labeled == mask_label, obj, config)
        patch_raw = patch.raw if patch.raw is not None else patch.corrected + patch.background_level
        trace, response, _ = _trace_detector(detector, patch.corrected, patch.object_mask)
        sat = detect_roi_saturation(
            patch,
            near_clip_margin=config.saturation_near_clip_margin,
            near_clip_fraction_threshold=config.saturation_near_clip_fraction,
        )
        _save_panel(oid, patch_raw, patch.object_mask, response, trace, saturated=sat.present)

    # Alternative recovery summary on zero-maxima set
    alt_summary = {
        "n_objects": len(table),
        "category_counts": cat_counts,
        "alt_recovers_any": {
            "plateau_aware": int((table["alt_plateau_aware"] > 0).sum()),
            "brightest_seed": int((table["alt_brightest_seed"] > 0).sum()),
            "min_dist_1": int((table["alt_min_dist_1"] > 0).sum()),
            "no_smoothing": int((table["alt_no_smoothing"] > 0).sum()),
            "masked_argmax": int((table["alt_masked_argmax"] > 0).sum()),
            "light_smoothing": int((table["alt_light_smoothing"] > 0).sum()),
        },
    }

    # Control: ordinary isolated fast_single objects outside dense ROI (no false peaks)
    cent = meas.groupby("object_id").first()[["initial_row", "initial_col"]]
    r0, c0, r1, c1 = ROI
    outside = cent[
        ~((cent.initial_row >= r0) & (cent.initial_row < r1) & (cent.initial_col >= c0) & (cent.initial_col < c1))
    ].index
    control = diag[(diag.object_id.isin(outside)) & (diag.path == "fast_single") & (diag.n_accepted_fit_ok == 1)]
    control = control.sample(min(30, len(control)), random_state=0) if len(control) else control
    control_fp: dict[str, int] = {}
    for oid in control.object_id.tolist():
        mrow = meas.loc[meas.object_id == oid].iloc[0]
        br, bc = float(mrow["initial_row"]), float(mrow["initial_col"])
        mask_label = int(labeled[int(round(br)), int(round(bc))])
        obj = objects_by_label[mask_label]
        patch = build_object_patch(image, labeled == mask_label, obj, config)
        trace, response, _ = _trace_detector(detector, patch.corrected, patch.object_mask)
        prod = trace.filtered_peak_count
        alt = {
            "plateau_aware": _alt_plateau_aware(response, patch.object_mask, min_distance=1),
            "brightest_seed": 1,
            "min_dist_1": _alt_min_dist_1(response, patch.object_mask, trace.threshold_abs),
            "no_smoothing": _alt_no_smoothing(
                patch.raw if patch.raw is not None else patch.corrected + patch.background_level,
                patch.object_mask,
            ),
            "masked_argmax": 1,
            "light_smoothing": _alt_light_smoothing(
                patch.raw if patch.raw is not None else patch.corrected + patch.background_level,
                patch.object_mask,
            ),
        }
        for name, count in alt.items():
            if count > max(prod, 1):  # more peaks than production single
                control_fp[name] = control_fp.get(name, 0) + 1

    report = {
        "note": (
            "Exported n_raw/n_filtered maxima=0 on fast-path fallback objects means "
            "zero IMAGE-LEVEL peaks assigned via assign_peaks_to_objects, not object-level detect(). "
            "This script analyzes both assignment and per-object MaximaDetector replay."
        ),
        "n_zero_maxima_fallback_in_dense_roi": len(target_ids),
        "category_counts": cat_counts,
        "alt_recovery_on_failures": alt_summary["alt_recovers_any"],
        "control_false_peak_risk_30_isolated_fast_single": control_fp,
        "recommended_read_only_alternatives": [],
    }

    # Recommend alternatives that recover most failures with low control FP
    for name, recovered in alt_summary["alt_recovers_any"].items():
        fp = control_fp.get(name, 0)
        if recovered >= len(table) * 0.5 and fp <= 2:
            report["recommended_read_only_alternatives"].append(
                {"method": name, "recovered": recovered, "control_extra_peaks": fp}
            )

    (OUT_DIR / "zero_maxima_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
