#!/usr/bin/env python3
"""Read-only full-image vs crop comparison for example4 peak/assignment differences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio
from skimage.draw import disk

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_detectors.python_log import PythonLoGDetector
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.detector_cache import load_peak_table_cache
from bioimage_pipeline.puncta.local_peak_recovery import finalize_recovery
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.peak_assignment import assign_peaks_to_objects, count_reliable_assigned_peaks
from bioimage_pipeline.puncta.types import PeakCandidate, PeakDetectionResult
from bioimage_pipeline.qc import normalize_for_display

BASE = Path(r"C:\Users\Administrator\Desktop\example4")
CROP_BASE = Path(r"C:\Users\Administrator\Desktop\example4_crop")
FULL_RUN = BASE / "phase_b_d_recovery_fast"
CROP_RUN = CROP_BASE / "phase_b_d"
FULL_IMAGE = BASE / "input" / "MAX_10% BSA block with EV_0007.tif"
FULL_MASK = BASE / "mask" / "MAX_10% BSA block with EV_0007_mask.tif"
CROP_IMAGE = CROP_BASE / "input" / "MAX_10% BSA block with EV_0007_crop.tif"
CROP_MASK = CROP_BASE / "mask" / "MAX_10% BSA block with EV_0007_crop_mask.tif"
OUT_DIR = FULL_RUN / "diagnostics" / "investigation" / "full_vs_crop"

# Prior dense-cluster ROI (center ~404,342)
DENSE_ROI = (258, 198, 552, 490)
PEAK_MATCH_TOL = 1.5
OBJECT_MATCH_TOL = 3.0


@dataclass
class DetectorContext:
    scope: str
    method: str
    threshold_abs: float
    response_max: float
    response_median: float
    response_p95: float
    n_raw_before_filter: int
    n_filtered: int


def _load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(skio.imread(path), dtype=np.float64)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def _find_crop_offset(full: np.ndarray, crop: np.ndarray) -> tuple[int, int]:
    h, w = crop.shape
    best_score = -1.0
    best = (0, 0)
    for r0 in range(0, full.shape[0] - h + 1, 2):
        for c0 in range(0, full.shape[1] - w + 1, 2):
            patch = full[r0 : r0 + h, c0 : c0 + w]
            score = float(np.corrcoef(patch.ravel(), crop.ravel())[0, 1])
            if score > best_score:
                best_score = score
                best = (r0, c0)
    r0, c0 = best
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            rr, cc = r0 + dr, c0 + dc
            if rr < 0 or cc < 0 or rr + h > full.shape[0] or cc + w > full.shape[1]:
                continue
            patch = full[rr : rr + h, cc : cc + w]
            score = float(np.corrcoef(patch.ravel(), crop.ravel())[0, 1])
            if score > best_score:
                best_score = score
                best = (rr, cc)
    return best


def _in_box(row: float, col: float, r0: int, c0: int, r1: int, c1: int) -> bool:
    return r0 <= row < r1 and c0 <= col < c1


def _map_crop_peak(peak: PeakCandidate, r0: int, c0: int) -> PeakCandidate:
    return PeakCandidate(row=peak.row + r0, col=peak.col + c0, intensity=peak.intensity)


def _match_peaks(
    left: list[PeakCandidate],
    right: list[PeakCandidate],
    *,
    tol: float = PEAK_MATCH_TOL,
) -> tuple[int, list[PeakCandidate], list[PeakCandidate]]:
    used: set[int] = set()
    matched = 0
    for rp in right:
        for i, lp in enumerate(left):
            if i in used:
                continue
            if float(np.hypot(lp.row - rp.row, lp.col - rp.col)) <= tol:
                matched += 1
                used.add(i)
                break
    left_only = [left[i] for i in range(len(left)) if i not in used]
    right_only: list[PeakCandidate] = []
    for rp in right:
        if not any(float(np.hypot(lp.row - rp.row, lp.col - rp.col)) <= tol for lp in left):
            right_only.append(rp)
    return matched, left_only, right_only


def _image_detector_context(image: np.ndarray, config: PunctaDeclumpConfig, *, scope: str) -> DetectorContext:
    detector = MaximaDetector(config)
    response, method = detector._build_response(image)  # noqa: SLF001
    threshold = float(detector._compute_threshold_abs(response, np.ones_like(response, dtype=bool)))  # noqa: SLF001
    raw_coords = __import__("skimage.feature", fromlist=["peak_local_max"]).peak_local_max(
        response,
        labels=np.ones(response.shape, dtype=np.int32),
        min_distance=1,
        threshold_abs=threshold,
        exclude_border=False,
    )
    filtered = detector._apply_relative_filters(  # noqa: SLF001
        detector._coords_to_peaks(raw_coords, response),  # noqa: SLF001
        response,
        np.ones(response.shape, dtype=bool),
    )
    vals = response.ravel()
    return DetectorContext(
        scope=scope,
        method=method,
        threshold_abs=threshold,
        response_max=float(vals.max()),
        response_median=float(np.median(vals)),
        response_p95=float(np.percentile(vals, 95)),
        n_raw_before_filter=len(raw_coords),
        n_filtered=len(filtered),
    )


def _object_rows(
    labels: np.ndarray,
    objects: list,
    image: np.ndarray,
    peak_table,
    config: PunctaDeclumpConfig,
    *,
    offset_r: int = 0,
    offset_c: int = 0,
) -> pd.DataFrame:
    assigned = assign_peaks_to_objects(labels, objects, peak_table, config)
    detector = MaximaDetector(config)
    rows: list[dict[str, object]] = []
    for obj in objects:
        peaks = assigned.get(obj.label, [])
        mask = labels == obj.label
        patch = build_object_patch(image, mask, obj, config)
        detection = detector.detect(patch.corrected, patch.object_mask)

        def _global(peaks_local: list[PeakCandidate]) -> list[PeakCandidate]:
            return [
                PeakCandidate(
                    row=p.row + patch.row_offset,
                    col=p.col + patch.col_offset,
                    intensity=p.intensity + patch.background_level,
                )
                for p in peaks_local
            ]

        local_det = PeakDetectionResult(
            raw_peaks=_global(detection.raw_peaks),
            filtered_peaks=_global(detection.filtered_peaks),
            method=detection.method,
        )
        recovery = finalize_recovery(local_det, patch, obj, config)
        rows.append(
            {
                "mask_label": obj.label,
                "centroid_row": obj.centroid[0] + offset_r,
                "centroid_col": obj.centroid[1] + offset_c,
                "area": obj.area,
                "assigned_peak_count": len(peaks),
                "assigned_reliable_peak_count": count_reliable_assigned_peaks(peaks, config),
                "local_raw_count": len(local_det.raw_peaks),
                "local_filtered_count": len(local_det.filtered_peaks),
                "recovery_filtered_count": recovery.filtered_count,
            }
        )
    return pd.DataFrame(rows)


def _match_objects(full_df: pd.DataFrame, crop_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    used_crop: set[int] = set()
    for _, frow in full_df.iterrows():
        best_idx = None
        best_dist = None
        for j, crow in crop_df.iterrows():
            if j in used_crop:
                continue
            dist = float(np.hypot(frow.centroid_row - crow.centroid_row, frow.centroid_col - crow.centroid_col))
            if dist <= OBJECT_MATCH_TOL and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = j
        if best_idx is None:
            continue
        used_crop.add(best_idx)
        crow = crop_df.loc[best_idx]
        records.append(
            {
                "full_centroid_row": frow.centroid_row,
                "full_centroid_col": frow.centroid_col,
                "crop_object_id": int(crow.get("object_id", best_idx)),
                "full_object_id": int(frow.get("object_id", -1)),
                "full_assigned_peaks": int(frow["assigned_peak_count"]),
                "crop_assigned_peaks": int(crow["assigned_peak_count"]),
                "full_local_filtered": int(frow["local_filtered_count"]),
                "crop_local_filtered": int(crow["local_filtered_count"]),
                "full_recovery_filtered": int(frow.get("recovery_filtered_count", 0)),
                "full_path": frow.get("path"),
                "crop_path": crow.get("path"),
                "full_accepted": int(frow.get("n_accepted_fit_ok", 0)),
                "crop_accepted": int(crow.get("n_accepted_fit_ok", 0)),
                "full_peak_source": frow.get("peak_source"),
                "full_n_raw_exported": frow.get("n_filtered_local_maxima"),
                "crop_n_raw_exported": crow.get("n_filtered_local_maxima"),
                "full_recovery_attempted": frow.get("local_peak_recovery_attempted"),
                "full_recovery_filtered_export": frow.get("local_peak_recovery_filtered_count"),
            }
        )
    return pd.DataFrame(records)


def _draw_cross(rgb: np.ndarray, row: float, col: float, color: tuple[int, int, int], size: int = 2) -> None:
    r, c = int(round(row)), int(round(col))
    h, w = rgb.shape[:2]
    for dr in range(-size, size + 1):
        rr, cc = r + dr, c
        if 0 <= rr < h and 0 <= cc < w:
            rgb[rr, cc] = color
        rr, cc = r, c + dr
        if 0 <= rr < h and 0 <= cc < w:
            rgb[rr, cc] = color


def _save_peak_overlay(
    image: np.ndarray,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    full_peaks: list[PeakCandidate],
    crop_peaks_full: list[PeakCandidate],
    crop_only: list[PeakCandidate],
    path: Path,
) -> None:
    roi = normalize_for_display(image[r0:r1, c0:c1])
    base = np.stack([roi, roi, roi], axis=-1).astype(np.uint8)
    for peak in full_peaks:
        _draw_cross(base, peak.row - r0, peak.col - c0, (0, 200, 255), size=2)
    for peak in crop_peaks_full:
        _draw_cross(base, peak.row - r0, peak.col - c0, (0, 255, 0), size=2)
    for peak in crop_only:
        _draw_cross(base, peak.row - r0, peak.col - c0, (255, 80, 255), size=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(path, base)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = PunctaDeclumpConfig(threshold_method="external_mask")

    full_image = _load_gray(FULL_IMAGE)
    full_mask = _load_gray(FULL_MASK) > 0
    crop_image = _load_gray(CROP_IMAGE)
    crop_mask = _load_gray(CROP_MASK) > 0
    crop_r0, crop_c0 = _find_crop_offset(full_image, crop_image)
    crop_h, crop_w = crop_image.shape
    crop_r1, crop_c1 = crop_r0 + crop_h, crop_c0 + crop_w

    full_labels, full_objects = ConnectedObjectAnalyzer().analyze(full_mask, full_image)
    crop_labels, crop_objects = ConnectedObjectAnalyzer().analyze(crop_mask, crop_image)

    full_peaks = load_peak_table_cache(
        FULL_RUN / ".puncta_cache" / "EV_0007_phaseBD_candidates.csv", "python_log"
    ).peaks
    crop_peaks = load_peak_table_cache(
        CROP_RUN / ".puncta_cache" / "EV_0007_crop_phaseBD_candidates.csv", "python_log"
    ).peaks
    crop_peaks_full = [_map_crop_peak(p, crop_r0, crop_c0) for p in crop_peaks]

    # A. Peak detection in actual crop bbox
    full_in_crop = [p for p in full_peaks if _in_box(p.row, p.col, crop_r0, crop_c0, crop_r1, crop_c1)]
    matched, full_only, crop_only = _match_peaks(full_in_crop, crop_peaks_full)

    # Simulated dense ROI crop: re-run detector on extracted subimage
    dr0, dc0, dr1, dc1 = DENSE_ROI
    dense_image = full_image[dr0:dr1, dc0:dc1]
    dense_mask = full_mask[dr0:dr1, dc0:dc1]
    dense_detector = PythonLoGDetector()
    dense_table = dense_detector.detect(dense_image, config=config, source_path="dense_extract")
    dense_peaks_full = [
        PeakCandidate(row=p.row + dr0, col=p.col + dc0, intensity=p.intensity) for p in dense_table.peaks
    ]
    full_in_dense = [p for p in full_peaks if _in_box(p.row, p.col, dr0, dc0, dr1, dc1)]
    dense_matched, dense_full_only, dense_crop_only = _match_peaks(full_in_dense, dense_peaks_full)

    detector_context = [
        _image_detector_context(full_image, config, scope="full_image"),
        _image_detector_context(crop_image, config, scope="on_disk_crop"),
        _image_detector_context(dense_image, config, scope="simulated_dense_crop"),
    ]

    # B. Per-object assignment replay in crop bbox and dense ROI
    full_diag = pd.read_csv(FULL_RUN / "EV_0007_phaseBD_object_diagnostics.csv")
    full_meas = pd.read_csv(FULL_RUN / "EV_0007_phaseBD_measurements.csv")
    crop_diag = pd.read_csv(CROP_RUN / "EV_0007_crop_phaseBD_object_diagnostics.csv")
    crop_meas = pd.read_csv(CROP_RUN / "EV_0007_crop_phaseBD_measurements.csv")

    cent_full = full_meas.groupby("object_id").first()[["initial_row", "initial_col"]]
    full_in_crop_ids = [
        int(oid)
        for oid, row in cent_full.iterrows()
        if _in_box(row.initial_row, row.initial_col, crop_r0, crop_c0, crop_r1, crop_c1)
    ]
    full_in_dense_ids = [
        int(oid)
        for oid, row in cent_full.iterrows()
        if _in_box(row.initial_row, row.initial_col, dr0, dc0, dr1, dc1)
    ]

    # Replay assignment/local counts for full-image objects in crop bbox
    full_obj_df = _object_rows(full_labels, full_objects, full_image, load_peak_table_cache(
        FULL_RUN / ".puncta_cache" / "EV_0007_phaseBD_candidates.csv", "python_log"
    ), config)
    crop_obj_df = _object_rows(crop_labels, crop_objects, crop_image, load_peak_table_cache(
        CROP_RUN / ".puncta_cache" / "EV_0007_crop_phaseBD_candidates.csv", "python_log"
    ), config, offset_r=crop_r0, offset_c=crop_c0)

    # Attach exported diagnostics
    full_export = full_diag.set_index("object_id")
    crop_export = crop_diag.reset_index(drop=True)
    crop_export["object_id"] = crop_export["object_id"].astype(int)
    crop_export = crop_export.set_index("object_id")

    full_obj_df["object_id"] = -1
    for oid in set(full_in_crop_ids) | set(full_in_dense_ids):
        mrow = cent_full.loc[oid]
        dists = np.hypot(full_obj_df.centroid_row - mrow.initial_row, full_obj_df.centroid_col - mrow.initial_col)
        idx = int(dists.idxmin())
        full_obj_df.loc[idx, "object_id"] = oid
        if oid in full_export.index:
            full_obj_df.loc[idx, "path"] = full_export.loc[oid, "path"]
            full_obj_df.loc[idx, "n_accepted_fit_ok"] = full_export.loc[oid, "n_accepted_fit_ok"]
            full_obj_df.loc[idx, "peak_source"] = full_export.loc[oid].get("peak_source")
            full_obj_df.loc[idx, "n_filtered_local_maxima"] = full_export.loc[oid, "n_filtered_local_maxima"]
            full_obj_df.loc[idx, "local_peak_recovery_attempted"] = full_export.loc[oid].get(
                "local_peak_recovery_attempted"
            )
            full_obj_df.loc[idx, "local_peak_recovery_filtered_count"] = full_export.loc[oid].get(
                "local_peak_recovery_filtered_count"
            )

    cent_crop = crop_meas.groupby("object_id").first()[["initial_row", "initial_col"]]
    crop_in_bbox_ids = list(crop_diag["object_id"].astype(int))
    crop_obj_df["object_id"] = -1
    for oid in crop_in_bbox_ids:
        mrow = cent_crop.loc[oid]
        dists = np.hypot(
            crop_obj_df.centroid_row - (mrow.initial_row + crop_r0),
            crop_obj_df.centroid_col - (mrow.initial_col + crop_c0),
        )
        idx = int(dists.idxmin())
        crop_obj_df.loc[idx, "object_id"] = oid
        if oid in crop_export.index:
            crop_obj_df.loc[idx, "path"] = crop_export.loc[oid, "path"]
            crop_obj_df.loc[idx, "n_accepted_fit_ok"] = crop_export.loc[oid, "n_accepted_fit_ok"]
            crop_obj_df.loc[idx, "n_filtered_local_maxima"] = crop_export.loc[oid, "n_filtered_local_maxima"]

    matched_objects = _match_objects(
        full_obj_df[full_obj_df.object_id >= 0].copy(),
        crop_obj_df[crop_obj_df.object_id >= 0].copy(),
    )

    peak_diff = matched_objects[
        matched_objects["full_assigned_peaks"] != matched_objects["crop_assigned_peaks"]
    ].copy()
    one_vs_many = matched_objects[
        (matched_objects["full_assigned_peaks"] == 1) & (matched_objects["crop_assigned_peaks"] >= 2)
    ].copy()
    accepted_gain = matched_objects[matched_objects["crop_accepted"] > matched_objects["full_accepted"]].copy()

    # Dense ROI object-level comparison (full export vs simulated dense crop replay)
    dense_full_obj = full_obj_df[
        full_obj_df["object_id"].isin(full_in_dense_ids)
    ].copy()
    dense_labels, dense_objects = ConnectedObjectAnalyzer().analyze(dense_mask, dense_image)
    dense_replay = _object_rows(
        dense_labels,
        dense_objects,
        dense_image,
        dense_table,
        config,
        offset_r=dr0,
        offset_c=dc0,
    )
    dense_match = _match_objects(dense_full_obj, dense_replay)
    dense_peak_diff = dense_match[
        dense_match["full_assigned_peaks"] != dense_match["crop_assigned_peaks"]
    ].copy()
    dense_one_vs_many = dense_match[
        (dense_match["full_assigned_peaks"] == 1) & (dense_match["crop_assigned_peaks"] >= 2)
    ].copy()
    dense_accepted_gain = dense_match[
        dense_match["crop_accepted"] > dense_match["full_accepted"]
    ].copy()

    # C. Routing differences (matched crop bbox objects)
    routing = {
        "fast_single_to_suspicious_or_gmm": int(
            (
                (matched_objects["full_path"] == "fast_single")
                & (matched_objects["crop_path"].isin(["gmm", "single"]))
            ).sum()
        ),
        "one_peak_to_two_plus_assigned": int(len(one_vs_many)),
        "accepted_count_increases_in_crop": int(len(accepted_gain)),
        "path_changed_any": int((matched_objects["full_path"] != matched_objects["crop_path"]).sum()),
    }

    # Overlays
    _save_peak_overlay(
        full_image,
        r0=crop_r0,
        c0=crop_c0,
        r1=crop_r1,
        c1=crop_c1,
        full_peaks=full_in_crop,
        crop_peaks_full=crop_peaks_full,
        crop_only=crop_only,
        path=OUT_DIR / "crop_bbox_peak_overlay.png",
    )
    _save_peak_overlay(
        full_image,
        r0=dr0,
        c0=dc0,
        r1=dr1,
        c1=dc1,
        full_peaks=full_in_dense,
        crop_peaks_full=dense_peaks_full,
        crop_only=dense_crop_only,
        path=OUT_DIR / "dense_roi_peak_overlay.png",
    )

    report = {
        "crop_offset_full_coords": {"row0": crop_r0, "col0": crop_c0, "row1": crop_r1, "col1": crop_c1},
        "dense_roi_full_coords": {"row0": dr0, "col0": dc0, "row1": dr1, "col1": dc1},
        "note": (
            "On-disk crop.tif matches full image at (860,636)-(985,767), which is NOT the dense cluster "
            f"center (~404,342). A simulated dense subimage replay is included for the dense ROI."
        ),
        "A_peak_detection": {
            "crop_bbox": {
                "full_image_peaks_inside": len(full_in_crop),
                "crop_run_peaks_mapped": len(crop_peaks_full),
                "matched": matched,
                "full_only": len(full_only),
                "crop_only": len(crop_only),
                "crop_only_examples": [
                    {"row": p.row, "col": p.col, "intensity": p.intensity} for p in crop_only[:20]
                ],
            },
            "dense_roi_simulated_crop": {
                "full_image_peaks_inside": len(full_in_dense),
                "simulated_dense_crop_peaks": len(dense_peaks_full),
                "matched": dense_matched,
                "full_only": len(dense_full_only),
                "crop_only": len(dense_crop_only),
            },
        },
        "B_per_object_assignment": {
            "crop_bbox": {
                "matched_objects": len(matched_objects),
                "peak_count_differs": len(peak_diff),
                "full_assigned_1_crop_assigned_2plus": len(one_vs_many),
                "accepted_count_increases_in_crop": len(accepted_gain),
            },
            "dense_roi_simulated_crop": {
                "matched_objects": len(dense_match),
                "peak_count_differs": len(dense_peak_diff),
                "full_assigned_1_simulated_2plus": len(dense_one_vs_many),
                "accepted_count_increases_in_simulated_crop": len(dense_accepted_gain),
            },
        },
        "C_routing_differences": {
            "crop_bbox": routing,
            "dense_roi_simulated_crop": {
                "fast_single_to_suspicious_or_gmm": int(
                    (
                        (dense_match["full_path"] == "fast_single")
                        & (dense_match["crop_path"].isin(["gmm", "single"]))
                    ).sum()
                ),
                "one_peak_to_two_plus_assigned": int(len(dense_one_vs_many)),
                "accepted_count_increases": int(len(dense_accepted_gain)),
                "path_changed_any": int((dense_match["full_path"] != dense_match["crop_path"]).sum()),
            },
        },
        "D_detector_context": [asdict(x) for x in detector_context],
        "accepted_counts": {
            "full_in_crop_bbox_fit_ok": int(
                (
                    (full_meas.object_id.isin(full_in_crop_ids))
                    & (full_meas.accepted.fillna(False))
                    & (full_meas.fit_status == "fit_ok")
                ).sum()
            ),
            "crop_fit_ok": int(
                ((crop_meas.accepted.fillna(False)) & (crop_meas.fit_status == "fit_ok")).sum()
            ),
            "full_in_dense_roi_fit_ok": int(
                (
                    (full_meas.object_id.isin(full_in_dense_ids))
                    & (full_meas.accepted.fillna(False))
                    & (full_meas.fit_status == "fit_ok")
                ).sum()
            ),
        },
    }

    peak_diff.to_csv(OUT_DIR / "crop_bbox_objects_peak_count_diff.csv", index=False)
    one_vs_many.to_csv(OUT_DIR / "crop_bbox_full1_crop2plus_objects.csv", index=False)
    dense_peak_diff.to_csv(OUT_DIR / "dense_roi_objects_peak_count_diff.csv", index=False)
    dense_one_vs_many.to_csv(OUT_DIR / "dense_roi_full1_sim2plus_objects.csv", index=False)
    dense_match.to_csv(OUT_DIR / "dense_roi_object_comparison.csv", index=False)
    matched_objects.to_csv(OUT_DIR / "crop_bbox_object_comparison.csv", index=False)
    (OUT_DIR / "full_vs_crop_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
