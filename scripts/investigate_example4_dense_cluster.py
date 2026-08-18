#!/usr/bin/env python3
"""Fast structural investigation of example4 dense central cluster (read-only)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io as skio
from skimage.draw import disk
from skimage.measure import find_contours

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.component_validity import detect_roi_saturation
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.phase_c_fallback import evaluate_phase_c_fallback
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult
from bioimage_pipeline.qc import normalize_for_display

BASE = Path(r"C:\Users\Administrator\Desktop\example4")
RUN_DIR = BASE / "phase_b_d"
IMAGE_PATH = BASE / "input" / "MAX_10% BSA block with EV_0007.tif"
MASK_PATH = BASE / "mask" / "MAX_10% BSA block with EV_0007_mask.tif"
OUT_DIR = RUN_DIR / "diagnostics" / "investigation" / "dense_cluster"

LARGE_AREA = 80
LARGE_DIAMETER = 10.0


def _parse_k(reason: str) -> tuple[int | None, int | None]:
    initial_k = final_k = None
    m = re.search(r"selected_gmm_n=(\d+)", reason)
    if m:
        initial_k = int(m.group(1))
        final_k = initial_k
    m2 = re.search(r"residual_split_applied_n=(\d+)->(\d+)", reason)
    if m2:
        final_k = int(m2.group(2))
    return initial_k, final_k


def _object_centroids(meas: pd.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        int(oid): (float(g.iloc[0]["initial_row"]), float(g.iloc[0]["initial_col"]))
        for oid, g in meas.groupby("object_id")
    }


def _find_dense_cluster_roi(
    image: np.ndarray,
    mask_bool: np.ndarray,
    labeled: np.ndarray,
    centroids: dict[int, tuple[float, float]],
) -> tuple[tuple[int, int, int, int], list[int]]:
    h, w = image.shape
    cy, cx = h / 2.0, w / 2.0
    corrected = np.clip(image - np.percentile(image[mask_bool], 10), 0, None)
    bright = corrected > np.percentile(corrected[mask_bool], 92)
    density = ndi.gaussian_filter(bright.astype(np.float64), sigma=12.0)
    yy, xx = np.indices((h, w))
    center_weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (min(h, w) * 0.22) ** 2))
    score = density * center_weight
    score[~mask_bool] = 0
    peak_row, peak_col = np.unravel_index(int(np.argmax(score)), score.shape)

    radius = int(round(min(h, w) * 0.12))
    er0, er1 = max(0, peak_row - radius), min(h, peak_row + radius + 1)
    ec0, ec1 = max(0, peak_col - radius), min(w, peak_col + radius + 1)

    # Map pipeline object_ids whose centroids fall in the dense ROI
    object_ids = [
        oid
        for oid, (cr, cc) in centroids.items()
        if er0 <= cr < er1 and ec0 <= cc < ec1
    ]

    # Expand bbox from centroid locations
    if object_ids:
        rows = [centroids[oid][0] for oid in object_ids]
        cols = [centroids[oid][1] for oid in object_ids]
        pad = 24
        r0, r1 = max(0, int(min(rows)) - pad), min(h, int(max(rows)) + pad + 1)
        c0, c1 = max(0, int(min(cols)) - pad), min(w, int(max(cols)) + pad + 1)
    else:
        r0, r1, c0, c1 = er0, er1, ec0, ec1

    return (r0, c0, r1, c1), sorted(object_ids)


def _record_from_export(
    oid: int,
    diag_row: pd.Series,
    meas_rows: pd.DataFrame,
    labeled: np.ndarray,
    image: np.ndarray,
    objects_by_label: dict,
) -> dict[str, object]:
    cr, cc = float(meas_rows.iloc[0]["initial_row"]), float(meas_rows.iloc[0]["initial_col"])
    mask_label = int(labeled[int(round(cr)), int(round(cc))])
    reason = str(diag_row.get("model_selection_reason", ""))
    initial_k, phase_b_k = _parse_k(reason)
    rejection_exported = Counter(
        meas_rows.loc[~meas_rows["accepted"].fillna(False), "rejection_reason"].dropna()
    )
    obj = objects_by_label.get(mask_label)
    sat_frac = None
    sat_present = None
    if obj is not None:
        config = PunctaDeclumpConfig(threshold_method="external_mask")
        patch = build_object_patch(image, labeled == mask_label, obj, config)
        sat = detect_roi_saturation(
            patch,
            near_clip_margin=config.saturation_near_clip_margin,
            near_clip_fraction_threshold=config.saturation_near_clip_fraction,
        )
        sat_frac = sat.near_clip_fraction
        sat_present = sat.present

    area = float(diag_row["area"])
    eq_d = float(diag_row["equivalent_diameter"])
    return {
        "object_id": oid,
        "mask_label": mask_label,
        "centroid_row": cr,
        "centroid_col": cc,
        "area": area,
        "equivalent_diameter": eq_d,
        "is_large_merged": area >= LARGE_AREA or eq_d >= LARGE_DIAMETER,
        "n_raw_local_maxima": int(diag_row["n_raw_local_maxima"]),
        "n_filtered_local_maxima": int(diag_row["n_filtered_local_maxima"]),
        "path": str(diag_row["path"]),
        "tried_gmm": bool(diag_row["tried_gmm"]),
        "initial_selected_k": initial_k,
        "phase_b_final_k": phase_b_k,
        "n_accepted": int(diag_row["n_accepted_fit_ok"]),
        "n_candidates": int(diag_row["n_candidates"]),
        "n_rejected": int(diag_row["n_candidates"]) - int(diag_row["n_accepted_fit_ok"]),
        "rejection_reasons": dict(rejection_exported),
        "one_gaussian_r_squared": float(diag_row["one_gaussian_r_squared"])
        if pd.notna(diag_row["one_gaussian_r_squared"])
        else None,
        "best_gmm_r_squared": float(diag_row["best_gmm_r_squared"])
        if pd.notna(diag_row["best_gmm_r_squared"])
        else None,
        "best_gmm_residual_relative": float(diag_row["best_gmm_residual_relative"])
        if pd.notna(diag_row["best_gmm_residual_relative"])
        else None,
        "under_split_suspect": bool(diag_row["under_split_suspect"]),
        "model_selection_reason": reason,
        "saturation_present": sat_present,
        "saturation_near_clip_fraction": sat_frac,
        "phase_c_fallback_would_trigger": None,
        "phase_c_fallback_reason": None,
    }


def _estimate_phase_c_fallback(
    record: dict[str, object],
    diag_row: pd.Series,
    image: np.ndarray,
    labeled: np.ndarray,
    objects_by_label: dict,
) -> None:
    """Lightweight fallback gate using exported counts + saturation (no full GMM replay)."""
    if not record["tried_gmm"]:
        record["phase_c_fallback_would_trigger"] = False
        record["phase_c_fallback_reason"] = "not_gmm"
        return
    n_filtered = int(record["n_filtered_local_maxima"])
    n_accepted = int(record["n_accepted"])
    under_split = bool(record["under_split_suspect"])
    if n_filtered < 2 or not under_split:
        record["phase_c_fallback_would_trigger"] = False
        record["phase_c_fallback_reason"] = "no_unresolved_multiplicity"
        return
    # Conservative: unresolved multiplicity + under-split evidence from export
    evidence = []
    if float(record["equivalent_diameter"]) > 7.0:
        evidence.append("large_diameter")
    if float(record["area"]) > 58.0:
        evidence.append("large_area")
    if record["best_gmm_residual_relative"] and record["best_gmm_residual_relative"] > 0.18:
        evidence.append("high_mixture_residual")
    if n_accepted == 0 and n_filtered >= 2:
        evidence.append("multiplicity_gap")
    record["phase_c_fallback_would_trigger"] = len(evidence) >= 1
    record["phase_c_fallback_reason"] = ";".join(evidence) if evidence else "insufficient_evidence"


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    disp = normalize_for_display(gray)
    return np.stack([disp, disp, disp], axis=-1).astype(np.uint8)


def _draw_cross(rgb: np.ndarray, row: float, col: float, color: tuple[int, int, int], size: int = 3) -> None:
    h, w = rgb.shape[:2]
    r, c = int(round(row)), int(round(col))
    for dr in range(-size, size + 1):
        for dc in range(-size, size + 1):
            if abs(dr) == size or abs(dc) == size:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    rgb[rr, cc] = color


def _save_panels(
    image: np.ndarray,
    labeled: np.ndarray,
    roi: tuple[int, int, int, int],
    object_ids: list[int],
    centroids: dict[int, tuple[float, float]],
    meas: pd.DataFrame,
) -> None:
    r0, c0, r1, c1 = roi
    roi_image = image[r0:r1, c0:c1]
    roi_labels = labeled[r0:r1, c0:c1]
    base = _to_rgb(roi_image)
    mask_panel = np.stack([roi_labels > 0] * 3, axis=-1).astype(np.uint8) * 255
    bound_panel = base.copy()
    for lbl in np.unique(roi_labels):
        if lbl <= 0:
            continue
        for contour in find_contours(roi_labels == lbl, 0.5):
            for rr, cc in contour:
                rri, cci = int(round(rr)), int(round(cc))
                if 0 <= rri < bound_panel.shape[0] and 0 <= cci < bound_panel.shape[1]:
                    bound_panel[rri, cci] = (0, 255, 255)
    max_panel = base.copy()
    for oid in object_ids:
        cr, cc = centroids[oid]
        if r0 <= cr < r1 and c0 <= cc < c1:
            _draw_cross(max_panel, cr - r0, cc - c0, (0, 255, 0))
    ids_panel = base.copy()
    palette = [(255, 80, 80), (80, 255, 80), (80, 120, 255), (255, 200, 80), (255, 80, 255)]
    for i, oid in enumerate(object_ids):
        cr, cc = centroids[oid]
        ml = int(labeled[int(round(cr)), int(round(cc))])
        tint = np.array(palette[i % len(palette)], dtype=np.float64)
        mask = roi_labels == ml
        ids_panel[mask] = np.clip(ids_panel[mask].astype(np.float64) * 0.55 + tint * 0.45, 0, 255).astype(np.uint8)
        if r0 <= cr < r1 and c0 <= cc < c1:
            _draw_cross(ids_panel, cr - r0, cc - c0, (255, 255, 0), size=2)
    acc_panel = base.copy()
    for _, row in meas[meas["accepted"].fillna(False)].iterrows():
        oid = int(row["object_id"])
        if oid not in object_ids:
            continue
        fr, fc = float(row["final_row"]), float(row["final_col"])
        if r0 <= fr < r1 and c0 <= fc < c1:
            rr_arr, cc_arr = disk((int(round(fr - r0)), int(round(fc - c0))), 3, shape=acc_panel.shape[:2])
            acc_panel[rr_arr, cc_arr] = (0, 255, 0)
    for _, row in meas[~meas["accepted"].fillna(False)].iterrows():
        oid = int(row["object_id"])
        if oid not in object_ids:
            continue
        fr = float(row.get("final_row", row["initial_row"]))
        fc = float(row.get("final_col", row["initial_col"]))
        if r0 <= fr < r1 and c0 <= fc < c1:
            _draw_cross(acc_panel, fr - r0, fc - c0, (255, 255, 0), size=2)
    panel = np.concatenate(
        [np.concatenate([base, mask_panel, bound_panel], axis=1), np.concatenate([max_panel, ids_panel, acc_panel], axis=1)],
        axis=0,
    )
    skio.imsave(OUT_DIR / "dense_cluster_roi_panels.png", panel)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = np.asarray(skio.imread(IMAGE_PATH), dtype=np.float64)
    mask = np.asarray(skio.imread(MASK_PATH))
    if image.ndim > 2:
        image = image[..., 0]
    if mask.ndim > 2:
        mask = mask[..., 0]
    mask_bool = mask > 0
    diag = pd.read_csv(RUN_DIR / "EV_0007_phaseBD_object_diagnostics.csv")
    meas = pd.read_csv(RUN_DIR / "EV_0007_phaseBD_measurements.csv")

    labeled, objects = ConnectedObjectAnalyzer().analyze(mask_bool, image)
    objects_by_label = {o.label: o for o in objects}
    centroids = _object_centroids(meas)
    roi, object_ids = _find_dense_cluster_roi(image, mask_bool, labeled, centroids)

    records = []
    for oid in object_ids:
        drow = diag.loc[diag["object_id"] == oid]
        mrows = meas.loc[meas["object_id"] == oid]
        if drow.empty or mrows.empty:
            continue
        rec = _record_from_export(oid, drow.iloc[0], mrows, labeled, image, objects_by_label)
        _estimate_phase_c_fallback(rec, drow.iloc[0], image, labeled, objects_by_label)
        records.append(rec)

    records.sort(key=lambda r: (-float(r["is_large_merged"]), -float(r["area"])))
    table = pd.DataFrame(records)
    table.to_csv(OUT_DIR / "dense_cluster_object_table.csv", index=False)
    _save_panels(image, labeled, roi, object_ids, centroids, meas)

    large = [r for r in records if r["is_large_merged"]]
    merged_blob = any(float(r["area"]) >= 120 for r in records)
    missing_maxima = any(r["n_filtered_local_maxima"] == 0 and r["is_large_merged"] for r in records)
    k_cap = any(
        (r.get("phase_b_final_k") or 0) >= 3 and r["n_filtered_local_maxima"] > r["n_accepted"] for r in large
    )
    filter_reject = any(r["n_accepted"] == 0 and r["tried_gmm"] for r in large)
    saturation = any(r.get("saturation_present") for r in large)

    report = {
        "roi_bbox_row_col": {"r0": roi[0], "c0": roi[1], "r1": roi[2], "c1": roi[3]},
        "image_shape": list(image.shape),
        "n_objects_in_roi": len(object_ids),
        "object_ids": object_ids,
        "large_merged_object_ids": [r["object_id"] for r in large],
        "failure_hypothesis": {
            "A_merged_large_connected_object": merged_blob,
            "B_local_maxima_missing": missing_maxima,
            "C_k_cap_too_small": k_cap,
            "D_fitting_window_too_large": True,
            "E_candidate_filter_rejection": filter_reject,
            "F_saturation": saturation,
            "G_combination": sum([merged_blob, k_cap, filter_reject, saturation]) >= 2,
        },
        "architecture_answer": {
            "one_small_k_mixture_per_connected_blob": True,
            "recommend_dense_prepartitioning": merged_blob and k_cap,
        },
        "objects": records,
    }
    (OUT_DIR / "dense_cluster_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "objects"}, indent=2))
    print(f"Wrote panels + table to {OUT_DIR}")


if __name__ == "__main__":
    main()
