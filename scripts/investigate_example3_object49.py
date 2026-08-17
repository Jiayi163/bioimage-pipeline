"""Investigate example3 object 49 failure (read-only diagnostic; no pipeline changes)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.types import ObjectPatch
from bioimage_pipeline.qc import normalize_for_display

OBJECT_ID = 49
RUN_DIR = Path(r"C:\Users\Administrator\Desktop\example3\phase_b_default")
IMAGE_PATH = Path(r"C:\Users\Administrator\Desktop\example3\input\MAX_10% BSA block with EV_0004.tif")
MASK_PATH = Path(r"C:\Users\Administrator\Desktop\example3\mask\MAX_10% BSA block with EV_0004_mask.tif")
OUT_DIR = RUN_DIR / "diagnostics" / "investigation"


def _draw_centers(
    rgb: np.ndarray,
    rows: list[float],
    cols: list[float],
    color: tuple[int, int, int],
    *,
    offset_row: float = 0.0,
    offset_col: float = 0.0,
) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for row, col in zip(rows, cols):
        r = int(round(row - offset_row))
        c = int(round(col - offset_col))
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image = np.asarray(skio.imread(IMAGE_PATH), dtype=np.float64)
    mask = np.asarray(skio.imread(MASK_PATH))
    if image.ndim > 2:
        image = image[..., 0]
    if mask.ndim > 2:
        mask = mask[..., 0]
    mask_bool = mask > 0

    analyzer = ConnectedObjectAnalyzer()
    labeled, objects = analyzer.analyze(mask_bool, image)

    meas = pd.read_csv(RUN_DIR / "EV_0004_phaseB_measurements.csv")
    m49 = meas[meas["object_id"] == OBJECT_ID]
    peak_row = float(m49["initial_row"].iloc[0])
    peak_col = float(m49["initial_col"].iloc[0])
    mask_label = int(labeled[int(round(peak_row)), int(round(peak_col))])
    obj = next(o for o in objects if o.label == mask_label)

    minr, minc, maxr, maxc = obj.bbox
    pad = 8
    r0, c0 = max(0, minr - pad), max(0, minc - pad)
    r1, c1 = min(image.shape[0], maxr + pad), min(image.shape[1], maxc + pad)
    roi_raw = image[r0:r1, c0:c1]
    roi_mask = labeled[r0:r1, c0:c1] == mask_label

    roi_pixels = roi_raw[roi_mask]
    image_max = float(image.max())
    roi_max = float(roi_pixels.max()) if roi_pixels.size else 0.0
    roi_p99 = float(np.percentile(roi_pixels, 99)) if roi_pixels.size else 0.0
    at_image_max = int((roi_pixels >= image_max - 1).sum()) if roi_pixels.size else 0
    at_99pct_image = int((roi_pixels >= 0.99 * image_max).sum()) if roi_pixels.size else 0
    saturation_frac = at_image_max / max(roi_pixels.size, 1)

    diag = pd.read_csv(RUN_DIR / "EV_0004_phaseB_object_diagnostics.csv")
    row = diag.loc[diag["object_id"] == OBJECT_ID].iloc[0]
    meas = pd.read_csv(RUN_DIR / "EV_0004_phaseB_measurements.csv")
    m49 = meas[meas["object_id"] == OBJECT_ID]
    background = float(m49["background"].iloc[0])

    config = PunctaDeclumpConfig()
    corrected = np.clip(roi_raw - background, 0.0, None)
    patch = ObjectPatch(
        object_id=OBJECT_ID,
        row_offset=r0,
        col_offset=c0,
        corrected=corrected,
        object_mask=roi_mask,
        background_level=background,
        global_bbox=(r0, c0, r1, c1),
        raw=roi_raw,
    )

    peak_result = MaximaDetector(config).detect(roi_raw, roi_mask)
    peaks = peak_result.filtered_peaks or peak_result.raw_peaks

    selector = GaussianModelSelector(config)
    primary = peaks[0]
    single = selector.mixture_fitter.single_fitter.fit_peak(
        patch, primary, component_id=1, n_components_in_model=1
    )
    comparison = selector.select_balanced_model(
        patch,
        peaks,
        single_component=single,
        n_filtered_peaks=len(peak_result.filtered_peaks),
        n_raw_peaks=len(peak_result.raw_peaks),
        obj=obj,
    )
    mixture = comparison.best_mixture

    report = {
        "object_id": OBJECT_ID,
        "mask_label_at_peak": mask_label,
        "area": float(row["area"]),
        "equivalent_diameter": float(row["equivalent_diameter"]),
        "eccentricity": float(row["eccentricity"]),
        "elongation": float(row["elongation"]),
        "solidity": float(row["solidity"]),
        "major_axis_length": float(row["major_axis_length"]),
        "minor_axis_length": float(row["minor_axis_length"]),
        "n_raw_local_maxima": int(row["n_raw_local_maxima"]),
        "n_filtered_local_maxima": int(row["n_filtered_local_maxima"]),
        "n_raw_local_maxima_replay": len(peak_result.raw_peaks),
        "n_filtered_local_maxima_replay": len(peak_result.filtered_peaks),
        "path": str(row["path"]),
        "routing": "suspicious (GMM path)" if row["path"] == "gmm" else "ordinary (fast_single)",
        "gmm_trigger_reasons": str(row["gmm_trigger_reasons"]),
        "tried_gmm": bool(row["tried_gmm"]),
        "single_gaussian_fitted": True,
        "gmm_attempted": bool(row["tried_gmm"]),
        "selected_k_exported": int(m49["best_gmm_n_components"].iloc[0]),
        "selected_k_replay": int(mixture.n_components) if mixture else 1,
        "replay_candidate_counts": comparison.candidate_component_counts,
        "replay_selection_reason": comparison.selection_reason,
        "n_accepted_fit_ok": int(row["n_accepted_fit_ok"]),
        "one_gaussian_r_squared": float(row["one_gaussian_r_squared"]),
        "one_gaussian_residual_relative": float(row["one_gaussian_residual_relative"]),
        "best_gmm_r_squared": float(row["best_gmm_r_squared"]),
        "best_gmm_residual_relative": float(row["best_gmm_residual_relative"]),
        "under_split_suspect": bool(row["under_split_suspect"]),
        "under_split_reasons": str(row["under_split_reasons"]),
        "model_selection_reason": str(row["model_selection_reason"]),
        "component_rejections": m49[
            ["component_id", "fit_status", "rejection_reason", "amplitude", "residual_rmse", "residual_relative"]
        ].to_dict("records"),
        "saturation": {
            "image_max": image_max,
            "roi_max": roi_max,
            "roi_p99": roi_p99,
            "pixels_at_image_max": at_image_max,
            "pixels_at_99pct_image_max": at_99pct_image,
            "roi_pixel_count": int(roi_pixels.size),
            "fraction_at_image_max": saturation_frac,
            "fraction_pixels_ge_10000": float((roi_pixels >= 10000).sum() / max(roi_pixels.size, 1)),
            "roi_p50": float(np.percentile(roi_pixels, 50)) if roi_pixels.size else 0.0,
            "likely_saturated": at_image_max >= 1 or float((roi_pixels >= 0.99 * image_max).sum()) >= 3,
        },
    }
    report_path = OUT_DIR / f"object_{OBJECT_ID:04d}_investigation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    raw_disp = normalize_for_display(roi_raw)
    mask_disp = roi_mask.astype(np.uint8) * 255
    rgb_raw = np.stack([raw_disp, raw_disp, raw_disp], axis=-1)
    rgb_mask = np.stack([mask_disp, mask_disp, mask_disp], axis=-1)

    panel_raw_max = _draw_centers(
        rgb_raw,
        [p.row for p in peak_result.raw_peaks],
        [p.col for p in peak_result.raw_peaks],
        (255, 0, 0),
    )
    panel_filt_max = _draw_centers(
        rgb_raw,
        [p.row for p in peak_result.filtered_peaks],
        [p.col for p in peak_result.filtered_peaks],
        (255, 0, 0),
    )

    accepted = m49[m49["accepted"] == True]  # noqa: E712
    rejected = m49[m49["accepted"] == False]  # noqa: E712
    panel_centers = rgb_raw.copy()
    panel_centers = _draw_centers(
        panel_centers,
        accepted["fitted_row"].tolist(),
        accepted["fitted_col"].tolist(),
        (0, 255, 0),
        offset_row=r0,
        offset_col=c0,
    )
    panel_centers = _draw_centers(
        panel_centers,
        rejected["fitted_row"].tolist(),
        rejected["fitted_col"].tolist(),
        (255, 128, 0),
        offset_row=r0,
        offset_col=c0,
    )

    pred = (
        normalize_for_display(mixture.predicted_patch)
        if mixture and mixture.predicted_patch is not None
        else np.zeros_like(raw_disp)
    )
    resid = (
        normalize_for_display(np.abs(mixture.residual_patch))
        if mixture and mixture.residual_patch is not None
        else np.zeros_like(raw_disp)
    )
    rgb_pred = np.stack([pred, pred, pred], axis=-1)
    rgb_resid = np.stack([resid, resid, resid], axis=-1)

    figure = np.concatenate(
        [rgb_raw, rgb_mask, panel_raw_max, panel_filt_max, panel_centers, rgb_pred, rgb_resid],
        axis=1,
    )
    out_png = OUT_DIR / f"object_{OBJECT_ID:04d}_investigation_panels.png"
    skio.imsave(out_png, figure.astype(np.uint8), check_contrast=False)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
