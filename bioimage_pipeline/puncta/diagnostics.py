"""Diagnostic exports for suspicious puncta fits and under-split objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io as skio

from bioimage_pipeline.puncta.types import (
    GaussianComponent,
    MixtureFitResult,
    ModelSelectionDebug,
    ObjectPatch,
    PeakCandidate,
    PeakDetectionResult,
    PunctumCandidate,
)
from bioimage_pipeline.qc import normalize_for_display


def should_export_diagnostic(candidate: PunctumCandidate, threshold: float) -> bool:
    if candidate.fit_status != "fit_ok":
        return True
    if candidate.residual_relative is not None and candidate.residual_relative > threshold:
        return True
    if candidate.r_squared is not None and candidate.r_squared < 0.5:
        return True
    if candidate.under_split_suspect:
        return True
    return False


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _draw_peaks(
    rgb: np.ndarray,
    peaks: list[PeakCandidate],
    *,
    row_offset: int,
    col_offset: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = rgb.copy()
    height, width = out.shape[:2]
    for peak in peaks:
        row = int(round(peak.row - row_offset))
        col = int(round(peak.col - col_offset))
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out[rr, cc] = color
        if 0 <= row < height and 0 <= col < width:
            out[row, col] = color
    return out


def _panel(image: np.ndarray) -> np.ndarray:
    return _to_rgb(normalize_for_display(image))


def export_object_diagnostic(
    output_dir: str | Path,
    *,
    object_id: int,
    patch: ObjectPatch,
    candidate: PunctumCandidate,
    mixture: MixtureFitResult | None = None,
) -> Path:
    """Save a 3-panel diagnostic PNG: corrected patch, predicted/fit, residual."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"object_{object_id:04d}_comp_{candidate.component_id}_diagnostic.png"

    corrected = normalize_for_display(patch.corrected)
    if mixture is not None and mixture.predicted_patch is not None:
        predicted = normalize_for_display(mixture.predicted_patch)
    elif candidate.fitted_row is not None:
        predicted = corrected
    else:
        predicted = np.zeros_like(corrected)

    if mixture is not None and mixture.residual_patch is not None:
        residual = normalize_for_display(np.abs(mixture.residual_patch))
    else:
        residual = np.zeros_like(corrected)

    figure = np.concatenate(
        [
            _to_rgb(corrected),
            _to_rgb(predicted),
            _to_rgb(residual),
        ],
        axis=1,
    )
    skio.imsave(filename, figure.astype(np.uint8), check_contrast=False)
    return filename


def export_under_split_diagnostic(
    output_dir: str | Path,
    *,
    object_id: int,
    patch: ObjectPatch,
    peak_detection: PeakDetectionResult | None,
    single: GaussianComponent | None,
    mixture: MixtureFitResult | None,
    debug: ModelSelectionDebug,
) -> Path:
    """
    Multi-panel under-split diagnostic:
    raw | corrected | mask | raw maxima | filtered maxima |
    one-Gaussian pred | one-Gaussian residual | two-Gaussian pred | two-Gaussian residual
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"object_{object_id:04d}_undersplit.png"

    raw = patch.raw if patch.raw is not None else patch.corrected + patch.background_level
    panels: list[np.ndarray] = [
        _panel(raw),
        _panel(patch.corrected),
        _panel(patch.object_mask.astype(np.float64)),
    ]

    raw_peaks = peak_detection.raw_peaks if peak_detection else []
    filtered_peaks = peak_detection.filtered_peaks if peak_detection else []
    panels.append(
        _draw_peaks(
            _panel(patch.corrected),
            raw_peaks,
            row_offset=patch.row_offset,
            col_offset=patch.col_offset,
            color=(255, 64, 64),
        )
    )
    panels.append(
        _draw_peaks(
            _panel(patch.corrected),
            filtered_peaks,
            row_offset=patch.row_offset,
            col_offset=patch.col_offset,
            color=(64, 255, 64),
        )
    )

    if single is not None and single.predicted_patch is not None:
        panels.append(_panel(single.predicted_patch))
    else:
        panels.append(_panel(np.zeros_like(patch.corrected)))

    if single is not None and single.residual_patch is not None:
        panels.append(_panel(np.abs(single.residual_patch)))
    else:
        panels.append(_panel(np.zeros_like(patch.corrected)))

    if mixture is not None and mixture.predicted_patch is not None:
        panels.append(_panel(mixture.predicted_patch))
    else:
        panels.append(_panel(np.zeros_like(patch.corrected)))

    if mixture is not None and mixture.residual_patch is not None:
        panels.append(_panel(np.abs(mixture.residual_patch)))
    else:
        panels.append(_panel(np.zeros_like(patch.corrected)))

    # Pad panels to same height/width if needed (should already match).
    figure = np.concatenate(panels, axis=1)

    # Append a thin text-like barcode of reason length is not useful; write sidecar txt.
    reason_path = output_path / f"object_{object_id:04d}_undersplit_reason.txt"
    lines = [
        f"object_id={object_id}",
        f"gmm_trigger_reasons={';'.join(debug.gmm_trigger_reasons)}",
        f"n_raw_local_maxima={debug.n_raw_local_maxima}",
        f"n_filtered_local_maxima={debug.n_filtered_local_maxima}",
        f"tried_gmm={debug.tried_gmm}",
        f"gmm_candidate_components={debug.gmm_candidate_components}",
        f"one_gaussian_r_squared={debug.one_gaussian_r_squared}",
        f"one_gaussian_residual_relative={debug.one_gaussian_residual_relative}",
        f"one_gaussian_sigma={debug.one_gaussian_sigma}",
        f"best_gmm_r_squared={debug.best_gmm_r_squared}",
        f"best_gmm_residual_relative={debug.best_gmm_residual_relative}",
        f"best_gmm_n_components={debug.best_gmm_n_components}",
        f"model_selection_reason={debug.model_selection_reason}",
        f"rejected_component_reason={debug.rejected_component_reason}",
        f"single_path_reason={debug.single_path_reason}",
        f"under_split_reasons={';'.join(debug.under_split_reasons)}",
        "",
        "panel_order: raw | corrected | mask | raw_maxima(red) | filtered_maxima(green) | "
        "one_gauss_pred | one_gauss_resid | gmm_pred | gmm_resid",
    ]
    reason_path.write_text("\n".join(lines), encoding="utf-8")

    skio.imsave(filename, figure.astype(np.uint8), check_contrast=False)
    return filename
