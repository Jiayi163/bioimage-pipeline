"""Export puncta declumping results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from bioimage_pipeline.export import (
    export_intensity_tiff,
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import DeclumpResult, PunctumCandidate
from bioimage_pipeline.puncta.under_split_report import export_under_split_report
from bioimage_pipeline.puncta.fiji_export import FijiResultExporter


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _count_failure_categories(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        cat = str(row.get("failure_category", "unknown"))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


class ResultExporter:
    """Write CSV, JSON summary, seed image, and intermediate mask/label TIFFs."""

    def export_all(
        self,
        output_dir: str | Path,
        result: DeclumpResult,
        *,
        stem: str = "puncta",
        image_shape: tuple[int, int] | None = None,
        image: np.ndarray | None = None,
        config: PunctaDeclumpConfig | None = None,
        show_rejected: bool = True,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        paths["csv"] = self.export_csv(output_path / f"{stem}_measurements.csv", result)
        paths["summary"] = self.export_summary_json(
            output_path / f"{stem}_summary.json",
            result,
        )

        if image_shape is not None:
            paths["seeds"] = self.export_seed_image(
                output_path / f"{stem}_seeds.tif",
                result,
                image_shape,
            )

        if result.mask is not None:
            paths["mask"] = export_mask_tiff(output_path / f"{stem}_mask.tif", result.mask)
        if result.labels is not None:
            paths["labels"] = export_label_tiff(
                output_path / f"{stem}_labels.tif",
                result.labels,
            )

        if result.image_only_diagnostics is not None:
            paths.update(self._export_image_only_diagnostics(output_path, result, stem=stem))

        undersplit_paths = export_under_split_report(
            output_path,
            result,
            stem=stem,
            top_n=50,
        )
        paths.update(undersplit_paths)

        if image is not None and image_shape is not None and (
            config is None or config.export_fiji_tiffs
        ):
            fiji_exporter = FijiResultExporter(config)
            fiji_paths = fiji_exporter.export_all(
                output_path,
                result,
                image,
                stem=stem,
                show_rejected=show_rejected,
            )
            paths.update(fiji_paths)

        return paths

    def export_csv(self, path: str | Path, result: DeclumpResult) -> Path:
        rows = [self._candidate_row(candidate) for candidate in result.candidates]
        dataframe = pd.DataFrame(rows)
        export_measurements_csv(path, dataframe)
        return Path(path)

    def export_summary_json(self, path: str | Path, result: DeclumpResult) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gaussian_fitted = [c for c in result.accepted if c.fit_status == "fit_ok"]
        payload = {
            "summary": asdict(result.summary),
            "threshold_metadata": result.threshold_metadata,
            "gmm_config": result.threshold_metadata.get("gmm_config"),
            "fit_quality": {
                "gaussian_fit_count": len(gaussian_fitted),
                "fallback_count": sum(
                    1 for c in result.accepted if c.fit_status == "fit_failed_fallback"
                ),
                "median_sigma": _median_or_none(
                    [c.sigma for c in gaussian_fitted if c.sigma is not None]
                ),
                "median_relative_residual": _median_or_none(
                    [
                        c.residual_relative
                        for c in gaussian_fitted
                        if c.residual_relative is not None
                    ]
                ),
                "median_r_squared": _median_or_none(
                    [c.r_squared for c in gaussian_fitted if c.r_squared is not None]
                ),
            },
            "diagnostic_artifacts": result.diagnostic_artifacts,
            "under_split": {
                "suspect_objects": result.summary.under_split_suspect_objects,
                "top_report_n": len(result.under_split_report),
                "top_failure_categories": _count_failure_categories(result.under_split_report),
            },
            "timing": result.timing,
            "detector": {
                "name": result.peak_table.detector_name if result.peak_table else None,
                "method": result.peak_table.method if result.peak_table else None,
                "peak_count": len(result.peak_table.peaks) if result.peak_table else 0,
                "cache_hit": result.peak_table.cache_hit if result.peak_table else False,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return output_path

    def _export_image_only_diagnostics(
        self,
        output_path: Path,
        result: DeclumpResult,
        *,
        stem: str,
    ) -> dict[str, Path]:
        """Export image-only intermediate maps and peak/group JSON."""
        paths: dict[str, Path] = {}
        diag = result.image_only_diagnostics
        if diag is None:
            return paths

        if diag.background is not None:
            paths["background"] = export_intensity_tiff(
                output_path / f"{stem}_background.tif",
                diag.background.astype(np.float32),
            )
        if diag.corrected is not None:
            paths["corrected"] = export_intensity_tiff(
                output_path / f"{stem}_corrected.tif",
                diag.corrected.astype(np.float32),
            )
        if diag.signal_support is not None:
            paths["signal_support"] = export_mask_tiff(
                output_path / f"{stem}_signal_support.tif",
                diag.signal_support.astype(np.uint8) * 255,
            )

        peaks_payload = {
            "raw_peaks": [asdict(p) for p in diag.raw_peaks],
            "validated_peaks": [asdict(p) for p in diag.validated_peaks],
            "rejected_peaks": [asdict(p) for p in diag.rejected_peaks],
        }
        peaks_path = output_path / f"{stem}_image_only_peaks.json"
        peaks_path.write_text(json.dumps(peaks_payload, indent=2), encoding="utf-8")
        paths["image_only_peaks"] = peaks_path

        groups_payload = {
            "groups": [
                {
                    "group_id": group.group_id,
                    "route": group.route,
                    "bbox": group.bbox,
                    "peak_indices": list(group.peak_indices),
                    "peaks": [asdict(p) for p in group.peaks],
                    "min_pairwise_separation": group.min_pairwise_separation,
                }
                for group in diag.peak_groups
            ],
            "group_routes": diag.group_routes,
        }
        groups_path = output_path / f"{stem}_peak_groups.json"
        groups_path.write_text(json.dumps(groups_payload, indent=2), encoding="utf-8")
        paths["peak_groups"] = groups_path
        return paths

    def export_seed_image(
        self,
        path: str | Path,
        result: DeclumpResult,
        image_shape: tuple[int, int],
    ) -> Path:
        seeds = np.zeros(image_shape, dtype=np.uint16)
        index = 1
        for candidate in result.accepted:
            row = int(round(candidate.final_row))
            col = int(round(candidate.final_col))
            if 0 <= row < image_shape[0] and 0 <= col < image_shape[1]:
                seeds[row, col] = index
                index += 1
        return export_label_tiff(path, seeds)

    @staticmethod
    def _candidate_row(candidate: PunctumCandidate) -> dict[str, object]:
        return {
            "object_id": candidate.object_id,
            "component_id": candidate.component_id,
            "candidate_id": candidate.candidate_id,
            "path": candidate.path,
            "fit_status": candidate.fit_status,
            "initial_row": candidate.initial_row,
            "initial_col": candidate.initial_col,
            "fitted_row": candidate.fitted_row,
            "fitted_col": candidate.fitted_col,
            "y_fit": candidate.fitted_row,
            "x_fit": candidate.fitted_col,
            "final_row": candidate.final_row,
            "final_col": candidate.final_col,
            "center_shift": candidate.center_shift,
            "sigma": candidate.sigma,
            "sigma_row": candidate.sigma_row,
            "sigma_col": candidate.sigma_col,
            "sigma_x": candidate.sigma_col,
            "sigma_y": candidate.sigma_row,
            "width_fwhm": candidate.width_fwhm,
            "amplitude": candidate.amplitude,
            "background": candidate.background,
            "residual_rmse": candidate.residual_rmse,
            "residual_relative": candidate.residual_relative,
            "r_squared": candidate.r_squared,
            "model_score": candidate.model_score,
            "n_components_in_model": candidate.n_components_in_model,
            "accepted": candidate.accepted,
            "rejection_reason": candidate.rejection_reason,
            "warning": candidate.warning,
            "object_area": candidate.object_area,
            "object_equivalent_diameter": candidate.object_equivalent_diameter,
            "object_eccentricity": candidate.object_eccentricity,
            "object_solidity": candidate.object_solidity,
            "object_major_axis_length": candidate.object_major_axis_length,
            "object_minor_axis_length": candidate.object_minor_axis_length,
            "object_elongation": candidate.object_elongation,
            "gmm_trigger_reasons": candidate.gmm_trigger_reasons,
            "n_raw_local_maxima": candidate.n_raw_local_maxima,
            "n_filtered_local_maxima": candidate.n_filtered_local_maxima,
            "tried_gmm": candidate.tried_gmm,
            "gmm_candidate_components": candidate.gmm_candidate_components,
            "one_gaussian_r_squared": candidate.one_gaussian_r_squared,
            "one_gaussian_residual_relative": candidate.one_gaussian_residual_relative,
            "best_gmm_r_squared": candidate.best_gmm_r_squared,
            "best_gmm_residual_relative": candidate.best_gmm_residual_relative,
            "best_gmm_n_components": candidate.best_gmm_n_components,
            "model_selection_reason": candidate.model_selection_reason,
            "rejected_component_reason": candidate.rejected_component_reason,
            "under_split_suspect": candidate.under_split_suspect,
            "under_split_reasons": candidate.under_split_reasons,
            "gmm_winning_init_strategy": candidate.gmm_winning_init_strategy,
            "gmm_duplicate_threshold_px": candidate.gmm_duplicate_threshold_px,
            "gmm_duplicate_distance_px": candidate.gmm_duplicate_distance_px,
            "gmm_bic_delta_vs_single": candidate.gmm_bic_delta_vs_single,
            "gmm_aic_delta_vs_single": candidate.gmm_aic_delta_vs_single,
            "gmm_search_mode": candidate.gmm_search_mode,
            "gmm_spurious_split_rejected": candidate.gmm_spurious_split_rejected,
            "gmm_multi_start_attempts": candidate.gmm_multi_start_attempts,
            "gmm_multi_start_converged": candidate.gmm_multi_start_converged,
            "local_peak_recovery_attempted": candidate.local_peak_recovery_attempted,
            "local_peak_recovery_success": candidate.local_peak_recovery_success,
            "local_peak_recovery_raw_count": candidate.local_peak_recovery_raw_count,
            "local_peak_recovery_filtered_count": candidate.local_peak_recovery_filtered_count,
            "peak_source": candidate.peak_source,
            "detection_provenance": candidate.detection_provenance,
        }
