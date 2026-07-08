"""Export puncta declumping results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from bioimage_pipeline.export import (
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.puncta.types import DeclumpResult, PunctumCandidate


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


class ResultExporter:
    """Write CSV, JSON summary, seed image, and intermediate mask/label TIFFs."""

    def export_all(
        self,
        output_dir: str | Path,
        result: DeclumpResult,
        *,
        stem: str = "puncta",
        image_shape: tuple[int, int] | None = None,
    ) -> dict[str, Path]:
        """Export all puncta declumping artifacts."""
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

        return paths

    def export_csv(self, path: str | Path, result: DeclumpResult) -> Path:
        """Export punctum candidates as a CSV table."""
        rows = [self._candidate_row(candidate) for candidate in result.candidates]
        dataframe = pd.DataFrame(rows)
        export_measurements_csv(path, dataframe)
        return Path(path)

    def export_summary_json(self, path: str | Path, result: DeclumpResult) -> Path:
        """Export aggregate summary as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": asdict(result.summary),
            "threshold_metadata": result.threshold_metadata,
            "fit_quality": {
                "gaussian_fit_count": sum(1 for c in result.accepted if c.sigma is not None),
                "fallback_count": sum(1 for c in result.accepted if c.sigma is None),
                "median_sigma": _median_or_none(
                    [c.sigma for c in result.accepted if c.sigma is not None]
                ),
                "median_relative_residual": _median_or_none(
                    [
                        c.residual_rmse / max(c.amplitude, 1.0)
                        for c in result.accepted
                        if c.residual_rmse is not None and c.amplitude is not None
                    ]
                ),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def export_seed_image(
        self,
        path: str | Path,
        result: DeclumpResult,
        image_shape: tuple[int, int],
    ) -> Path:
        """Write accepted puncta centers as a uint16 seed label image."""
        seeds = np.zeros(image_shape, dtype=np.uint16)
        for index, candidate in enumerate(result.accepted, start=1):
            row = int(round(candidate.final_row))
            col = int(round(candidate.final_col))
            if 0 <= row < image_shape[0] and 0 <= col < image_shape[1]:
                seeds[row, col] = index
        return export_label_tiff(path, seeds)

    @staticmethod
    def _candidate_row(candidate: PunctumCandidate) -> dict[str, object]:
        return {
            "object_id": candidate.object_id,
            "candidate_id": candidate.candidate_id,
            "path": candidate.path,
            "initial_row": candidate.initial_row,
            "initial_col": candidate.initial_col,
            "fitted_row": candidate.fitted_row,
            "fitted_col": candidate.fitted_col,
            "final_row": candidate.final_row,
            "final_col": candidate.final_col,
            "center_shift": candidate.center_shift,
            "sigma": candidate.sigma,
            "width_fwhm": candidate.width_fwhm,
            "amplitude": candidate.amplitude,
            "background": candidate.background,
            "residual_rmse": candidate.residual_rmse,
            "residual_relative": (
                candidate.residual_rmse / max(candidate.amplitude, 1.0)
                if candidate.residual_rmse is not None and candidate.amplitude is not None
                else None
            ),
            "accepted": candidate.accepted,
            "rejection_reason": candidate.rejection_reason,
            "warning": candidate.warning,
        }
