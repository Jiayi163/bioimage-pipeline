#!/usr/bin/env python3
"""Replay Phase D diagnostics for real-image failure cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.phase_c_fallback import evaluate_phase_c_fallback
from bioimage_pipeline.puncta.types import PeakCandidate


def _paths(example: str) -> tuple[Path, Path, Path, Path]:
    if example == "example3":
        base = Path(r"C:\Users\Administrator\Desktop\example3")
        return (
            base / "input" / "MAX_10% BSA block with EV_0004.tif",
            base / "mask" / "MAX_10% BSA block with EV_0004_mask.tif",
            base / "phase_b_default" / "EV_0004_phaseB_object_diagnostics.csv",
            base / "phase_b_default" / "EV_0004_phaseB_measurements.csv",
        )
    base = Path(r"C:\Users\Administrator\Desktop\example1")
    return (
        base / "input" / "MAX_10% BSA block with EV_0001.tif",
        base / "mask" / "MAX_10% BSA block with EV_0001_mask.tif",
        base / "phase_b_on_true" / "EV_phaseB_on_true_object_diagnostics.csv",
        base / "phase_b_on_true" / "EV_phaseB_on_true_measurements.csv",
    )


def replay(example: str, object_id: int) -> dict[str, object]:
    image_path, mask_path, diag_csv, meas_csv = _paths(example)
    image = np.asarray(skio.imread(image_path), dtype=np.float64)
    mask = np.asarray(skio.imread(mask_path))
    if image.ndim > 2:
        image = image[..., 0]
    if mask.ndim > 2:
        mask = mask[..., 0]
    mask_bool = mask > 0

    analyzer = ConnectedObjectAnalyzer()
    labeled, objects = analyzer.analyze(mask_bool, image)

    diag = pd.read_csv(diag_csv)
    diag_row = diag.loc[diag["object_id"] == object_id]
    if diag_row.empty:
        raise ValueError(f"object_id {object_id} not found in diagnostics")
    diag_row = diag_row.iloc[0]

    meas = pd.read_csv(meas_csv)
    meas_row = meas.loc[meas["object_id"] == object_id]
    if meas_row.empty:
        raise ValueError(f"object_id {object_id} not found in measurements")
    peak_row = float(meas_row["initial_row"].iloc[0])
    peak_col = float(meas_row["initial_col"].iloc[0])
    mask_label = int(labeled[int(round(peak_row)), int(round(peak_col))])
    obj = next(o for o in objects if o.label == mask_label)
    object_mask = labeled == mask_label

    config = PunctaDeclumpConfig(threshold_method="external_mask")
    patch = build_object_patch(image, object_mask, obj, config)
    peak_result = MaximaDetector(config).detect(patch.raw if patch.raw is not None else image[patch.global_bbox[0]:patch.global_bbox[2], patch.global_bbox[1]:patch.global_bbox[3]], patch.object_mask)
    peaks = peak_result.filtered_peaks or peak_result.raw_peaks
    if not peaks:
        peaks = [
            PeakCandidate(
                row=obj.brightest_row,
                col=obj.brightest_col,
                intensity=obj.brightest_intensity,
            )
        ]

    selector = GaussianModelSelector(config)
    single = selector.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    comparison = selector.select_balanced_model(
        patch,
        peaks,
        single_component=single,
        n_filtered_peaks=len(peak_result.filtered_peaks),
        n_raw_peaks=len(peak_result.raw_peaks),
        obj=obj,
    )

    ambiguous = "ambiguous" in comparison.selection_reason
    ambiguous_reason = comparison.selection_reason if ambiguous else ""

    mixture = comparison.best_mixture
    selected_k = mixture.n_components if mixture is not None else 1
    accepted: list[dict[str, object]] = []
    rejection_counts: dict[str, int] = {}

    if mixture is not None and mixture.fit_succeeded:
        filt = CandidateFilter(config)
        candidates = filt.evaluate_mixture_components(
            obj,
            peaks,
            mixture,
            candidate_id_start=1,
            object_mask=patch.object_mask,
            patch=patch,
        )
        n_accepted = sum(1 for c in candidates if c.accepted)
        for candidate in candidates:
            if candidate.accepted:
                accepted.append(
                    {
                        "row": candidate.final_row,
                        "col": candidate.final_col,
                        "residual_relative": candidate.residual_relative,
                        "r_squared": candidate.r_squared,
                    }
                )
            elif candidate.rejection_reason:
                rejection_counts[candidate.rejection_reason] = (
                    rejection_counts.get(candidate.rejection_reason, 0) + 1
                )

        under_split = n_accepted < len(peak_result.filtered_peaks)
        fallback = evaluate_phase_c_fallback(
            config=config,
            obj=obj,
            single=single,
            selected=mixture,
            patch=patch,
            n_filtered_peaks=len(peak_result.filtered_peaks),
            n_accepted=n_accepted,
            under_split_suspect=under_split,
        )
        if fallback.trigger:
            comparison = selector.apply_phase_c_fallback_refinement(
                comparison,
                patch,
                peaks,
                trigger_reason=fallback.reason,
            )
            mixture = comparison.best_mixture
            if mixture is not None:
                selected_k = mixture.n_components
                filt = CandidateFilter(config)
                candidates = filt.evaluate_mixture_components(
                    obj,
                    peaks,
                    mixture,
                    candidate_id_start=1,
                    object_mask=patch.object_mask,
                    patch=patch,
                )
                accepted = []
                rejection_counts = {}
                for candidate in candidates:
                    if candidate.accepted:
                        accepted.append(
                            {
                                "row": candidate.final_row,
                                "col": candidate.final_col,
                                "residual_relative": candidate.residual_relative,
                                "r_squared": candidate.r_squared,
                            }
                        )
                    elif candidate.rejection_reason:
                        rejection_counts[candidate.rejection_reason] = (
                            rejection_counts.get(candidate.rejection_reason, 0) + 1
                        )
            ambiguous = "ambiguous" in comparison.selection_reason
            ambiguous_reason = comparison.selection_reason if ambiguous else ambiguous_reason
    else:
        selected = comparison.selected
        if hasattr(selected, "n_components"):
            selected_k = selected.n_components  # type: ignore[union-attr]
        filt = CandidateFilter(config)
        candidate = filt.evaluate_component(
            obj,
            peaks[0],
            single,
            candidate_id=1,
            component_id=1,
            path="single",
            object_mask=patch.object_mask,
            patch=patch,
        )
        if candidate.accepted:
            accepted.append(
                {
                    "row": candidate.final_row,
                    "col": candidate.final_col,
                    "residual_relative": candidate.residual_relative,
                    "r_squared": candidate.r_squared,
                }
            )
        elif candidate.rejection_reason:
            rejection_counts[candidate.rejection_reason] = 1

    mixture_r2 = mixture.r_squared if mixture is not None else single.r_squared
    mixture_resid = (
        mixture.residual_rmse / max(max(c.amplitude for c in mixture.components), 1.0)
        if mixture is not None and mixture.components
        else single.residual_relative
    )

    return {
        "example": example,
        "object_id": object_id,
        "mask_label": mask_label,
        "selected_k": selected_k,
        "accepted_count": len(accepted),
        "best_gmm_r_squared": mixture_r2,
        "best_gmm_residual_relative": mixture_resid,
        "ambiguous": ambiguous,
        "ambiguous_reason": ambiguous_reason,
        "n_raw_maxima": len(peak_result.raw_peaks),
        "n_filtered_maxima": len(peak_result.filtered_peaks),
        "rejection_counts": rejection_counts,
        "accepted_components": accepted,
        "selection_reason": comparison.selection_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=("example1", "example3"), required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = replay(args.example, args.object_id)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
