"""Orchestrator for size-gated puncta declumping."""

from __future__ import annotations

import numpy as np

from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.gaussian_fitter import GaussianFitter2D
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.threshold_mask import ThresholdMaskGenerator
from bioimage_pipeline.puncta.types import (
    DeclumpResult,
    DeclumpSummary,
    ObjectInfo,
    PeakCandidate,
    PunctumCandidate,
)


class PunctaDeclumpPipeline:
    """Run size-gated local maxima detection and Gaussian fitting."""

    def __init__(self, config: PunctaDeclumpConfig | None = None) -> None:
        self.config = config or PunctaDeclumpConfig()
        self.mask_generator = ThresholdMaskGenerator(self.config)
        self.object_analyzer = ConnectedObjectAnalyzer()
        self.maxima_detector = MaximaDetector(self.config)
        self.fitter = GaussianFitter2D(self.config)
        self.filter = CandidateFilter(self.config)

    def run(
        self,
        image: np.ndarray,
        *,
        external_mask: np.ndarray | None = None,
    ) -> DeclumpResult:
        """Execute the full puncta declumping pipeline."""
        image_arr = np.asarray(image)
        if image_arr.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported")

        mask, threshold_metadata = self.mask_generator.generate(
            image_arr,
            external_mask=external_mask,
        )
        labels, objects = self.object_analyzer.analyze(mask, image_arr)

        self.filter.reset()
        candidates: list[PunctumCandidate] = []
        candidate_counter = 0
        small_count = 0
        large_count = 0
        fallback_count = 0

        for obj in objects:
            object_mask = labels == obj.label
            if obj.equivalent_diameter <= self.config.single_spot_max_diameter:
                small_count += 1
                candidate_counter += 1
                peak = PeakCandidate(
                    row=obj.brightest_row,
                    col=obj.brightest_col,
                    intensity=obj.brightest_intensity,
                )
                fit = self.fitter.fit(image_arr, peak)
                candidate = self.filter.evaluate(
                    obj,
                    peak,
                    fit,
                    candidate_id=candidate_counter,
                    path="single",
                    object_mask=object_mask,
                )
                if (
                    not candidate.accepted
                    and self.config.accept_brightest_on_fit_failure
                ):
                    candidate = self.filter.accept_without_fit(
                        obj,
                        peak,
                        candidate_id=candidate_counter,
                        path="single",
                        warning="fit_failed_used_brightest_pixel",
                    )
                candidates.append(candidate)
            else:
                large_count += 1
                peaks = self.maxima_detector.find_in_full_image(
                    image_arr,
                    mask,
                    obj.bbox,
                )
                object_candidates: list[PunctumCandidate] = []
                for peak in peaks:
                    candidate_counter += 1
                    fit = self.fitter.fit(image_arr, peak)
                    candidate = self.filter.evaluate(
                        obj,
                        peak,
                        fit,
                        candidate_id=candidate_counter,
                        path="declump",
                        object_mask=object_mask,
                    )
                    object_candidates.append(candidate)

                accepted_for_object = [c for c in object_candidates if c.accepted]
                if not accepted_for_object:
                    fallback_count += 1
                    candidate_counter += 1
                    peak = PeakCandidate(
                        row=obj.brightest_row,
                        col=obj.brightest_col,
                        intensity=obj.brightest_intensity,
                    )
                    fit = self.fitter.fit(image_arr, peak)
                    fallback_candidate = self.filter.evaluate(
                        obj,
                        peak,
                        fit,
                        candidate_id=candidate_counter,
                        path="fallback",
                        object_mask=object_mask,
                    )
                    if not fallback_candidate.accepted:
                        fallback_candidate = self.filter.accept_without_fit(
                            obj,
                            peak,
                            candidate_id=candidate_counter,
                            path="fallback",
                            warning="no_maxima_survived_used_brightest_pixel",
                        )
                    object_candidates.append(fallback_candidate)

                candidates.extend(object_candidates)

        summary = DeclumpSummary(
            total_mask_objects=len(objects),
            small_single_objects=small_count,
            large_clumped_objects=large_count,
            total_candidates=len(candidates),
            total_accepted=len(self.filter.accepted),
            total_rejected=len(candidates) - len(self.filter.accepted),
            fallback_objects=fallback_count,
        )

        return DeclumpResult(
            candidates=candidates,
            summary=summary,
            mask=mask,
            labels=labels,
            objects=objects,
            threshold_metadata=threshold_metadata,
        )


def run_puncta_declump(
    image: np.ndarray,
    config: PunctaDeclumpConfig | None = None,
    *,
    external_mask: np.ndarray | None = None,
) -> DeclumpResult:
    """Convenience entry point for puncta declumping."""
    pipeline = PunctaDeclumpPipeline(config)
    return pipeline.run(image, external_mask=external_mask)
