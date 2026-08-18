"""Process one mask object through background correction, maxima, and GMM fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector, ModelComparisonResult
from bioimage_pipeline.puncta.local_peak_recovery import (
    LocalPeakRecoveryAttempt,
    PeakSource,
    apply_recovery_to_debug,
    finalize_recovery,
)
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.object_router import RouteDecision
from bioimage_pipeline.puncta.phase_c_fallback import evaluate_phase_c_fallback
from bioimage_pipeline.puncta.types import (
    DetectionPath,
    GaussianComponent,
    MixtureFitResult,
    ModelSelectionDebug,
    ObjectInfo,
    ObjectPatch,
    PeakCandidate,
    PeakDetectionResult,
    PunctumCandidate,
)


@dataclass
class ObjectProcessResult:
    candidates: list[PunctumCandidate]
    path: DetectionPath
    patch: ObjectPatch | None = None
    mixture: MixtureFitResult | None = None
    single_component: GaussianComponent | None = None
    peak_detection: PeakDetectionResult | None = None
    debug: ModelSelectionDebug = field(default_factory=ModelSelectionDebug)
    comparison: ModelComparisonResult | None = None


class ObjectProcessor:
    """Run the full Gaussian/GMM workflow for one connected mask object."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config
        self.maxima_detector = MaximaDetector(config)
        self.model_selector = GaussianModelSelector(config)
        self.filter = CandidateFilter(config)

    def process(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
        obj: ObjectInfo,
        *,
        candidate_id_start: int,
        assigned_peaks: list[PeakCandidate] | None = None,
    ) -> ObjectProcessResult:
        """Legacy entry: full suspicious-path processing."""
        return self.process_suspicious(
            image,
            object_mask,
            obj,
            assigned_peaks=assigned_peaks or [],
            candidate_id_start=candidate_id_start,
        )

    def process_fast(
        self,
        obj: ObjectInfo,
        assigned_peaks: list[PeakCandidate],
        *,
        candidate_id_start: int,
        route: RouteDecision,
        recovery: LocalPeakRecoveryAttempt | None = None,
    ) -> ObjectProcessResult:
        """Fast path: no patch extraction, no Gaussian fitting."""
        peak_source: PeakSource = "assigned_global" if assigned_peaks else "fallback"
        debug = ModelSelectionDebug(
            n_raw_local_maxima=len(assigned_peaks),
            n_filtered_local_maxima=len(assigned_peaks),
            model_selection_reason=f"fast_path:{','.join(route.reasons)}",
            single_path_reason="fast_path_no_fit",
            peak_source=peak_source,
        )
        if recovery is not None:
            apply_recovery_to_debug(debug, recovery, peak_source=peak_source)

        if len(assigned_peaks) == 1:
            candidate = self.filter.accept_fast_peak(
                obj,
                assigned_peaks[0],
                candidate_id=candidate_id_start,
                route_reason=",".join(route.reasons),
            )
            path: DetectionPath = "fast_single" if candidate.accepted else "fallback"
            if not candidate.accepted and self.config.accept_brightest_on_fit_failure:
                candidate = self.filter.accept_fallback(
                    obj,
                    PeakCandidate(
                        row=obj.brightest_row,
                        col=obj.brightest_col,
                        intensity=obj.brightest_intensity,
                    ),
                    candidate_id=candidate_id_start,
                    component_id=1,
                    path="fallback",
                    prior_rejection=candidate.rejection_reason,
                )
                path = "fallback"
                debug.peak_source = "fallback"
        else:
            peak = PeakCandidate(
                row=obj.brightest_row,
                col=obj.brightest_col,
                intensity=obj.brightest_intensity,
            )
            candidate = self.filter.accept_fallback(
                obj,
                peak,
                candidate_id=candidate_id_start,
                component_id=1,
                path="fallback",
                prior_rejection="fast_path_no_assigned_peak",
            )
            path = "fallback"
            debug.model_selection_reason = "fast_path_brightest_fallback"
            debug.peak_source = "fallback"

        self._attach_debug(candidate, obj, debug)
        return ObjectProcessResult(
            candidates=[candidate],
            path=path,
            debug=debug,
            peak_detection=PeakDetectionResult(
                raw_peaks=assigned_peaks,
                filtered_peaks=assigned_peaks,
                method="image_level_assigned",
            ),
        )

    def recover_local_peaks(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
        obj: ObjectInfo,
    ) -> LocalPeakRecoveryAttempt:
        """Cheap object-level MaximaDetector recovery; no GMM."""
        patch = build_object_patch(image, object_mask, obj, self.config)
        detection = self._detect_peaks(patch, obj)
        return finalize_recovery(detection, patch, obj, self.config)

    def process_recovered_single(
        self,
        obj: ObjectInfo,
        recovery: LocalPeakRecoveryAttempt,
        *,
        candidate_id_start: int,
        route: RouteDecision,
    ) -> ObjectProcessResult:
        """Fast-single path for one recovered peak; no Gaussian fitting."""
        peaks = list(recovery.peaks)
        if len(peaks) != 1:
            raise ValueError("process_recovered_single requires exactly one recovered peak")

        peak_detection = recovery.detection or PeakDetectionResult(
            raw_peaks=list(peaks),
            filtered_peaks=list(peaks),
            method=recovery.peak_source,
        )
        primary = peaks[0]
        
        # Build debug with recovery provenance; no Gaussian fit metrics
        debug = ModelSelectionDebug(
            n_raw_local_maxima=len(peak_detection.raw_peaks),
            n_filtered_local_maxima=len(peak_detection.filtered_peaks),
            tried_gmm=False,
            model_selection_reason="local_peak_recovery_fast_single",
            single_path_reason="recovered_local_single_no_fit",
            peak_source=recovery.peak_source,
        )
        apply_recovery_to_debug(debug, recovery, peak_source=recovery.peak_source)
        if route.reasons:
            debug.model_selection_reason = (
                f"local_peak_recovery_fast_single:{','.join(route.reasons)}"
            )

        # Accept as fast peak, analogous to ordinary fast_single with assigned global peak
        candidate = self.filter.accept_fast_peak(
            obj,
            primary,
            candidate_id=candidate_id_start,
            route_reason=",".join(route.reasons),
        )
        path: DetectionPath = "fast_single" if candidate.accepted else "fallback"
        
        # If fast peak validation failed, apply brightest fallback if configured
        if not candidate.accepted and self.config.accept_brightest_on_fit_failure:
            candidate = self.filter.accept_fallback(
                obj,
                PeakCandidate(
                    row=obj.brightest_row,
                    col=obj.brightest_col,
                    intensity=obj.brightest_intensity,
                ),
                candidate_id=candidate_id_start,
                component_id=1,
                path="fallback",
                prior_rejection=candidate.rejection_reason,
            )
            path = "fallback"
            debug.peak_source = "fallback"
            debug.model_selection_reason = "local_peak_recovery_fast_validation_failed_fallback"

        self._attach_debug(candidate, obj, debug)
        return ObjectProcessResult(
            candidates=[candidate],
            path=path,
            debug=debug,
            patch=recovery.patch,
            peak_detection=peak_detection,
        )

    def process_suspicious(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
        obj: ObjectInfo,
        *,
        assigned_peaks: list[PeakCandidate],
        candidate_id_start: int,
        peak_source: PeakSource | None = None,
        recovery: LocalPeakRecoveryAttempt | None = None,
    ) -> ObjectProcessResult:
        patch = (
            recovery.patch
            if recovery is not None and recovery.patch is not None
            else build_object_patch(image, object_mask, obj, self.config)
        )
        assigned_method = "image_level_assigned"
        if peak_source == "recovered_local_detector":
            assigned_method = "recovered_local_detector"
        elif peak_source == "recovered_masked_argmax":
            assigned_method = "recovered_masked_argmax"
        elif peak_source in ("image_only_group", "image_only_gmm"):
            assigned_method = peak_source
        if assigned_peaks:
            peak_detection = PeakDetectionResult(
                raw_peaks=list(assigned_peaks),
                filtered_peaks=list(assigned_peaks),
                method=assigned_method,
            )
            peaks = list(assigned_peaks)
        else:
            peak_detection = self._detect_peaks(patch, obj)
            peaks = peak_detection.filtered_peaks
        if not peaks:
            peaks = [
                PeakCandidate(
                    row=obj.brightest_row,
                    col=obj.brightest_col,
                    intensity=obj.brightest_intensity,
                )
            ]

        debug = ModelSelectionDebug(
            n_raw_local_maxima=len(peak_detection.raw_peaks),
            n_filtered_local_maxima=len(peak_detection.filtered_peaks),
            peak_source=peak_source or ("assigned_global" if assigned_peaks else None),
        )
        if recovery is not None:
            apply_recovery_to_debug(
                debug, recovery, peak_source=peak_source or recovery.peak_source
            )

        # Fit one Gaussian initialized from assigned peaks.
        primary = peaks[0]
        single = self.model_selector.single_fitter.fit_peak(
            patch,
            primary,
            component_id=1,
            n_components_in_model=1,
        )
        if not single.fit_succeeded and len(peaks) > 1:
            single = self.model_selector.single_fitter.fit_peak(
                patch,
                peaks[1],
                component_id=1,
                n_components_in_model=1,
            )
        debug.one_gaussian_r_squared = single.r_squared if single.fit_succeeded else None
        debug.one_gaussian_residual_relative = (
            single.residual_relative if single.fit_succeeded else None
        )
        debug.one_gaussian_sigma = single.sigma if single.fit_succeeded else None
        debug.one_gaussian_sigma_row = single.sigma_row if single.fit_succeeded else None
        debug.one_gaussian_sigma_col = single.sigma_col if single.fit_succeeded else None
        debug.one_gaussian_amplitude = single.amplitude if single.fit_succeeded else None
        debug.one_gaussian_center_shift = single.center_shift if single.fit_succeeded else None

        trigger_reasons = self._collect_balanced_gmm_triggers(
            obj,
            peak_detection,
            single,
        )
        debug.gmm_trigger_reasons = trigger_reasons
        use_gmm = bool(trigger_reasons) and self.config.enable_gmm

        if not peak_detection.filtered_peaks and not peak_detection.raw_peaks:
            # No maxima at all — still try single fit; fallback if it fails.
            result = self._from_single_component(
                obj,
                patch,
                single,
                path="single",
                candidate_id_start=candidate_id_start,
                debug=debug,
                peak_detection=peak_detection,
            )
            debug.single_path_reason = "no_local_maxima_used_brightest_seed"
            debug.model_selection_reason = "single_only_no_maxima"
            self._finalize_under_split(debug, result.candidates)
            for candidate in result.candidates:
                self._attach_debug(candidate, obj, debug)
            result.debug = debug
            return result

        if not use_gmm:
            debug.tried_gmm = False
            debug.single_path_reason = self._single_path_reason(obj, peak_detection, single)
            debug.model_selection_reason = "kept_single_no_gmm_trigger"
            result = self._from_single_component(
                obj,
                patch,
                single,
                path="single",
                candidate_id_start=candidate_id_start,
                debug=debug,
                peak_detection=peak_detection,
            )
            self._finalize_under_split(debug, result.candidates)
            for candidate in result.candidates:
                self._attach_debug(candidate, obj, debug)
            result.debug = debug
            result.single_component = single
            return result

        debug.tried_gmm = True
        comparison = self.model_selector.select_balanced_model(
            patch,
            peaks,
            single_component=single,
            n_filtered_peaks=len(peak_detection.filtered_peaks),
            n_raw_peaks=len(peak_detection.raw_peaks),
            obj=obj,
        )
        result = self._result_from_comparison(
            obj,
            patch,
            peaks,
            peak_detection,
            single,
            comparison,
            candidate_id_start=candidate_id_start,
            debug=debug,
        )
        self._finalize_under_split(debug, result.candidates)

        fallback_decision = evaluate_phase_c_fallback(
            config=self.config,
            obj=obj,
            single=single,
            selected=comparison.selected,
            patch=patch,
            n_filtered_peaks=len(peak_detection.filtered_peaks),
            n_accepted=sum(
                1 for candidate in result.candidates if candidate.accepted and candidate.fit_status == "fit_ok"
            ),
            under_split_suspect=debug.under_split_suspect,
        )
        if fallback_decision.trigger:
            debug.under_split_reasons = []
            debug.under_split_suspect = False
            comparison = self.model_selector.apply_phase_c_fallback_refinement(
                comparison,
                patch,
                peaks,
                trigger_reason=fallback_decision.reason,
            )
            self._populate_debug_from_comparison(debug, comparison, patch, single)
            result = self._result_from_comparison(
                obj,
                patch,
                peaks,
                peak_detection,
                single,
                comparison,
                candidate_id_start=candidate_id_start,
                debug=debug,
            )
            self._finalize_under_split(debug, result.candidates)

        result.single_component = single
        result.comparison = comparison
        for candidate in result.candidates:
            self._attach_debug(candidate, obj, debug)
        result.debug = debug
        return result

    def _populate_debug_from_comparison(
        self,
        debug: ModelSelectionDebug,
        comparison: ModelComparisonResult,
        patch: ObjectPatch,
        single: GaussianComponent,
    ) -> None:
        debug.gmm_candidate_components = max(comparison.candidate_component_counts or [0])
        debug.model_selection_reason = comparison.selection_reason
        debug.rejected_component_reason = comparison.rejected_component_reason
        debug.gmm_search_mode = self.config.gmm_multi_start_mode
        debug.gmm_spurious_split_rejected = "spurious_tight_split" in (
            comparison.selection_reason or ""
        )
        if comparison.best_mixture is not None and comparison.best_mixture.fit_succeeded:
            mixture = comparison.best_mixture
            debug.best_gmm_r_squared = mixture.r_squared
            debug.best_gmm_residual_relative = mixture.residual_rmse / max(
                max((component.amplitude for component in mixture.components), default=1.0),
                1.0,
            )
            debug.best_gmm_n_components = mixture.n_components
            debug.gmm_winning_init_strategy = mixture.winning_init_strategy
            debug.gmm_multi_start_attempts = mixture.multi_start_attempts
            debug.gmm_multi_start_converged = mixture.multi_start_converged
            debug.gmm_acceptance_min_separation_px = (
                self.config.gmm_acceptance_min_separation
                if self.config.gmm_use_mixture_acceptance_separation
                else self.config.min_center_separation
            )
            if len(mixture.components) >= 2:
                c0 = mixture.components[0]
                c1 = mixture.components[1]
                debug.gmm_fitted_center_distance_px = math.hypot(
                    c0.fitted_col - c1.fitted_col,
                    c0.fitted_row - c1.fitted_row,
                )
            single_bic = self.model_selector._single_component_bic(patch, single)
            debug.gmm_bic_delta_vs_single = mixture.bic - single_bic
            debug.gmm_aic_delta_vs_single = mixture.aic - self.model_selector._single_component_aic(
                patch, single
            )

    def _result_from_comparison(
        self,
        obj: ObjectInfo,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
        peak_detection: PeakDetectionResult,
        single: GaussianComponent,
        comparison: ModelComparisonResult,
        *,
        candidate_id_start: int,
        debug: ModelSelectionDebug,
    ) -> ObjectProcessResult:
        self._populate_debug_from_comparison(debug, comparison, patch, single)
        selected = comparison.selected
        if isinstance(selected, MixtureFitResult):
            return self._from_mixture(
                obj,
                patch,
                selected,
                peaks,
                candidate_id_start,
                debug=debug,
                peak_detection=peak_detection,
                comparison=comparison,
            )
        result = self._from_single_component(
            obj,
            patch,
            selected,
            path="single",
            candidate_id_start=candidate_id_start,
            debug=debug,
            peak_detection=peak_detection,
            comparison=comparison,
        )
        if debug.rejected_component_reason:
            debug.under_split_reasons.append(
                f"gmm_tried_but_rejected:{debug.rejected_component_reason}"
            )
        return result

    def _detect_peaks(self, patch: ObjectPatch, obj: ObjectInfo) -> PeakDetectionResult:
        detection = self.maxima_detector.detect(patch.corrected, patch.object_mask)

        def to_global(peaks: list[PeakCandidate]) -> list[PeakCandidate]:
            return [
                PeakCandidate(
                    row=peak.row + patch.row_offset,
                    col=peak.col + patch.col_offset,
                    intensity=peak.intensity + patch.background_level,
                )
                for peak in peaks
            ]

        return PeakDetectionResult(
            raw_peaks=to_global(detection.raw_peaks),
            filtered_peaks=to_global(detection.filtered_peaks),
            method=detection.method,
        )

    def _collect_balanced_gmm_triggers(
        self,
        obj: ObjectInfo,
        peak_detection: PeakDetectionResult,
        single: GaussianComponent,
    ) -> list[str]:
        """Return strong GMM warnings; any one triggers two-stage GMM fitting."""
        reasons: list[str] = []
        n_filtered = len(peak_detection.filtered_peaks)
        n_raw = len(peak_detection.raw_peaks)
        cfg = self.config

        if n_filtered >= cfg.min_reliable_peaks_for_gmm:
            reasons.append(f"filtered_peaks={n_filtered}")

        if n_raw >= cfg.min_reliable_peaks_for_gmm and self._one_gaussian_fit_weak(single):
            reasons.append(f"raw_peaks={n_raw}_with_weak_single_fit")

        if obj.equivalent_diameter > cfg.single_spot_max_diameter:
            reasons.append(
                f"large_diameter={obj.equivalent_diameter:.2f}>{cfg.single_spot_max_diameter}"
            )

        expected_area = cfg.expected_single_spot_area * cfg.gmm_trigger_area_factor
        if obj.area > expected_area:
            reasons.append(f"large_area={obj.area:.1f}>{expected_area:.1f}")

        # Elongation/eccentricity only matter when multi-peak or truly oversized — not alone.
        shape_suspicious = (
            obj.elongation >= cfg.elongation_gmm_threshold
            or obj.eccentricity >= cfg.eccentricity_gmm_threshold
        )
        if shape_suspicious and (
            n_filtered >= cfg.min_reliable_peaks_for_gmm
            or n_raw >= cfg.min_reliable_peaks_for_gmm
            or obj.equivalent_diameter > cfg.single_spot_max_diameter
        ):
            if obj.elongation >= cfg.elongation_gmm_threshold:
                reasons.append(
                    f"elongated={obj.elongation:.2f}>={cfg.elongation_gmm_threshold}"
                )
            if obj.eccentricity >= cfg.eccentricity_gmm_threshold:
                reasons.append(
                    f"eccentric={obj.eccentricity:.2f}>={cfg.eccentricity_gmm_threshold}"
                )

        if not single.fit_succeeded:
            reasons.append("one_gaussian_fit_failed")
        else:
            if single.r_squared < cfg.gmm_trigger_r_squared:
                reasons.append(
                    f"low_one_gaussian_r2={single.r_squared:.3f}<{cfg.gmm_trigger_r_squared}"
                )
            if single.residual_relative > cfg.gmm_trigger_residual_relative:
                reasons.append(
                    f"high_one_gaussian_residual={single.residual_relative:.3f}"
                    f">{cfg.gmm_trigger_residual_relative}"
                )
            expected_sigma = cfg.expected_single_spot_diameter / 2.355
            sigma_limit = expected_sigma * cfg.gmm_trigger_sigma_factor
            if single.sigma > sigma_limit:
                reasons.append(
                    f"large_sigma={single.sigma:.2f}>{sigma_limit:.2f}"
                )
            if (
                single.residual_patch is not None
                and (
                    single.r_squared < cfg.gmm_weak_fit_r_squared
                    or single.residual_relative > cfg.gmm_trigger_residual_relative * 0.75
                )
                and self._has_structured_residual(single.residual_patch)
            ):
                reasons.append("structured_residual_two_lobes")

        return reasons

    def _one_gaussian_fit_weak(self, single: GaussianComponent) -> bool:
        if not single.fit_succeeded:
            return True
        cfg = self.config
        return (
            single.r_squared < cfg.gmm_weak_fit_r_squared
            or single.residual_relative > cfg.gmm_trigger_residual_relative
        )

    def _has_structured_residual(self, residual_patch: np.ndarray) -> bool:
        """Heuristic: positive residual has 2+ separated bright lobes."""
        positive = np.clip(residual_patch, 0.0, None)
        peak = float(positive.max())
        if peak <= 0:
            return False
        # Fast path: threshold and count connected components of bright residual.
        mask = positive > (0.35 * peak)
        if int(mask.sum()) < 4:
            return False
        from scipy import ndimage

        labeled, n_labels = ndimage.label(mask)
        if n_labels < 2:
            return False
        # Require two lobes with meaningful area and separation.
        sizes = ndimage.sum(mask, labeled, index=range(1, n_labels + 1))
        strong = [i + 1 for i, size in enumerate(sizes) if size >= 2]
        if len(strong) < 2:
            return False
        centers = ndimage.center_of_mass(mask, labeled, index=strong[:4])
        for i, c1 in enumerate(centers):
            for c2 in centers[i + 1 :]:
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if dist >= self.config.min_peak_distance:
                    return True
        return False

    def _single_path_reason(
        self,
        obj: ObjectInfo,
        peak_detection: PeakDetectionResult,
        single: GaussianComponent,
    ) -> str:
        parts = [
            f"diameter={obj.equivalent_diameter:.2f}<={self.config.single_spot_max_diameter}",
            f"filtered_peaks={len(peak_detection.filtered_peaks)}",
            f"raw_peaks={len(peak_detection.raw_peaks)}",
            f"elongation={obj.elongation:.2f}",
        ]
        if single.fit_succeeded:
            parts.append(f"one_r2={single.r_squared:.3f}")
            parts.append(f"one_resid={single.residual_relative:.3f}")
        return "single_path:" + ",".join(parts)

    def _finalize_under_split(
        self,
        debug: ModelSelectionDebug,
        candidates: list[PunctumCandidate],
    ) -> None:
        accepted = [c for c in candidates if c.accepted and c.fit_status == "fit_ok"]
        n_accepted = len(accepted)
        reasons = list(debug.under_split_reasons)

        if debug.n_raw_local_maxima >= 2 and n_accepted <= 1:
            reasons.append(
                f"raw_maxima={debug.n_raw_local_maxima}_but_accepted_components={n_accepted}"
            )
        if debug.n_filtered_local_maxima >= 2 and n_accepted <= 1:
            reasons.append(
                f"filtered_maxima={debug.n_filtered_local_maxima}_but_accepted_components={n_accepted}"
            )
        if debug.tried_gmm and n_accepted <= 1 and debug.rejected_component_reason:
            if not any(r.startswith("gmm_tried_but_rejected") for r in reasons):
                reasons.append(f"gmm_tried_but_rejected:{debug.rejected_component_reason}")
        if (
            debug.one_gaussian_r_squared is not None
            and debug.one_gaussian_r_squared < self.config.residual_gmm_r_squared
            and n_accepted <= 1
        ):
            reasons.append(f"low_one_gaussian_r2={debug.one_gaussian_r_squared:.3f}")
        if (
            debug.one_gaussian_residual_relative is not None
            and debug.one_gaussian_residual_relative > self.config.residual_gmm_relative
            and n_accepted <= 1
        ):
            reasons.append(
                f"high_one_gaussian_residual={debug.one_gaussian_residual_relative:.3f}"
            )
        if (
            debug.one_gaussian_sigma is not None
            and debug.one_gaussian_sigma
            > (self.config.expected_single_spot_diameter / 2.355)
            * self.config.residual_gmm_sigma_factor
            and n_accepted <= 1
        ):
            reasons.append(f"large_one_gaussian_sigma={debug.one_gaussian_sigma:.2f}")

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for reason in reasons:
            if reason not in seen:
                seen.add(reason)
                unique.append(reason)
        debug.under_split_reasons = unique
        debug.under_split_suspect = bool(unique) and n_accepted <= 1

    def _attach_debug(
        self,
        candidate: PunctumCandidate,
        obj: ObjectInfo,
        debug: ModelSelectionDebug,
    ) -> PunctumCandidate:
        candidate.object_area = obj.area
        candidate.object_equivalent_diameter = obj.equivalent_diameter
        candidate.object_eccentricity = obj.eccentricity
        candidate.object_solidity = obj.solidity
        candidate.object_major_axis_length = obj.major_axis_length
        candidate.object_minor_axis_length = obj.minor_axis_length
        candidate.object_elongation = obj.elongation
        candidate.gmm_trigger_reasons = ";".join(debug.gmm_trigger_reasons) or None
        candidate.n_raw_local_maxima = debug.n_raw_local_maxima
        candidate.n_filtered_local_maxima = debug.n_filtered_local_maxima
        candidate.tried_gmm = debug.tried_gmm
        candidate.gmm_candidate_components = debug.gmm_candidate_components
        candidate.one_gaussian_r_squared = debug.one_gaussian_r_squared
        candidate.one_gaussian_residual_relative = debug.one_gaussian_residual_relative
        candidate.best_gmm_r_squared = debug.best_gmm_r_squared
        candidate.best_gmm_residual_relative = debug.best_gmm_residual_relative
        candidate.best_gmm_n_components = debug.best_gmm_n_components
        candidate.gmm_bic_delta_vs_single = debug.gmm_bic_delta_vs_single
        candidate.gmm_aic_delta_vs_single = debug.gmm_aic_delta_vs_single
        candidate.model_selection_reason = debug.model_selection_reason or None
        candidate.rejected_component_reason = debug.rejected_component_reason
        candidate.under_split_suspect = debug.under_split_suspect
        candidate.under_split_reasons = ";".join(debug.under_split_reasons) or None
        candidate.gmm_winning_init_strategy = debug.gmm_winning_init_strategy
        candidate.gmm_search_mode = debug.gmm_search_mode
        candidate.gmm_spurious_split_rejected = debug.gmm_spurious_split_rejected
        candidate.gmm_multi_start_attempts = debug.gmm_multi_start_attempts
        candidate.gmm_multi_start_converged = debug.gmm_multi_start_converged
        candidate.local_peak_recovery_attempted = debug.local_peak_recovery_attempted
        candidate.local_peak_recovery_success = debug.local_peak_recovery_success
        candidate.local_peak_recovery_raw_count = debug.local_peak_recovery_raw_count
        candidate.local_peak_recovery_filtered_count = debug.local_peak_recovery_filtered_count
        candidate.peak_source = debug.peak_source
        if debug.peak_source in ("image_only_peak", "image_only_group", "image_only_gmm"):
            candidate.detection_provenance = debug.peak_source  # type: ignore[assignment]
        return candidate

    def _from_mixture(
        self,
        obj: ObjectInfo,
        patch: ObjectPatch,
        mixture: MixtureFitResult,
        peaks: list[PeakCandidate],
        candidate_id_start: int,
        *,
        debug: ModelSelectionDebug,
        peak_detection: PeakDetectionResult,
        comparison: ModelComparisonResult | None = None,
    ) -> ObjectProcessResult:
        if not mixture.fit_succeeded or not mixture.components:
            return self._fallback(
                obj,
                patch,
                candidate_id_start,
                path="gmm",
                reason=mixture.fit_error or "gmm_fit_failed",
                debug=debug,
                peak_detection=peak_detection,
            )

        candidates = self.filter.evaluate_mixture_components(
            obj,
            peaks,
            mixture,
            candidate_id_start=candidate_id_start,
            object_mask=patch.object_mask,
            patch=patch,
        )
        for candidate in candidates:
            candidate.gmm_winning_init_strategy = mixture.winning_init_strategy
            self._attach_debug(candidate, obj, debug)
        return ObjectProcessResult(
            candidates=candidates,
            path="gmm",
            patch=patch,
            mixture=mixture,
            peak_detection=peak_detection,
            debug=debug,
            comparison=comparison,
        )

    def _from_single_component(
        self,
        obj: ObjectInfo,
        patch: ObjectPatch,
        component: GaussianComponent,
        *,
        path: DetectionPath,
        candidate_id_start: int,
        debug: ModelSelectionDebug,
        peak_detection: PeakDetectionResult,
        comparison: ModelComparisonResult | None = None,
    ) -> ObjectProcessResult:
        peak = PeakCandidate(
            row=component.initial_row,
            col=component.initial_col,
            intensity=component.amplitude + component.background
            if component.fit_succeeded
            else obj.brightest_intensity,
        )
        candidate = self.filter.evaluate_component(
            obj,
            peak,
            component,
            candidate_id=candidate_id_start,
            component_id=1,
            path=path,
            object_mask=patch.object_mask,
            patch=patch,
        )
        result_path: DetectionPath = path
        if not candidate.accepted and self.config.accept_brightest_on_fit_failure:
            candidate = self.filter.accept_fallback(
                obj,
                PeakCandidate(
                    row=obj.brightest_row,
                    col=obj.brightest_col,
                    intensity=obj.brightest_intensity,
                ),
                candidate_id=candidate_id_start,
                component_id=1,
                path="fallback",
                prior_rejection=candidate.rejection_reason,
            )
            result_path = "fallback"
        self._attach_debug(candidate, obj, debug)
        return ObjectProcessResult(
            candidates=[candidate],
            path=result_path,
            patch=patch,
            peak_detection=peak_detection,
            debug=debug,
            comparison=comparison,
            single_component=component,
        )

    def _fallback(
        self,
        obj: ObjectInfo,
        patch: ObjectPatch,
        candidate_id_start: int,
        *,
        path: DetectionPath,
        reason: str,
        debug: ModelSelectionDebug,
        peak_detection: PeakDetectionResult | None = None,
    ) -> ObjectProcessResult:
        peak = PeakCandidate(
            row=obj.brightest_row,
            col=obj.brightest_col,
            intensity=obj.brightest_intensity,
        )
        debug.model_selection_reason = debug.model_selection_reason or f"fallback:{reason}"
        if self.config.accept_brightest_on_fit_failure:
            candidate = self.filter.accept_fallback(
                obj,
                peak,
                candidate_id=candidate_id_start,
                component_id=1,
                path=path,
                prior_rejection=reason,
            )
        else:
            candidate = PunctumCandidate(
                object_id=obj.label,
                candidate_id=candidate_id_start,
                component_id=1,
                path=path,
                fit_status="rejected_bad_fit",
                initial_row=peak.row,
                initial_col=peak.col,
                accepted=False,
                rejection_reason=reason,
            )
        self._attach_debug(candidate, obj, debug)
        return ObjectProcessResult(
            candidates=[candidate],
            path="fallback",
            patch=patch,
            peak_detection=peak_detection,
            debug=debug,
        )
