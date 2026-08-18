"""Image-only puncta detection orchestration."""

from __future__ import annotations

import sys
import time

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.image_only_peaks import detect_raw_peaks, validate_peaks
from bioimage_pipeline.puncta.object_processor import ObjectProcessor
from bioimage_pipeline.puncta.peak_grouping import group_peaks
from bioimage_pipeline.puncta.signal_support import build_signal_support, estimate_background
from bioimage_pipeline.puncta.synthetic_object import (
    build_group_patch_and_mask,
    make_peak_object_info,
)
from bioimage_pipeline.puncta.timing import PunctaTimingMetrics
from bioimage_pipeline.puncta.types import (
    DeclumpResult,
    DeclumpSummary,
    ImageOnlyDiagnostics,
    ImagePeakTable,
    ObjectInfo,
    PunctumCandidate,
)
from bioimage_pipeline.puncta.under_split_report import build_under_split_report


def _log_progress(config: PunctaDeclumpConfig, message: str) -> None:
    if config.log_progress:
        print(message, flush=True, file=sys.stderr)


def format_image_only_done_message(
    timing: dict[str, object],
    *,
    export_time: float = 0.0,
) -> str:
    """Format the final image-only timing summary line."""
    pipeline_total = float(timing.get("total_time", 0.0))
    grand_total = pipeline_total + export_time
    return (
        "[image_only] done "
        f"total={grand_total:.2f}s "
        f"(background={float(timing.get('image_only_background_time', 0.0)):.2f}s "
        f"support={float(timing.get('image_only_support_time', 0.0)):.2f}s "
        f"detect={float(timing.get('image_only_peak_detection_time', 0.0)):.2f}s "
        f"filter={float(timing.get('image_only_peak_filter_time', 0.0)):.2f}s "
        f"group={float(timing.get('image_only_grouping_time', 0.0)):.2f}s "
        f"gmm={float(timing.get('gaussian_fit_time', 0.0)):.2f}s "
        f"export={export_time:.2f}s)"
    )


def run_image_only(
    image: np.ndarray,
    config: PunctaDeclumpConfig,
    *,
    diagnostics_dir: str | None = None,
    output_dir: str | None = None,
    stem: str = "puncta",
) -> DeclumpResult:
    """Run peak-first image-only puncta detection."""
    del diagnostics_dir, output_dir, stem

    image_arr = np.asarray(image, dtype=np.float64)
    processor = ObjectProcessor(config)
    timing = PunctaTimingMetrics(detector_name="image_only")
    total_start = time.perf_counter()

    _log_progress(config, "[image_only] estimating background...")
    bg_start = time.perf_counter()
    background = estimate_background(image_arr, config)
    timing.image_only_background_time = time.perf_counter() - bg_start

    support_start = time.perf_counter()
    support_result = build_signal_support(background.corrected, config)
    timing.image_only_support_time = time.perf_counter() - support_start
    _log_progress(
        config,
        f"[image_only] support map generated in {timing.image_only_support_time:.2f}s",
    )

    _log_progress(config, "[image_only] detecting peaks...")
    detect_start = time.perf_counter()
    raw_peaks, response_method = detect_raw_peaks(
        background.corrected,
        support_result.support,
        config,
    )
    timing.image_only_peak_detection_time = time.perf_counter() - detect_start

    filter_start = time.perf_counter()
    peak_result = validate_peaks(
        image_arr,
        background.corrected,
        support_result.support,
        raw_peaks,
        config,
        response_method=response_method,
    )
    timing.image_only_peak_filter_time = time.perf_counter() - filter_start
    _log_progress(
        config,
        f"[image_only] raw_peaks={len(peak_result.raw_peaks)} "
        f"filtered_peaks={len(peak_result.validated_peaks)}",
    )

    _log_progress(config, "[image_only] grouping peaks...")
    group_start = time.perf_counter()
    peak_groups = group_peaks(peak_result.validated_peaks, image_arr.shape, config)
    timing.image_only_grouping_time = time.perf_counter() - group_start
    direct_groups = sum(1 for group in peak_groups if group.route == "direct")
    ambiguous_group_total = sum(1 for group in peak_groups if group.route == "gmm")
    _log_progress(
        config,
        f"[image_only] groups={len(peak_groups)} direct={direct_groups} "
        f"ambiguous={ambiguous_group_total}",
    )

    diagnostics = ImageOnlyDiagnostics(
        background=background.background,
        corrected=background.corrected,
        signal_support=support_result.support,
        raw_peaks=peak_result.raw_peaks,
        validated_peaks=peak_result.validated_peaks,
        rejected_peaks=peak_result.rejected_peaks,
        peak_groups=peak_groups,
        group_routes={group.group_id: group.route for group in peak_groups},
        group_routing_reasons={group.group_id: group.routing_reason for group in peak_groups},
    )

    processor.filter.reset()
    candidates: list[PunctumCandidate] = []
    objects: list[ObjectInfo] = []
    candidate_counter = 0
    direct_peaks = 0
    ambiguous_groups = 0
    gmm_groups = 0
    fast_count = 0
    gmm_count = 0
    fallback_count = 0
    single_count = 0
    gmm_triggered_count = 0
    fit_ok_count = 0
    fallback_status_count = 0
    under_split_objects = 0

    if ambiguous_group_total > 0:
        _log_progress(config, "[image_only] processing ambiguous GMM groups...")

    fit_start = time.perf_counter()
    gmm_group_index = 0
    peak_label = 1
    for group in peak_groups:
        if group.route == "direct":
            direct_provenance = (
                "direct_resolved_multi_peak"
                if group.routing_reason == "direct_resolved_multi_peak"
                else "image_only_peak"
            )
            for peak in group.peaks:
                candidate_counter += 1
                obj = make_peak_object_info(
                    peak,
                    label=peak_label,
                    raw=image_arr,
                    disk_radius=config.image_only_peak_disk_radius,
                )
                objects.append(obj)
                peak_label += 1
                candidate = processor.filter.accept_fast_peak(
                    obj,
                    peak,
                    candidate_id=candidate_counter,
                    route_reason=group.routing_reason,
                )
                candidate.peak_source = "image_only_peak"
                candidate.detection_provenance = direct_provenance
                candidate.object_area = obj.area
                candidate.object_equivalent_diameter = obj.equivalent_diameter
                candidates.append(candidate)
                if candidate.accepted:
                    direct_peaks += 1
                    fast_count += 1
                    fit_ok_count += 1
        else:
            gmm_group_index += 1
            _log_progress(
                config,
                f"[image_only] GMM group {gmm_group_index}/{ambiguous_group_total}",
            )
            ambiguous_groups += 1
            candidate_counter += 1
            patch, full_mask, obj = build_group_patch_and_mask(
                image_arr,
                group,
                support_result.support,
                config,
            )
            objects.append(obj)
            result = processor.process_suspicious(
                image_arr,
                full_mask,
                obj,
                assigned_peaks=group.peaks,
                candidate_id_start=candidate_counter,
                peak_source="image_only_group",
            )
            gmm_groups += 1 if result.debug.tried_gmm else 0
            if result.debug.tried_gmm:
                gmm_triggered_count += 1
            provenance = "image_only_gmm" if result.debug.tried_gmm else "image_only_group"
            for candidate in result.candidates:
                candidate.peak_source = provenance
                candidate.detection_provenance = "gmm_unresolved_multi_peak"
            if result.path == "gmm":
                gmm_count += 1
            elif result.path == "fallback":
                fallback_count += 1
            elif result.path == "fast_single":
                fast_count += 1
            else:
                single_count += 1
            if result.debug.under_split_suspect:
                under_split_objects += 1
            for candidate in result.candidates:
                if candidate.fit_status == "fit_ok":
                    fit_ok_count += 1
                if candidate.fit_status == "fit_failed_fallback":
                    fallback_status_count += 1
            candidates.extend(result.candidates)
            candidate_counter += max(0, len(result.candidates) - 1)

    timing.gaussian_fit_time = time.perf_counter() - fit_start
    _log_progress(
        config,
        f"[image_only] GMM fitting total={timing.gaussian_fit_time:.2f}s",
    )
    timing.image_only_direct_peaks = direct_peaks
    timing.image_only_ambiguous_groups = ambiguous_groups
    timing.image_only_gmm_groups = gmm_groups
    timing.number_of_objects = len(peak_groups)
    timing.number_of_fast_path_objects = fast_count
    timing.number_of_suspicious_objects = ambiguous_groups
    timing.number_of_fitted_objects = gmm_groups

    total_runtime = time.perf_counter() - total_start
    timing.total_time = total_runtime

    summary = DeclumpSummary(
        total_mask_objects=len(peak_groups),
        single_path_objects=single_count,
        gmm_path_objects=gmm_count,
        fallback_objects=fallback_count,
        total_candidates=len(candidates),
        total_accepted=len(processor.filter.accepted),
        total_rejected=len(candidates) - len(processor.filter.accepted),
        fit_ok_count=fit_ok_count,
        fit_failed_fallback_count=fallback_status_count,
        under_split_suspect_objects=under_split_objects,
        gmm_triggered_objects=gmm_triggered_count,
        gmm_accepted_objects=gmm_count,
        fast_path_objects=fast_count,
        suspicious_objects=ambiguous_groups,
        fitted_objects=gmm_groups,
        total_runtime_seconds=total_runtime,
    )

    threshold_metadata: dict[str, object] = {
        "detection_mask_mode": "image_only",
        "signal_support_method": support_result.method,
        "support_kind": support_result.support_kind,
        "background_method": background.method,
        "rolling_ball_radius": background.radius,
        "support_threshold": support_result.threshold,
        "support_noise_median": support_result.noise_median,
        "support_noise_mad": support_result.noise_mad,
        "image_only_counts": {
            "raw_peaks": len(peak_result.raw_peaks),
            "validated_peaks": len(peak_result.validated_peaks),
            "rejected_peaks": len(peak_result.rejected_peaks),
            "peak_groups": len(peak_groups),
            "direct_peaks": direct_peaks,
            "direct_groups": direct_groups,
            "ambiguous_groups": ambiguous_groups,
            "gmm_groups": gmm_groups,
        },
        "gmm_config": {
            "gmm_multi_start_enabled": config.gmm_multi_start_enabled,
            "gmm_max_multi_starts": config.gmm_max_multi_starts,
            "gmm_multi_start_max_nfev": config.gmm_multi_start_max_nfev,
            "gmm_multi_start_separations": list(config.gmm_multi_start_separations),
            "gmm_acceptance_min_separation": config.gmm_acceptance_min_separation,
            "gmm_multi_start_mode": config.gmm_multi_start_mode,
        },
        "runtime": {
            "total_seconds": round(total_runtime, 3),
            "direct_peaks": direct_peaks,
            "ambiguous_groups": ambiguous_groups,
            "gmm_groups": gmm_groups,
            "rejected_candidates": summary.total_rejected,
        },
        "timing": timing.to_dict(),
    }

    peak_table = ImagePeakTable(
        peaks=peak_result.validated_peaks,
        detector_name="image_only",
        method=peak_result.response_method,
        cache_hit=False,
    )

    declump_result = DeclumpResult(
        candidates=candidates,
        summary=summary,
        mask=support_result.support,
        labels=support_result.labels,
        objects=objects,
        threshold_metadata=threshold_metadata,
        peak_table=peak_table,
        image_only_diagnostics=diagnostics,
        timing=timing.to_dict(),
    )
    declump_result.under_split_report = build_under_split_report(
        declump_result,
        top_n=config.under_split_report_top_n,
    )

    return declump_result
