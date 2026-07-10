"""Orchestrator for selective puncta declumping with image-level detection."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from bioimage_pipeline.puncta.candidate_detectors.base import get_detector
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.diagnostic_policy import (
    classify_object_for_diagnostics,
    select_objects_for_diagnostics,
)
from bioimage_pipeline.puncta.diagnostics import (
    export_object_diagnostic,
    export_under_split_diagnostic,
)
from bioimage_pipeline.puncta.object_processor import ObjectProcessResult, ObjectProcessor
from bioimage_pipeline.puncta.object_router import ObjectRouter
from bioimage_pipeline.puncta.peak_assignment import assign_peaks_to_objects
from bioimage_pipeline.puncta.threshold_mask import ThresholdMaskGenerator
from bioimage_pipeline.puncta.timing import PunctaTimingMetrics
from bioimage_pipeline.puncta.types import (
    DeclumpResult,
    DeclumpSummary,
    ObjectInfo,
    PunctumCandidate,
)
from bioimage_pipeline.puncta.under_split_report import build_under_split_report
from bioimage_pipeline.puncta.watershed_declump import apply_watershed_declump


class PunctaDeclumpPipeline:
    """Run selective Gaussian / GMM puncta declumping."""

    def __init__(self, config: PunctaDeclumpConfig | None = None) -> None:
        self.config = config or PunctaDeclumpConfig()
        self.mask_generator = ThresholdMaskGenerator(self.config)
        self.object_analyzer = ConnectedObjectAnalyzer()
        self.processor = ObjectProcessor(self.config)
        self.router = ObjectRouter(self.config)

    def run(
        self,
        image: np.ndarray,
        *,
        external_mask: np.ndarray | None = None,
        diagnostics_dir: str | None = None,
        source_path: str | None = None,
        output_dir: str | Path | None = None,
        stem: str = "puncta",
    ) -> DeclumpResult:
        image_arr = np.asarray(image)
        if image_arr.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported")

        timing = PunctaTimingMetrics(detector_name=self.config.candidate_detector)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        mask, threshold_metadata = self.mask_generator.generate(
            image_arr,
            external_mask=external_mask,
        )
        timing.preprocessing_time = time.perf_counter() - preprocess_start

        cc_start = time.perf_counter()
        labels, objects = self.object_analyzer.analyze(mask, image_arr)
        timing.connected_component_time = time.perf_counter() - cc_start
        timing.number_of_objects = len(objects)

        cache_dir = self._resolve_cache_dir(output_dir)
        detect_start = time.perf_counter()
        detector = get_detector(self.config)
        peak_table = detector.detect(
            image_arr,
            config=self.config,
            cache_dir=cache_dir,
            source_path=source_path,
            stem=stem,
        )
        timing.candidate_detection_time = time.perf_counter() - detect_start
        timing.cache_hit = peak_table.cache_hit

        assigned = assign_peaks_to_objects(labels, objects, peak_table, self.config)

        if self.config.enable_selective_routing and self.config.log_progress:
            routing_summary = self.router.summarize(objects, assigned)
            reason_parts = " ".join(
                f"{key}={value}"
                for key, value in sorted(routing_summary.reason_counts.items())
            )
            print(
                f"[puncta] routing: ordinary={routing_summary.ordinary} "
                f"suspicious={routing_summary.suspicious}"
                + (f" | reasons: {reason_parts}" if reason_parts else ""),
                flush=True,
                file=sys.stderr,
            )

        self.processor.filter.reset()
        candidates: list[PunctumCandidate] = []
        diagnostic_artifacts: list[str] = []
        object_results: list[tuple[ObjectInfo, ObjectProcessResult]] = []
        candidate_counter = 0
        single_count = 0
        fast_count = 0
        gmm_count = 0
        fallback_count = 0
        fit_ok_count = 0
        fallback_status_count = 0
        under_split_objects = 0
        gmm_triggered_count = 0
        gmm_accepted_count = 0
        suspicious_count = 0
        fitted_count = 0

        total_objects = len(objects)
        run_start = time.perf_counter()
        object_times: list[float] = []
        fit_start = time.perf_counter()

        for index, obj in enumerate(objects, start=1):
            object_start = time.perf_counter()
            object_mask = labels == obj.label
            obj_peaks = assigned.get(obj.label, [])
            candidate_counter += 1

            if self.config.enable_selective_routing:
                route = self.router.classify(obj, obj_peaks)
                if route.route == "ordinary_single":
                    result = self.processor.process_fast(
                        obj,
                        obj_peaks,
                        candidate_id_start=candidate_counter,
                        route=route,
                    )
                else:
                    suspicious_count += 1
                    result = self.processor.process_suspicious(
                        image_arr,
                        object_mask,
                        obj,
                        assigned_peaks=obj_peaks,
                        candidate_id_start=candidate_counter,
                    )
                    fitted_count += 1
            else:
                result = self.processor.process(
                    image_arr,
                    object_mask,
                    obj,
                    candidate_id_start=candidate_counter,
                    assigned_peaks=obj_peaks,
                )
                fitted_count += 1

            object_times.append(time.perf_counter() - object_start)
            candidate_counter += max(0, len(result.candidates) - 1)
            object_results.append((obj, result))

            if result.debug.tried_gmm:
                gmm_triggered_count += 1
            if result.path == "gmm":
                gmm_count += 1
                gmm_accepted_count += 1
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

            if self.config.log_progress and (
                index == 1
                or index == total_objects
                or index % self.config.progress_log_interval == 0
            ):
                self._log_progress(
                    index,
                    total_objects,
                    gmm_triggered_count,
                    run_start,
                    object_times,
                    suspicious_count,
                )

        timing.gaussian_fit_time = time.perf_counter() - fit_start
        timing.number_of_suspicious_objects = suspicious_count
        timing.number_of_fitted_objects = fitted_count
        timing.number_of_fast_path_objects = fast_count

        ws_start = time.perf_counter()
        if self.config.enable_watershed_declump:
            candidates_by_object: dict[int, list[PunctumCandidate]] = {}
            for candidate in candidates:
                candidates_by_object.setdefault(candidate.object_id, []).append(candidate)
            labels, _ = apply_watershed_declump(
                labels,
                image_arr,
                objects,
                candidates_by_object,
            )
        timing.watershed_time = time.perf_counter() - ws_start

        loop_runtime = time.perf_counter() - run_start

        summary = DeclumpSummary(
            total_mask_objects=total_objects,
            single_path_objects=single_count,
            gmm_path_objects=gmm_count,
            fallback_objects=fallback_count,
            total_candidates=len(candidates),
            total_accepted=len(self.processor.filter.accepted),
            total_rejected=len(candidates) - len(self.processor.filter.accepted),
            fit_ok_count=fit_ok_count,
            fit_failed_fallback_count=fallback_status_count,
            under_split_suspect_objects=under_split_objects,
            gmm_triggered_objects=gmm_triggered_count,
            gmm_accepted_objects=gmm_accepted_count,
            fast_path_objects=fast_count,
            suspicious_objects=suspicious_count,
            fitted_objects=fitted_count,
        )

        declump_result = DeclumpResult(
            candidates=candidates,
            summary=summary,
            mask=mask,
            labels=labels,
            objects=objects,
            threshold_metadata=threshold_metadata,
            diagnostic_artifacts=diagnostic_artifacts,
            peak_table=peak_table,
        )
        declump_result.under_split_report = build_under_split_report(
            declump_result,
            top_n=self.config.under_split_report_top_n,
        )

        diagnostics_exported = 0
        if diagnostics_dir is not None and self.config.diagnostic_mode not in ("off", "summary"):
            diag_start = time.perf_counter()
            diagnostic_artifacts.extend(
                self._export_selected_diagnostics(diagnostics_dir, object_results)
            )
            diagnostics_exported = len(diagnostic_artifacts)
            declump_result.diagnostic_artifacts = diagnostic_artifacts
            timing.diagnostic_export_time = time.perf_counter() - diag_start

        total_runtime = time.perf_counter() - total_start
        timing.total_time = total_runtime
        summary.diagnostics_exported = diagnostics_exported
        summary.total_runtime_seconds = total_runtime
        declump_result.summary = summary
        declump_result.timing = timing.to_dict()

        threshold_metadata["runtime"] = {
            "total_objects": total_objects,
            "total_seconds": round(total_runtime, 3),
            "loop_seconds": round(loop_runtime, 3),
            "average_seconds_per_object": round(
                total_runtime / max(total_objects, 1),
                4,
            ),
            "single_gaussian_objects": single_count,
            "fast_path_objects": fast_count,
            "suspicious_objects": suspicious_count,
            "fitted_objects": fitted_count,
            "gmm_triggered_objects": gmm_triggered_count,
            "gmm_accepted_objects": gmm_accepted_count,
            "fallback_objects": fallback_count,
            "rejected_candidates": summary.total_rejected,
            "diagnostics_exported": diagnostics_exported,
            "candidate_detector": self.config.candidate_detector,
            "detector_cache_hit": peak_table.cache_hit,
        }
        threshold_metadata["timing"] = timing.to_dict()
        threshold_metadata["diagnostics"] = {
            "mode": self.config.diagnostic_mode,
            "max_objects": self.config.max_diagnostic_objects,
            "exported_count": diagnostics_exported,
            "manual_object_ids": list(self.config.diagnostic_object_ids),
        }
        declump_result.threshold_metadata = threshold_metadata

        if self.config.log_progress:
            self._log_runtime_summary(summary, timing)

        return declump_result

    def _resolve_cache_dir(self, output_dir: str | Path | None) -> str | None:
        if self.config.detector_cache_dir is not None:
            return self.config.detector_cache_dir
        if output_dir is not None:
            return str(Path(output_dir) / ".puncta_cache")
        return None

    def _log_progress(
        self,
        processed: int,
        total: int,
        gmm_triggered: int,
        run_start: float,
        object_times: list[float],
        suspicious: int,
    ) -> None:
        elapsed = time.perf_counter() - run_start
        avg = elapsed / max(processed, 1)
        remaining = avg * max(total - processed, 0)
        print(
            f"[puncta] processed {processed}/{total} objects | "
            f"suspicious: {suspicious} | "
            f"GMM tried: {gmm_triggered} | "
            f"elapsed: {elapsed:.1f}s | "
            f"avg: {avg * 1000:.0f} ms/obj | "
            f"ETA: {remaining:.1f}s",
            flush=True,
            file=sys.stderr,
        )

    def _log_runtime_summary(
        self,
        summary: DeclumpSummary,
        timing: PunctaTimingMetrics,
    ) -> None:
        print(
            f"[puncta] done — objects={summary.total_mask_objects} "
            f"time={summary.total_runtime_seconds:.1f}s "
            f"fast={summary.fast_path_objects} "
            f"suspicious={summary.suspicious_objects} "
            f"fitted={summary.fitted_objects} "
            f"single={summary.single_path_objects} "
            f"gmm_triggered={summary.gmm_triggered_objects} "
            f"gmm_accepted={summary.gmm_accepted_objects} "
            f"fallback={summary.fallback_objects} "
            f"rejected={summary.total_rejected} "
            f"diagnostics={summary.diagnostics_exported} | "
            f"detect={timing.candidate_detection_time:.1f}s "
            f"fit={timing.gaussian_fit_time:.1f}s "
            f"watershed={timing.watershed_time:.1f}s",
            flush=True,
            file=sys.stderr,
        )

    def _export_selected_diagnostics(
        self,
        diagnostics_dir: str,
        object_results: list[tuple[ObjectInfo, ObjectProcessResult]],
    ) -> list[str]:
        records = []
        for obj, result in object_results:
            record = classify_object_for_diagnostics(obj, result, self.config)
            if record is not None:
                records.append(record)

        selected = select_objects_for_diagnostics(records, self.config)
        artifacts: list[str] = []

        for record in selected:
            obj = record.obj
            result = record.result
            if result.patch is None:
                continue

            use_undersplit = (
                "undersplit" in record.categories
                or "gmm" in record.categories
                or "gmm_rejected" in record.categories
                or record.result.debug.tried_gmm
            )
            if use_undersplit:
                mixture_for_diag = result.mixture
                if mixture_for_diag is None and result.comparison is not None:
                    mixture_for_diag = result.comparison.best_mixture
                path = export_under_split_diagnostic(
                    Path(diagnostics_dir) / "undersplit",
                    object_id=obj.label,
                    patch=result.patch,
                    peak_detection=result.peak_detection,
                    single=result.single_component,
                    mixture=mixture_for_diag,
                    debug=result.debug,
                )
                artifacts.append(str(path))
            else:
                primary = result.candidates[0] if result.candidates else None
                if primary is not None:
                    path = export_object_diagnostic(
                        diagnostics_dir,
                        object_id=obj.label,
                        patch=result.patch,
                        candidate=primary,
                        mixture=result.mixture,
                    )
                    artifacts.append(str(path))

        return artifacts


def run_puncta_declump(
    image: np.ndarray,
    config: PunctaDeclumpConfig | None = None,
    *,
    external_mask: np.ndarray | None = None,
    diagnostics_dir: str | None = None,
    source_path: str | None = None,
    output_dir: str | Path | None = None,
    stem: str = "puncta",
) -> DeclumpResult:
    pipeline = PunctaDeclumpPipeline(config)
    return pipeline.run(
        image,
        external_mask=external_mask,
        diagnostics_dir=diagnostics_dir,
        source_path=source_path,
        output_dir=output_dir,
        stem=stem,
    )
