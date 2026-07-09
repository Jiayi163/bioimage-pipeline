"""Orchestrator for size-gated puncta declumping with GMM support."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

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
from bioimage_pipeline.puncta.threshold_mask import ThresholdMaskGenerator
from bioimage_pipeline.puncta.types import (
    DeclumpResult,
    DeclumpSummary,
    ObjectInfo,
    PunctumCandidate,
)
from bioimage_pipeline.puncta.under_split_report import build_under_split_report


class PunctaDeclumpPipeline:
    """Run background-corrected Gaussian / GMM puncta declumping."""

    def __init__(self, config: PunctaDeclumpConfig | None = None) -> None:
        self.config = config or PunctaDeclumpConfig()
        self.mask_generator = ThresholdMaskGenerator(self.config)
        self.object_analyzer = ConnectedObjectAnalyzer()
        self.processor = ObjectProcessor(self.config)

    def run(
        self,
        image: np.ndarray,
        *,
        external_mask: np.ndarray | None = None,
        diagnostics_dir: str | None = None,
    ) -> DeclumpResult:
        image_arr = np.asarray(image)
        if image_arr.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported")

        mask, threshold_metadata = self.mask_generator.generate(
            image_arr,
            external_mask=external_mask,
        )
        labels, objects = self.object_analyzer.analyze(mask, image_arr)

        self.processor.filter.reset()
        candidates: list[PunctumCandidate] = []
        diagnostic_artifacts: list[str] = []
        object_results: list[tuple[ObjectInfo, ObjectProcessResult]] = []
        candidate_counter = 0
        single_count = 0
        gmm_count = 0
        fallback_count = 0
        fit_ok_count = 0
        fallback_status_count = 0
        under_split_objects = 0
        gmm_triggered_count = 0
        gmm_accepted_count = 0

        total_objects = len(objects)
        run_start = time.perf_counter()
        object_times: list[float] = []

        for index, obj in enumerate(objects, start=1):
            object_start = time.perf_counter()
            object_mask = labels == obj.label
            candidate_counter += 1
            result = self.processor.process(
                image_arr,
                object_mask,
                obj,
                candidate_id_start=candidate_counter,
            )
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
                )

        total_runtime = time.perf_counter() - run_start

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
        )

        declump_result = DeclumpResult(
            candidates=candidates,
            summary=summary,
            mask=mask,
            labels=labels,
            objects=objects,
            threshold_metadata=threshold_metadata,
            diagnostic_artifacts=diagnostic_artifacts,
        )
        declump_result.under_split_report = build_under_split_report(
            declump_result,
            top_n=self.config.under_split_report_top_n,
        )

        diagnostics_exported = 0
        if diagnostics_dir is not None and self.config.diagnostic_mode not in ("off", "summary"):
            diagnostic_artifacts.extend(
                self._export_selected_diagnostics(diagnostics_dir, object_results)
            )
            diagnostics_exported = len(diagnostic_artifacts)
            declump_result.diagnostic_artifacts = diagnostic_artifacts

        summary.diagnostics_exported = diagnostics_exported
        summary.total_runtime_seconds = total_runtime
        declump_result.summary = summary

        threshold_metadata["runtime"] = {
            "total_objects": total_objects,
            "total_seconds": round(total_runtime, 3),
            "average_seconds_per_object": round(
                total_runtime / max(total_objects, 1),
                4,
            ),
            "single_gaussian_objects": single_count,
            "gmm_triggered_objects": gmm_triggered_count,
            "gmm_accepted_objects": gmm_accepted_count,
            "fallback_objects": fallback_count,
            "rejected_candidates": summary.total_rejected,
            "diagnostics_exported": diagnostics_exported,
        }
        threshold_metadata["diagnostics"] = {
            "mode": self.config.diagnostic_mode,
            "max_objects": self.config.max_diagnostic_objects,
            "exported_count": diagnostics_exported,
            "manual_object_ids": list(self.config.diagnostic_object_ids),
        }
        declump_result.threshold_metadata = threshold_metadata

        if self.config.log_progress:
            self._log_runtime_summary(summary)

        return declump_result

    def _log_progress(
        self,
        processed: int,
        total: int,
        gmm_triggered: int,
        run_start: float,
        object_times: list[float],
    ) -> None:
        elapsed = time.perf_counter() - run_start
        avg = elapsed / max(processed, 1)
        remaining = avg * max(total - processed, 0)
        print(
            f"[puncta] processed {processed}/{total} objects | "
            f"GMM tried: {gmm_triggered} | "
            f"elapsed: {elapsed:.1f}s | "
            f"avg: {avg * 1000:.0f} ms/obj | "
            f"ETA: {remaining:.1f}s",
            flush=True,
            file=sys.stderr,
        )

    def _log_runtime_summary(self, summary: DeclumpSummary) -> None:
        print(
            f"[puncta] done — objects={summary.total_mask_objects} "
            f"time={summary.total_runtime_seconds:.1f}s "
            f"single={summary.single_path_objects} "
            f"gmm_triggered={summary.gmm_triggered_objects} "
            f"gmm_accepted={summary.gmm_accepted_objects} "
            f"fallback={summary.fallback_objects} "
            f"rejected={summary.total_rejected} "
            f"diagnostics={summary.diagnostics_exported}",
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

            # Under-split multi-panel PNG (preferred for debugging clumps).
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
                # Simple 3-panel for fallback / low-r2 / high-residual singles.
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
) -> DeclumpResult:
    pipeline = PunctaDeclumpPipeline(config)
    return pipeline.run(
        image,
        external_mask=external_mask,
        diagnostics_dir=diagnostics_dir,
    )
