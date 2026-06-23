"""Subset-first threshold recommender orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bioimage_pipeline.threshold_extraction import (
    load_identify_primary_objects_threshold_profiles,
)
from bioimage_pipeline.threshold_optimistic_qc import (
    OptimisticQcAssessment,
    assess_optimistic_qc,
    save_optimistic_qc_report,
)
from bioimage_pipeline.threshold_subset import (
    SUBSET_MANIFEST_FILENAME,
    ThresholdSubsetManifest,
    ThresholdSubsetSelection,
    list_candidate_input_images,
    materialize_input_subset,
    save_subset_manifest,
    select_input_subset,
)
from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
    ThresholdVariantSizeThresholds,
    compare_threshold_variant_run_results,
    save_threshold_variant_comparison,
)
from bioimage_pipeline.threshold_variant_runner import (
    ThresholdVariantRunResult,
    run_threshold_variant_artifact,
    run_threshold_variant_artifacts,
)
from bioimage_pipeline.threshold_variant_scoring import (
    ThresholdVariantScore,
    rank_threshold_variant_summaries,
    save_threshold_variant_ranking,
)
from bioimage_pipeline.threshold_variants import (
    ThresholdVariantArtifact,
    ThresholdVariantSpec,
    generate_basic_threshold_variant_specs,
    generate_optimistic_threshold_variant_spec,
    select_ipo_threshold_profile,
    write_threshold_pipeline_variants,
)

RECOMMENDER_ROOT_DIR = "threshold_recommender"
SUBSET_DIR = "subset"
OPTIMISTIC_DIR = "optimistic"
VARIANTS_DIR = "threshold_variants"
CONFIRMED_RUN_DIR = "confirmed_full_run"
SESSION_FILENAME = "recommender_session.json"
CONFIRMED_SELECTION_FILENAME = "confirmed_selection.json"


@dataclass
class ThresholdRecommenderConfig:
    """Inputs for a subset trial or confirmed full-dataset apply."""

    imported_cppipe_path: Path
    input_dir: Path
    output_dir: Path
    cellprofiler_executable: str = "cellprofiler"
    generate_qc: bool = True
    strict: bool = False
    max_variants: int | None = None
    ipo_module_index: int | None = None
    ipo_module_num: int | None = None
    ipo_object_name: str | None = None
    subset_selection: ThresholdSubsetSelection = field(
        default_factory=ThresholdSubsetSelection
    )
    manual_subset_image_names: list[str] = field(default_factory=list)
    size_thresholds: ThresholdVariantSizeThresholds = field(
        default_factory=ThresholdVariantSizeThresholds
    )
    full_dataset_trial: bool = False
    fast_optimistic: bool = True
    force_full_search: bool = False


@dataclass
class ThresholdVariantPreview:
    """Preview assets for one ranked variant."""

    variant_id: str
    variant_dir: Path
    qc_preview_paths: list[Path] = field(default_factory=list)


@dataclass
class ThresholdRecommenderTrialResult:
    """Outcome of a subset (or debug full-dataset) threshold recommender trial."""

    recommender_root: Path
    subset_dir: Path
    subset_manifest: ThresholdSubsetManifest
    variants_dir: Path
    artifacts: list[ThresholdVariantArtifact]
    run_results: list[ThresholdVariantRunResult]
    summaries: list[ThresholdVariantMeasurementSummary]
    ranked_scores: list[ThresholdVariantScore]
    comparison_paths: dict[str, Path]
    ranking_paths: dict[str, Path]
    session_path: Path
    preview_index: dict[str, ThresholdVariantPreview]
    trial_mode: str = "full_search"
    optimistic_qc: OptimisticQcAssessment | None = None
    optimistic_qc_path: Path | None = None
    fell_back_to_full_search: bool = False
    forced_full_search: bool = False


@dataclass
class ThresholdRecommenderApplyResult:
    """Outcome of applying one confirmed variant to the full dataset."""

    recommender_root: Path
    variant_id: str
    pipeline_path: Path
    full_input_dir: Path
    confirmed_run_dir: Path
    run_result: ThresholdVariantRunResult
    selection_path: Path


def recommender_root_dir(output_dir: str | Path) -> Path:
    return Path(output_dir).resolve() / RECOMMENDER_ROOT_DIR


def subset_staging_dir(recommender_root: str | Path) -> Path:
    return Path(recommender_root).resolve() / SUBSET_DIR


def variants_dir(recommender_root: str | Path) -> Path:
    return Path(recommender_root).resolve() / VARIANTS_DIR


def optimistic_dir(recommender_root: str | Path) -> Path:
    return Path(recommender_root).resolve() / OPTIMISTIC_DIR


def confirmed_run_dir(recommender_root: str | Path) -> Path:
    return Path(recommender_root).resolve() / CONFIRMED_RUN_DIR


def session_path(recommender_root: str | Path) -> Path:
    return Path(recommender_root).resolve() / SESSION_FILENAME


def _build_preview_index(
    run_results: Sequence[ThresholdVariantRunResult],
) -> dict[str, ThresholdVariantPreview]:
    preview_index: dict[str, ThresholdVariantPreview] = {}
    for result in run_results:
        qc_paths = sorted(result.qc_dir.glob("*.png")) if result.qc_dir.is_dir() else []
        if not qc_paths and result.qc_artifacts:
            for artifact_map in result.qc_artifacts.values():
                qc_paths.extend(
                    path for path in artifact_map.values() if path.suffix.lower() == ".png"
                )
            qc_paths = sorted(set(qc_paths))
        preview_index[result.spec.variant_id] = ThresholdVariantPreview(
            variant_id=result.spec.variant_id,
            variant_dir=result.variant_dir,
            qc_preview_paths=[path.resolve() for path in qc_paths],
        )
    return preview_index


def _resolve_variant_artifact(
    recommender_root: Path,
    variant_id: str,
    *,
    artifacts: Sequence[ThresholdVariantArtifact] | None = None,
    imported_cppipe_path: Path | None = None,
) -> ThresholdVariantArtifact:
    if artifacts is None:
        session = load_recommender_session(recommender_root)
        artifacts = _artifacts_from_session(session, recommender_root)
        if imported_cppipe_path is None:
            imported_cppipe_path = Path(session["imported_cppipe_path"])

    for artifact in artifacts:
        if artifact.spec.variant_id == variant_id:
            _validate_variant_pipeline_path(
                artifact.pipeline_path,
                recommender_root,
                imported_cppipe_path=imported_cppipe_path,
            )
            return artifact

    raise ValueError(f"Unknown variant_id for this recommender session: {variant_id}")


def _artifacts_from_session(
    session: dict[str, Any],
    recommender_root: Path,
) -> list[ThresholdVariantArtifact]:
    artifacts: list[ThresholdVariantArtifact] = []
    for entry in session.get("artifacts", []):
        variant_dir = Path(entry["variant_dir"])
        pipeline_path = Path(entry["pipeline_path"])
        spec_payload = entry["spec"]
        artifacts.append(
            ThresholdVariantArtifact(
                spec=ThresholdVariantSpec(
                    variant_id=spec_payload["variant_id"],
                    display_name=spec_payload["display_name"],
                    target_module_index=spec_payload["target_module_index"],
                    target_module_num=spec_payload.get("target_module_num"),
                    thresholding_method=spec_payload.get("thresholding_method"),
                    threshold_strategy=spec_payload.get("threshold_strategy"),
                    threshold_correction_factor=spec_payload.get(
                        "threshold_correction_factor"
                    ),
                    threshold_smoothing_scale=spec_payload.get(
                        "threshold_smoothing_scale"
                    ),
                    threshold_bounds=spec_payload.get("threshold_bounds"),
                    adaptive_window_size=spec_payload.get("adaptive_window_size"),
                    notes=spec_payload.get("notes"),
                    is_baseline=spec_payload.get("is_baseline", False),
                ),
                variant_dir=variant_dir,
                pipeline_path=pipeline_path,
            )
        )
    return artifacts


def _validate_variant_pipeline_path(
    pipeline_path: Path,
    recommender_root: Path,
    *,
    imported_cppipe_path: Path | None,
) -> None:
    resolved = pipeline_path.resolve()
    allowed_roots = (
        variants_dir(recommender_root).resolve(),
        optimistic_dir(recommender_root).resolve(),
        confirmed_run_dir(recommender_root).resolve(),
    )
    if not any(
        _path_is_relative_to(resolved, allowed_root) for allowed_root in allowed_roots
    ):
        raise ValueError(
            f"Refusing to run pipeline outside recommender directories: {resolved}"
        )
    if imported_cppipe_path is not None and resolved == imported_cppipe_path.resolve():
        raise ValueError(
            "Refusing to run the imported pipeline directly; use a generated variant."
        )


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def save_recommender_session(
    recommender_root: Path,
    *,
    config: ThresholdRecommenderConfig,
    subset_manifest: ThresholdSubsetManifest,
    artifacts: Sequence[ThresholdVariantArtifact],
    ranked_scores: Sequence[ThresholdVariantScore],
    comparison_paths: dict[str, Path],
    ranking_paths: dict[str, Path],
    preview_index: dict[str, ThresholdVariantPreview],
    trial_mode: str = "full_search",
    optimistic_qc: OptimisticQcAssessment | None = None,
    optimistic_qc_path: Path | None = None,
    fell_back_to_full_search: bool = False,
    forced_full_search: bool = False,
) -> Path:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "imported_cppipe_path": str(config.imported_cppipe_path.resolve()),
        "input_dir": str(config.input_dir.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "recommender_root": str(recommender_root.resolve()),
        "trial_mode": trial_mode,
        "fast_optimistic": config.fast_optimistic,
        "force_full_search": config.force_full_search,
        "fell_back_to_full_search": fell_back_to_full_search,
        "forced_full_search": forced_full_search,
        "subset_manifest": subset_manifest.to_dict(),
        "comparison_paths": {key: str(path) for key, path in comparison_paths.items()},
        "ranking_paths": {key: str(path) for key, path in ranking_paths.items()},
        "artifacts": [
            {
                "variant_dir": str(artifact.variant_dir),
                "pipeline_path": str(artifact.pipeline_path),
                "spec": asdict(artifact.spec),
            }
            for artifact in artifacts
        ],
        "ranking": [score.to_dict() for score in ranked_scores],
        "preview_index": {
            variant_id: {
                "variant_dir": str(preview.variant_dir),
                "qc_preview_paths": [str(path) for path in preview.qc_preview_paths],
            }
            for variant_id, preview in preview_index.items()
        },
    }
    if optimistic_qc is not None:
        payload["optimistic_qc"] = optimistic_qc.to_dict()
    if optimistic_qc_path is not None:
        payload["optimistic_qc_path"] = str(optimistic_qc_path.resolve())
    path = session_path(recommender_root)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path.resolve()


def load_recommender_session(recommender_root: str | Path) -> dict[str, Any]:
    path = session_path(recommender_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Recommender session not found: {path}. Run a subset trial first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_trial_subset(
    config: ThresholdRecommenderConfig,
    *,
    root: Path,
    input_path: Path,
) -> tuple[ThresholdSubsetManifest, Path]:
    if config.full_dataset_trial:
        subset_paths = list_candidate_input_images(input_path)
        if not subset_paths:
            raise ValueError(f"No input images found under: {input_path}")
        subset_manifest = ThresholdSubsetManifest(
            source_dir=input_path,
            staged_dir=input_path,
            mode="auto",
            sample_count=len(subset_paths),
            sample_method=config.subset_selection.sample_method,
            image_names=[path.name for path in subset_paths],
        )
        save_subset_manifest(
            subset_manifest,
            root / SUBSET_DIR / SUBSET_MANIFEST_FILENAME,
        )
        return subset_manifest, input_path

    subset_paths = select_input_subset(
        input_path,
        config.subset_selection,
        manual_image_names=config.manual_subset_image_names or None,
    )
    subset_manifest = materialize_input_subset(
        input_path,
        subset_staging_dir(root),
        subset_paths,
        mode=config.subset_selection.mode,
        sample_method=config.subset_selection.sample_method,
    )
    return subset_manifest, subset_manifest.staged_dir


def _run_variant_search_trial(
    *,
    imported_path: Path,
    profile: object,
    config: ThresholdRecommenderConfig,
    trial_input_dir: Path,
    variant_root: Path,
) -> tuple[
    list[ThresholdVariantArtifact],
    list[ThresholdVariantRunResult],
    list[ThresholdVariantMeasurementSummary],
    list[ThresholdVariantScore],
    dict[str, Path],
    dict[str, Path],
    dict[str, ThresholdVariantPreview],
]:
    specs = generate_basic_threshold_variant_specs(profile)
    if config.max_variants is not None:
        specs = specs[: config.max_variants]

    artifact_list = write_threshold_pipeline_variants(
        imported_path,
        variant_root,
        specs,
    )
    run_results = run_threshold_variant_artifacts(
        artifact_list,
        trial_input_dir,
        cellprofiler_executable=config.cellprofiler_executable,
        generate_qc=config.generate_qc,
        strict=config.strict,
    )
    summaries = compare_threshold_variant_run_results(
        run_results,
        size_thresholds=config.size_thresholds,
    )
    comparison_paths = save_threshold_variant_comparison(summaries, variant_root)
    ranked_scores = rank_threshold_variant_summaries(summaries)
    ranking_paths = save_threshold_variant_ranking(ranked_scores, variant_root)
    preview_index = _build_preview_index(run_results)
    return (
        artifact_list,
        run_results,
        summaries,
        ranked_scores,
        comparison_paths,
        ranking_paths,
        preview_index,
    )


def _run_optimistic_trial(
    *,
    imported_path: Path,
    profile: object,
    config: ThresholdRecommenderConfig,
    trial_input_dir: Path,
    optimistic_root: Path,
) -> tuple[
    list[ThresholdVariantArtifact],
    list[ThresholdVariantRunResult],
    list[ThresholdVariantMeasurementSummary],
    list[ThresholdVariantScore],
    dict[str, Path],
    dict[str, Path],
    dict[str, ThresholdVariantPreview],
    OptimisticQcAssessment,
    Path,
]:
    optimistic_spec = generate_optimistic_threshold_variant_spec(profile)
    baseline_spec = generate_basic_threshold_variant_specs(profile)[0]
    artifact_list = write_threshold_pipeline_variants(
        imported_path,
        optimistic_root,
        [baseline_spec, optimistic_spec],
    )
    run_results = run_threshold_variant_artifacts(
        artifact_list,
        trial_input_dir,
        cellprofiler_executable=config.cellprofiler_executable,
        generate_qc=config.generate_qc,
        strict=config.strict,
    )
    summaries = compare_threshold_variant_run_results(
        run_results,
        size_thresholds=config.size_thresholds,
    )
    comparison_paths = save_threshold_variant_comparison(
        summaries,
        optimistic_root,
        basename="optimistic_threshold_comparison",
    )
    baseline_summary = next(
        (item for item in summaries if item.variant_id == baseline_spec.variant_id),
        None,
    )
    optimistic_summary = next(
        (item for item in summaries if item.variant_id == optimistic_spec.variant_id),
        summaries[-1] if summaries else None,
    )
    if optimistic_summary is None:
        raise ValueError("Optimistic trial produced no measurement summary.")
    optimistic_qc = assess_optimistic_qc(
        optimistic_summary,
        baseline=baseline_summary,
    )
    preview_index = _build_preview_index(run_results)
    preview_paths = []
    optimistic_preview = preview_index.get(optimistic_spec.variant_id)
    if optimistic_preview is not None:
        preview_paths = optimistic_preview.qc_preview_paths
    optimistic_qc_path = save_optimistic_qc_report(
        optimistic_qc,
        optimistic_root,
        run_result=run_results[0] if run_results else None,
        preview_paths=preview_paths,
    )

    if optimistic_qc.passed and optimistic_qc.score is not None:
        ranked_scores = [
            ThresholdVariantScore(
                rank=1,
                variant_id=optimistic_qc.score.variant_id,
                display_name=optimistic_qc.score.display_name,
                score=optimistic_qc.score.score,
                reason=optimistic_qc.score.reason,
                explanations=list(optimistic_qc.score.explanations),
                component_scores=dict(optimistic_qc.score.component_scores),
                success=optimistic_qc.score.success,
                object_count=optimistic_qc.score.object_count,
                normal_frac=optimistic_qc.score.normal_frac,
                tiny_frac=optimistic_qc.score.tiny_frac,
                huge_frac=optimistic_qc.score.huge_frac,
                median_intensity=optimistic_qc.score.median_intensity,
                object_count_ratio_vs_baseline=optimistic_qc.score.object_count_ratio_vs_baseline,
            )
        ]
    else:
        ranked_scores = rank_threshold_variant_summaries(summaries)

    ranking_paths = save_threshold_variant_ranking(
        ranked_scores,
        optimistic_root,
        basename="optimistic_threshold_ranking",
    )
    return (
        artifact_list,
        run_results,
        summaries,
        ranked_scores,
        comparison_paths,
        ranking_paths,
        preview_index,
        optimistic_qc,
        optimistic_qc_path,
    )


def run_threshold_recommender_trial(
    config: ThresholdRecommenderConfig,
) -> ThresholdRecommenderTrialResult:
    """Run candidate variants on a subset, compare measurements, and rank.

    When ``fast_optimistic`` is enabled (default), try one optimistic Otsu
    adaptive candidate on the subset first. If basic QC passes, return that
    single candidate for optional full-dataset apply. Otherwise fall back to the
    multi-variant threshold search.
    """
    imported_path = Path(config.imported_cppipe_path).resolve()
    input_path = Path(config.input_dir).resolve()
    output_path = Path(config.output_dir).resolve()
    root = recommender_root_dir(output_path)
    root.mkdir(parents=True, exist_ok=True)

    subset_manifest, trial_input_dir = _prepare_trial_subset(
        config,
        root=root,
        input_path=input_path,
    )

    profiles = load_identify_primary_objects_threshold_profiles(imported_path)
    profile = select_ipo_threshold_profile(
        profiles,
        module_index=config.ipo_module_index,
        module_num=config.ipo_module_num,
        object_name=config.ipo_object_name,
    )

    trial_mode = "full_search"
    optimistic_qc: OptimisticQcAssessment | None = None
    optimistic_qc_path: Path | None = None
    fell_back_to_full_search = False
    forced_full_search = False

    if config.fast_optimistic:
        (
            artifact_list,
            run_results,
            summaries,
            ranked_scores,
            comparison_paths,
            ranking_paths,
            preview_index,
            optimistic_qc,
            optimistic_qc_path,
        ) = _run_optimistic_trial(
            imported_path=imported_path,
            profile=profile,
            config=config,
            trial_input_dir=trial_input_dir,
            optimistic_root=optimistic_dir(root),
        )
        variant_root = optimistic_dir(root)
        accept_optimistic = optimistic_qc.passed and not config.force_full_search
        if accept_optimistic:
            trial_mode = "optimistic"
        else:
            trial_mode = "full_search"
            if config.force_full_search and optimistic_qc.passed:
                forced_full_search = True
            else:
                fell_back_to_full_search = True
            (
                artifact_list,
                run_results,
                summaries,
                ranked_scores,
                comparison_paths,
                ranking_paths,
                preview_index,
            ) = _run_variant_search_trial(
                imported_path=imported_path,
                profile=profile,
                config=config,
                trial_input_dir=trial_input_dir,
                variant_root=variants_dir(root),
            )
            variant_root = variants_dir(root)
    else:
        variant_root = variants_dir(root)
        (
            artifact_list,
            run_results,
            summaries,
            ranked_scores,
            comparison_paths,
            ranking_paths,
            preview_index,
        ) = _run_variant_search_trial(
            imported_path=imported_path,
            profile=profile,
            config=config,
            trial_input_dir=trial_input_dir,
            variant_root=variant_root,
        )

    session_file = save_recommender_session(
        root,
        config=config,
        subset_manifest=subset_manifest,
        artifacts=artifact_list,
        ranked_scores=ranked_scores,
        comparison_paths=comparison_paths,
        ranking_paths=ranking_paths,
        preview_index=preview_index,
        trial_mode=trial_mode,
        optimistic_qc=optimistic_qc,
        optimistic_qc_path=optimistic_qc_path,
        fell_back_to_full_search=fell_back_to_full_search,
        forced_full_search=forced_full_search,
    )

    return ThresholdRecommenderTrialResult(
        recommender_root=root,
        subset_dir=subset_manifest.staged_dir,
        subset_manifest=subset_manifest,
        variants_dir=variant_root,
        artifacts=artifact_list,
        run_results=run_results,
        summaries=summaries,
        ranked_scores=ranked_scores,
        comparison_paths=comparison_paths,
        ranking_paths=ranking_paths,
        session_path=session_file,
        preview_index=preview_index,
        trial_mode=trial_mode,
        optimistic_qc=optimistic_qc,
        optimistic_qc_path=optimistic_qc_path,
        fell_back_to_full_search=fell_back_to_full_search,
        forced_full_search=forced_full_search,
    )


def apply_confirmed_threshold_variant(
    config: ThresholdRecommenderConfig,
    variant_id: str,
    *,
    confirmed: bool = False,
) -> ThresholdRecommenderApplyResult:
    """Run one confirmed variant pipeline on the full input folder."""
    if not confirmed:
        raise ValueError(
            "Full-dataset apply requires explicit confirmation (confirmed=True)."
        )

    root = recommender_root_dir(config.output_dir)
    session = load_recommender_session(root)
    imported_path = Path(session["imported_cppipe_path"]).resolve()
    artifact_list = _artifacts_from_session(session, root)
    artifact = _resolve_variant_artifact(
        root,
        variant_id,
        artifacts=artifact_list,
        imported_cppipe_path=imported_path,
    )

    full_input_dir = Path(config.input_dir).resolve()
    if not full_input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {full_input_dir}")

    run_root = confirmed_run_dir(root)
    run_root.mkdir(parents=True, exist_ok=True)
    confirmed_pipeline_path = run_root / "pipeline.cppipe"
    shutil.copy2(artifact.pipeline_path, confirmed_pipeline_path)
    apply_artifact = ThresholdVariantArtifact(
        spec=artifact.spec,
        variant_dir=run_root,
        pipeline_path=confirmed_pipeline_path,
    )

    run_result = run_threshold_variant_artifact(
        apply_artifact,
        full_input_dir,
        cellprofiler_executable=config.cellprofiler_executable,
        generate_qc=config.generate_qc,
    )

    selection_file = run_root / CONFIRMED_SELECTION_FILENAME
    selection_file.write_text(
        json.dumps(
            {
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
                "variant_id": variant_id,
                "display_name": artifact.spec.display_name,
                "pipeline_path": str(confirmed_pipeline_path.resolve()),
                "source_variant_pipeline_path": str(artifact.pipeline_path.resolve()),
                "imported_cppipe_path": str(imported_path),
                "full_input_dir": str(full_input_dir),
                "success": run_result.success,
                "error_message": run_result.error_message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ThresholdRecommenderApplyResult(
        recommender_root=root,
        variant_id=variant_id,
        pipeline_path=confirmed_pipeline_path.resolve(),
        full_input_dir=full_input_dir,
        confirmed_run_dir=run_root,
        run_result=run_result,
        selection_path=selection_file.resolve(),
    )


def load_trial_result_from_session(
    output_dir: str | Path,
) -> ThresholdRecommenderTrialResult:
    """Rebuild a trial result view from a saved recommender session."""
    root = recommender_root_dir(output_dir)
    session = load_recommender_session(root)
    subset_manifest = ThresholdSubsetManifest.from_dict(session["subset_manifest"])
    artifact_list = _artifacts_from_session(session, root)

    ranked_payload = session.get("ranking", [])
    ranked_scores = [
        ThresholdVariantScore(
            rank=entry["rank"],
            variant_id=entry["variant_id"],
            display_name=entry["display_name"],
            score=entry["score"],
            reason=entry["reason"],
            explanations=list(entry.get("explanations", [])),
            component_scores=dict(entry.get("component_scores", {})),
            success=entry.get("success", True),
            object_count=entry.get("object_count"),
            normal_frac=entry.get("normal_frac"),
            tiny_frac=entry.get("tiny_frac"),
            huge_frac=entry.get("huge_frac"),
            median_intensity=entry.get("median_intensity"),
            object_count_ratio_vs_baseline=entry.get("object_count_ratio_vs_baseline"),
        )
        for entry in ranked_payload
    ]

    preview_index: dict[str, ThresholdVariantPreview] = {}
    for variant_id, payload in session.get("preview_index", {}).items():
        preview_index[variant_id] = ThresholdVariantPreview(
            variant_id=variant_id,
            variant_dir=Path(payload["variant_dir"]),
            qc_preview_paths=[Path(path) for path in payload.get("qc_preview_paths", [])],
        )

    comparison_paths = {
        key: Path(path) for key, path in session.get("comparison_paths", {}).items()
    }
    ranking_paths = {
        key: Path(path) for key, path in session.get("ranking_paths", {}).items()
    }

    trial_mode = session.get("trial_mode", "full_search")
    session_variants_dir = (
        optimistic_dir(root) if trial_mode == "optimistic" else variants_dir(root)
    )
    optimistic_qc_path = (
        Path(session["optimistic_qc_path"])
        if session.get("optimistic_qc_path")
        else None
    )
    optimistic_qc = None
    optimistic_payload = session.get("optimistic_qc")
    if optimistic_payload is not None:
        summary_payload = optimistic_payload.get("summary", {})
        optimistic_qc = OptimisticQcAssessment(
            passed=optimistic_payload.get("passed", False),
            summary=ThresholdVariantMeasurementSummary(
                variant_id=summary_payload.get("variant_id", ""),
                display_name=summary_payload.get("display_name", ""),
                success=summary_payload.get("success", False),
                object_count=summary_payload.get("object_count"),
                median_area=summary_payload.get("median_area"),
                mean_area=summary_payload.get("mean_area"),
                tiny_frac=summary_payload.get("tiny_frac"),
                huge_frac=summary_payload.get("huge_frac"),
                normal_frac=summary_payload.get("normal_frac"),
                median_intensity=summary_payload.get("median_intensity"),
                mean_intensity=summary_payload.get("mean_intensity"),
                object_table_names=list(summary_payload.get("object_table_names", [])),
                measurements_dir=(
                    Path(summary_payload["measurements_dir"])
                    if summary_payload.get("measurements_dir")
                    else None
                ),
                warnings=list(summary_payload.get("warnings", [])),
                error_message=summary_payload.get("error_message"),
            ),
            warnings=list(optimistic_payload.get("warnings", [])),
            reasons=list(optimistic_payload.get("reasons", [])),
            baseline_object_count=optimistic_payload.get("baseline_object_count"),
            object_count_ratio_vs_baseline=optimistic_payload.get(
                "object_count_ratio_vs_baseline"
            ),
        )
        score_payload = optimistic_payload.get("score")
        if score_payload is not None:
            optimistic_qc.score = ThresholdVariantScore(
                rank=score_payload.get("rank", 0),
                variant_id=score_payload.get("variant_id", ""),
                display_name=score_payload.get("display_name", ""),
                score=score_payload.get("score", 0.0),
                reason=score_payload.get("reason", ""),
                explanations=list(score_payload.get("explanations", [])),
                component_scores=dict(score_payload.get("component_scores", {})),
                success=score_payload.get("success", True),
                object_count=score_payload.get("object_count"),
                normal_frac=score_payload.get("normal_frac"),
                tiny_frac=score_payload.get("tiny_frac"),
                huge_frac=score_payload.get("huge_frac"),
                median_intensity=score_payload.get("median_intensity"),
                object_count_ratio_vs_baseline=score_payload.get(
                    "object_count_ratio_vs_baseline"
                ),
            )

    return ThresholdRecommenderTrialResult(
        recommender_root=root,
        subset_dir=subset_manifest.staged_dir,
        subset_manifest=subset_manifest,
        variants_dir=session_variants_dir,
        artifacts=artifact_list,
        run_results=[],
        summaries=[],
        ranked_scores=ranked_scores,
        comparison_paths=comparison_paths,
        ranking_paths=ranking_paths,
        session_path=session_path(root),
        preview_index=preview_index,
        trial_mode=trial_mode,
        optimistic_qc=optimistic_qc,
        optimistic_qc_path=optimistic_qc_path,
        fell_back_to_full_search=session.get("fell_back_to_full_search", False),
        forced_full_search=session.get("forced_full_search", False),
    )
