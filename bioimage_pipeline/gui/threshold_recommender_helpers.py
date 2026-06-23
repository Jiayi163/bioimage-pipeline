"""Build threshold recommender config values from GUI selections."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.threshold_recommender import ThresholdRecommenderConfig
from bioimage_pipeline.threshold_subset import (
    SubsetSampleMethod,
    ThresholdSubsetSelection,
)
from bioimage_pipeline.threshold_variant_comparison import ThresholdVariantSizeThresholds


def build_recommender_config(
    *,
    imported_cppipe_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    cellprofiler_executable: str,
    subset_count: int,
    subset_method: SubsetSampleMethod,
    manual_subset_image_names: list[str] | None = None,
    max_variants: int | None = None,
    generate_qc: bool = True,
    tiny_area_px: float = 2.0,
    huge_area_px: float = 200.0,
    fast_optimistic: bool = True,
    force_full_search: bool = False,
) -> ThresholdRecommenderConfig:
    """Map GUI/CLI-like values to :class:`ThresholdRecommenderConfig`."""
    manual_names = manual_subset_image_names or []
    return ThresholdRecommenderConfig(
        imported_cppipe_path=Path(imported_cppipe_path),
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        cellprofiler_executable=cellprofiler_executable or "cellprofiler",
        generate_qc=generate_qc,
        max_variants=max_variants,
        subset_selection=ThresholdSubsetSelection(
            mode="manual" if manual_names else "auto",
            sample_count=subset_count,
            sample_method=subset_method,
        ),
        manual_subset_image_names=manual_names,
        size_thresholds=ThresholdVariantSizeThresholds(
            tiny_area_px=tiny_area_px,
            huge_area_px=huge_area_px,
        ),
        fast_optimistic=fast_optimistic,
        force_full_search=force_full_search,
    )


def selected_manual_subset_names(
    all_image_names: list[str],
    selected_indices: list[int],
) -> list[str]:
    """Return manually selected image basenames, or empty for auto sampling."""
    if not selected_indices:
        return []
    return [all_image_names[index] for index in selected_indices]
