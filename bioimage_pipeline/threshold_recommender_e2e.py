"""Synthetic fixtures and validation checks for real CellProfiler E2E trials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioimage_pipeline.cppipe_io import (
    CppipePipeline,
    DEFAULT_CPPIPE_PREAMBLE,
    module_template,
    save_cppipe,
)
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.pipeline_catalog import get_module_definition
from bioimage_pipeline.threshold_recommender import ThresholdRecommenderTrialResult
from bioimage_pipeline.threshold_subset import SUBSET_MANIFEST_FILENAME


@dataclass(frozen=True)
class ThresholdRecommenderE2eFixtures:
    """Paths for a minimal spot-detection recommender smoke test."""

    root: Path
    input_dir: Path
    cppipe_path: Path
    image_names: tuple[str, ...]


def _make_spot_image(
    shape: tuple[int, int] = (256, 256),
    *,
    spot_count: int,
    seed: int,
) -> np.ndarray:
    """Sparse bright spots on a dark background (EV-style fluorescence)."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 12, size=shape, dtype=np.uint16)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    for _ in range(spot_count):
        center_y = int(rng.integers(24, shape[0] - 24))
        center_x = int(rng.integers(24, shape[1] - 24))
        radius = int(rng.integers(2, 5))
        intensity = int(rng.integers(800, 2500))
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        image[circle] = np.maximum(image[circle], intensity)
    noise = rng.normal(0, 8, size=shape)
    return np.clip(image.astype(np.float32) + noise, 0, 4095).astype(np.uint16)


def _write_spot_tiff(path: Path, image: np.ndarray) -> None:
    """Write a TIFF CellProfiler and Bio-Formats reliably recognize."""
    save_tiff(path, image, imagej_compatible=True)


def build_spot_detection_cppipe() -> CppipePipeline:
    """Build a minimal catalog-based pipeline that CellProfiler can run headlessly."""
    ipo_updates = {
        "Select the input image": "Green",
        "Name the primary objects to be identified": "Spots",
        "Typical diameter of objects, in pixel units (Min,Max)": "3,12",
        "Method to distinguish clumped objects": "None",
        "Method to draw dividing lines between clumped objects": "None",
        "Discard objects outside the diameter range?": "Yes",
    }
    preamble = list(DEFAULT_CPPIPE_PREAMBLE)
    preamble[4] = "ModuleCount:5"
    modules = [
        module_template(
            get_module_definition("Images"),
            module_num=1,
            setting_overrides={"Filter images?": "Images only"},
        ),
        module_template(
            get_module_definition("NamesAndTypes"),
            module_num=2,
            include_hidden=True,
            setting_overrides={
                "Assign a name to": "All images",
                "Name to assign these images": "Green",
            },
        ),
        module_template(
            get_module_definition("IdentifyPrimaryObjects"),
            module_num=3,
            include_hidden=True,
            setting_overrides=ipo_updates,
        ),
        module_template(
            get_module_definition("ExportToSpreadsheet"),
            module_num=4,
            include_hidden=True,
            setting_overrides={
                "Data to export": "Spots",
                "Use the object name for the file name?": "Yes",
            },
        ),
        module_template(
            get_module_definition("SaveImages"),
            module_num=5,
            include_hidden=True,
            setting_overrides={
                "Select the type of image to save": "Mask",
                "Select the image to save": "Spots",
                "Select method for constructing file names": "From image filename",
                "Enter file prefix": "spots_mask",
            },
        ),
    ]
    return CppipePipeline(preamble=preamble, modules=modules)


def materialize_threshold_recommender_e2e_fixtures(
    root: Path,
    *,
    image_count: int = 3,
    force: bool = False,
) -> ThresholdRecommenderE2eFixtures:
    """Write synthetic TIFFs and a runnable ``.cppipe`` under ``root``."""
    root = root.resolve()
    input_dir = root / "input"
    pipeline_dir = root / "pipeline"
    input_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    cppipe_path = pipeline_dir / "spot_detection.cppipe"
    if force or not cppipe_path.is_file():
        save_cppipe(build_spot_detection_cppipe(), cppipe_path)

    image_names: list[str] = []
    for index in range(image_count):
        name = f"spots_{index:02d}.tif"
        image_path = input_dir / name
        if force or not image_path.is_file():
            image = _make_spot_image(spot_count=6 + index * 2, seed=index + 1)
            _write_spot_tiff(image_path, image)
        image_names.append(name)

    return ThresholdRecommenderE2eFixtures(
        root=root,
        input_dir=input_dir,
        cppipe_path=cppipe_path,
        image_names=tuple(image_names),
    )


def validate_threshold_recommender_trial_result(
    trial_result: ThresholdRecommenderTrialResult,
    *,
    min_ranked_variants: int = 1,
    require_successful_runs: bool = True,
) -> list[str]:
    """Return validation warnings; raise ``ValueError`` on hard failures."""
    errors: list[str] = []
    warnings: list[str] = []

    if not trial_result.session_path.is_file():
        errors.append(f"Missing session file: {trial_result.session_path}")

    subset_manifest_path = trial_result.subset_dir / SUBSET_MANIFEST_FILENAME
    if not subset_manifest_path.is_file():
        errors.append(f"Missing subset manifest: {subset_manifest_path}")

    if not trial_result.subset_manifest.image_names:
        errors.append("Subset manifest contains no image names.")

    if len(trial_result.ranked_scores) < min_ranked_variants:
        errors.append(
            f"Expected at least {min_ranked_variants} ranked variant(s), "
            f"got {len(trial_result.ranked_scores)}."
        )

    ranking_csv = trial_result.ranking_paths.get("csv")
    if ranking_csv is None or not Path(ranking_csv).is_file():
        errors.append("Ranking CSV was not written.")

    comparison_csv = trial_result.comparison_paths.get("csv")
    if comparison_csv is None or not Path(comparison_csv).is_file():
        errors.append("Comparison CSV was not written.")

    if require_successful_runs:
        successes = [result for result in trial_result.run_results if result.success]
        if not successes:
            messages = [
                f"{result.spec.variant_id}: {result.error_message or 'failed'}"
                for result in trial_result.run_results
            ]
            errors.append(
                "No variant CellProfiler runs succeeded. Details: "
                + "; ".join(messages)
            )
        successful_summaries = [
            summary for summary in trial_result.summaries if summary.success
        ]
        if successful_summaries and all(
            summary.object_count is None or summary.object_count == 0
            for summary in successful_summaries
        ):
            warnings.append(
                "CellProfiler runs succeeded but reported zero objects for every "
                "variant. Check input images, NamesAndTypes image names, and IPO "
                "input image settings."
            )

    if errors:
        raise ValueError("\n".join(errors))

    if trial_result.trial_mode == "optimistic" and trial_result.optimistic_qc is None:
        warnings.append("Trial mode is optimistic but no optimistic QC record was saved.")

    if trial_result.fell_back_to_full_search:
        warnings.append("Optimistic candidate failed QC; full variant search was used.")

    if trial_result.forced_full_search:
        warnings.append("Full variant search was forced despite optimistic QC pass.")

    return warnings
