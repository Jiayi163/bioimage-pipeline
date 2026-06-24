"""Tests for ground-truth catalog and GT variant scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioimage_pipeline.ground_truth import (
    build_ground_truth_manifest,
    discover_reference_masks,
    reference_mask_path_for_image,
    save_ground_truth_manifest,
)
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.threshold_variant_gt_scoring import (
    GroundTruthImageComparison,
    rank_ground_truth_variant_scores,
)
from bioimage_pipeline.validation import compare_objects


def test_reference_mask_path_for_image() -> None:
    path = reference_mask_path_for_image("/tmp/refs", "sample.tif")
    assert path.name == "sample_reference_mask.tif"


def test_discover_reference_masks(tmp_path: Path) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:15, 10:15] = 255
    save_tiff(tmp_path / "sample_reference_mask.tif", mask)

    discovered = discover_reference_masks(tmp_path, ["sample.tif", "missing.tif"])

    assert "sample.tif" in discovered
    assert "missing.tif" not in discovered


def test_build_ground_truth_manifest_pairs_subset_images(tmp_path: Path) -> None:
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()

    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:8, 4:8] = 120
    save_tiff(subset_dir / "a.tif", image)
    save_tiff(refs_dir / "a_reference_mask.tif", (image > 0).astype(np.uint8) * 255)

    manifest = build_ground_truth_manifest(subset_dir, refs_dir, ["a.tif"])
    manifest_path = save_ground_truth_manifest(manifest, tmp_path / "ground_truth")

    assert manifest_path.is_file()
    assert len(manifest.entries) == 1
    assert manifest.entries[0].image_name == "a.tif"


def test_rank_ground_truth_variant_scores_orders_by_mean_f1() -> None:
    rows = [
        GroundTruthImageComparison(
            variant_id="001_baseline",
            display_name="Baseline",
            image_name="a.tif",
            success=True,
            f1=0.4,
            dice=0.5,
            count_error=2,
        ),
        GroundTruthImageComparison(
            variant_id="002_otsu",
            display_name="Otsu",
            image_name="a.tif",
            success=True,
            f1=0.9,
            dice=0.92,
            count_error=0,
        ),
    ]

    ranked = rank_ground_truth_variant_scores(rows)

    assert ranked[0].variant_id == "002_otsu"
    assert ranked[0].gt_rank == 1
    assert ranked[0].gt_label == "best_match"
    assert ranked[1].variant_id == "001_baseline"


def test_compare_objects_requires_matching_shapes() -> None:
    predicted = np.zeros((8, 8), dtype=bool)
    reference = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="Mask shapes do not match"):
        compare_objects(predicted, reference)


def test_resolve_predicted_mask_path_persists_after_return(tmp_path: Path) -> None:
    from bioimage_pipeline.threshold_variant_gt_scoring import (
        load_predicted_mask,
        resolve_predicted_mask_path,
    )

    raw = tmp_path / "cellprofiler_raw"
    raw.mkdir()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    save_tiff(raw / "sample_mask.tif", mask)

    path = resolve_predicted_mask_path(raw, "sample.tif")

    assert path is not None
    assert path.exists()
    loaded = load_predicted_mask(path)
    assert loaded.shape == (32, 32)
    assert loaded[15, 15]
