"""Tests for CellProfiler input staging from classifier outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioimage_pipeline.cellprofiler_input_staging import stage_cellprofiler_input
from bioimage_pipeline.io import save_tiff


def test_stage_cellprofiler_input_binary_mask(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    classifier_output = tmp_path / "classifier_output"
    originals.mkdir()
    probability_dir = classifier_output / "probability_maps"
    masks_dir = classifier_output / "masks"
    masks_dir.mkdir(parents=True)
    image = np.zeros((16, 16), dtype=np.uint16)
    image[4:8, 4:8] = 1000
    save_tiff(originals / "EV_001_zmax.tif", image)
    save_tiff(masks_dir / "EV_001_zmax_ev_mask.tif", image > 0)
    staging_dir = tmp_path / "staging"
    manifest = stage_cellprofiler_input(originals, classifier_output, staging_dir)
    assert len(manifest.pairs) == 1
    assert (staging_dir / "EV_001_zmax.tif").is_file()
    assert (staging_dir / "EV_001_zmax_mask.tif").is_file()
