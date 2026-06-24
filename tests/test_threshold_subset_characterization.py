"""Tests for subset image characterization reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.threshold_subset import (
    build_subset_characterization_report,
    save_subset_characterization_report,
)


def test_build_subset_characterization_report_writes_histogram_summary(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "spots_00.tif"
    image = np.zeros((64, 64), dtype=np.uint16)
    image[20:24, 20:24] = 2000
    save_tiff(image_path, image, imagej_compatible=True)

    report = build_subset_characterization_report(tmp_path, ["spots_00.tif"])

    assert len(report) == 1
    entry = report[0]
    assert entry.image_name == "spots_00.tif"
    assert entry.max_value >= 2000
    assert entry.p95_intensity >= entry.p50_intensity

    paths = save_subset_characterization_report(report, tmp_path / "subset")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload[0]["image_name"] == "spots_00.tif"
    assert paths["csv"].is_file()
