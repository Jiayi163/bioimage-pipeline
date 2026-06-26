"""Materialize bundled CellProfiler measurement templates for classifier outputs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_TEMPLATES_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "cellprofiler_workflows"
)


def default_binary_mask_cppipe_template() -> Path:
    return _TEMPLATES_DIR / "measure_from_binary_mask.cppipe"


def default_probability_map_cppipe_template() -> Path:
    return _TEMPLATES_DIR / "measure_from_probability_map.cppipe"


def materialize_classifier_pipeline(
    output_path: str | Path,
    *,
    template_path: str | Path | None = None,
    measurement_mode: str = "binary_mask",
    probability_threshold: float = 0.5,
    object_diameter_min: int = 3,
    object_diameter_max: int = 12,
) -> Path:
    """Copy a measurement template and patch minimal runtime settings."""
    if template_path is None:
        template = (
            default_probability_map_cppipe_template()
            if measurement_mode == "probability_map"
            else default_binary_mask_cppipe_template()
        )
    else:
        template = Path(template_path)
    if not template.is_file():
        raise FileNotFoundError(f"CellProfiler template not found: {template}")

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, destination)
    text = destination.read_text(encoding="utf-8")
    replacements = {
        r"Select the input image:Original": "Select the input image:EV",
        r"Select the input image:Mask": (
            "Select the input image:Prob"
            if measurement_mode == "probability_map"
            else "Select the input image:Mask"
        ),
        r"Threshold correction factor:0\.50": f"Threshold correction factor:{probability_threshold:.2f}",
        r"Typical diameter of objects, in pixel units \(Min,Max\):3,12": (
            f"Typical diameter of objects, in pixel units (Min,Max):"
            f"{object_diameter_min},{object_diameter_max}"
        ),
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    destination.write_text(text, encoding="utf-8")
    return destination
