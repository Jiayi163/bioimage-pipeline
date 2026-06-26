"""End-to-end Python classifier segmentation + CellProfiler measurement workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioimage_pipeline.analysis import run_cellprofiler_workflow_from_config, CellProfilerWorkflowConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Predict EV masks with a trained Python RF classifier, stage outputs, "
            "and run CellProfiler measurement once."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cppipe", type=Path, default=None, help="Optional CP template override.")
    parser.add_argument("--measurement-mode", choices=("binary_mask", "probability_map"), default="binary_mask")
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--executable", default="cellprofiler")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args()

    result = run_cellprofiler_workflow_from_config(
        CellProfilerWorkflowConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            segmentation_mode="python_classifier",
            classifier_model_path=args.model,
            classifier_cppipe_template_path=args.cppipe,
            classifier_measurement_mode=args.measurement_mode,
            classifier_probability_threshold=args.probability_threshold,
            cellprofiler_executable=args.executable,
        )
    )
    if args.json_summary:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Results: {result.results_dir}")
        print(f"Measurements: {result.measurements_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
