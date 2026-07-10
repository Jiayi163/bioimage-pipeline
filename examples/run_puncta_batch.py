"""CLI entry point for batch puncta declumping."""

from __future__ import annotations

from bioimage_pipeline.puncta.ui import build_arg_parser, run_cli


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output = run_cli(args)

    if "batch_result" in output:
        batch = output["batch_result"]
        print(f"Processed: {len(batch.processed)}")
        for stem in batch.processed:
            print(f"  {stem}")
        if batch.failed:
            print(f"Failed: {len(batch.failed)}")
            for stem, message in batch.failed:
                print(f"  {stem}: {message}")
        return

    result = output["result"]
    paths = output["paths"]
    summary = result.summary
    print(f"Mask objects: {summary.total_mask_objects}")
    print(f"Fast path: {summary.fast_path_objects}")
    print(f"Suspicious: {summary.suspicious_objects}")
    print(f"Fitted: {summary.fitted_objects}")
    print(f"Single Gaussian objects: {summary.single_path_objects}")
    print(f"GMM triggered: {summary.gmm_triggered_objects}")
    print(f"GMM accepted: {summary.gmm_accepted_objects}")
    print(f"Accepted puncta: {summary.total_accepted}")
    print(f"Total runtime: {summary.total_runtime_seconds:.1f}s")
    if result.timing:
        print("Timing breakdown:")
        for key in (
            "preprocessing_time",
            "connected_component_time",
            "candidate_detection_time",
            "gaussian_fit_time",
            "watershed_time",
            "diagnostic_export_time",
        ):
            if key in result.timing:
                print(f"  {key}: {result.timing[key]:.3f}s")
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
