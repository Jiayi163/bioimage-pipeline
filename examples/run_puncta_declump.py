"""CLI entry point for puncta declumping."""

from __future__ import annotations

from bioimage_pipeline.puncta.ui import build_arg_parser, run_cli


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output = run_cli(args)
    result = output["result"]
    paths = output["paths"]

    summary = result.summary
    print(f"Mask objects: {summary.total_mask_objects}")
    print(f"Single Gaussian objects: {summary.single_path_objects}")
    print(f"GMM triggered: {summary.gmm_triggered_objects}")
    print(f"GMM accepted: {summary.gmm_accepted_objects}")
    print(f"Accepted puncta: {summary.total_accepted}")
    print(f"Gaussian fits (fit_ok): {summary.fit_ok_count}")
    print(f"Fallback (fit_failed_fallback): {summary.fit_failed_fallback_count}")
    print(f"Under-split suspects: {summary.under_split_suspect_objects}")
    print(f"Rejected candidates: {summary.total_rejected}")
    print(f"Fallback objects: {summary.fallback_objects}")
    print(f"Total runtime: {summary.total_runtime_seconds:.1f}s")
    if summary.total_mask_objects:
        avg_ms = summary.total_runtime_seconds / summary.total_mask_objects * 1000
        print(f"Average per object: {avg_ms:.0f} ms")
    if result.under_split_report:
        print("Top under-split failure categories:")
        categories: dict[str, int] = {}
        for row in result.under_split_report:
            cat = str(row.get("failure_category", "unknown"))
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items(), key=lambda item: -item[1]):
            print(f"  {cat}: {count}")
    if "image_plane" in result.threshold_metadata:
        plane_info = result.threshold_metadata["image_plane"]
        print(
            f"Image plane: frame_index={plane_info.get('frame_index')} "
            f"source_shape={plane_info.get('source_shape')}"
        )
    if "diagnostics" in result.threshold_metadata:
        diag = result.threshold_metadata["diagnostics"]
        print(
            f"Diagnostics: mode={diag.get('mode')} "
            f"exported={diag.get('exported_count')} "
            f"max={diag.get('max_objects')}"
        )
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
