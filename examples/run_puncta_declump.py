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
    print(f"Small single objects: {summary.small_single_objects}")
    print(f"Large clumped objects: {summary.large_clumped_objects}")
    print(f"Accepted puncta: {summary.total_accepted}")
    gaussian_count = sum(1 for c in result.accepted if c.sigma is not None)
    fallback_count = sum(1 for c in result.accepted if c.sigma is None)
    print(f"Gaussian fits: {gaussian_count}")
    print(f"Fallback (no fit): {fallback_count}")
    print(f"Rejected candidates: {summary.total_rejected}")
    print(f"Fallback objects: {summary.fallback_objects}")
    if "image_plane" in result.threshold_metadata:
        plane_info = result.threshold_metadata["image_plane"]
        print(
            f"Image plane: frame_index={plane_info.get('frame_index')} "
            f"source_shape={plane_info.get('source_shape')}"
        )
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
