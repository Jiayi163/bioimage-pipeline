"""Demonstrate watershed splitting for touching objects (Phase 11.4)."""

from pathlib import Path

import numpy as np

from bioimage_pipeline.analysis import build_default_pipeline
from bioimage_pipeline.export import (
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.qc import export_qc_artifacts


def make_touching_circles_image(shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    """Create an image with two touching bright circular objects."""
    image = np.full(shape, 40, dtype=np.uint16)
    rows, cols = np.ogrid[: shape[0], : shape[1]]

    for center_y, center_x, radius, intensity in (
        (40, 28, 12, 500),
        (40, 52, 12, 500),
    ):
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        image[circle] = intensity

    return image


def run_and_export(
    image: np.ndarray,
    output_dir: Path,
    stem: str,
    *,
    labeling_method: str,
) -> int:
    """Run the default pipeline and save outputs for one labeling method."""
    pipeline = build_default_pipeline(
        blur_sigma=1.0,
        min_object_size=20,
        labeling_method=labeling_method,  # type: ignore[arg-type]
    )
    result = pipeline.run({"image": image, "filename": f"{stem}.tif"})

    export_mask_tiff(output_dir / f"{stem}_mask.tif", result["mask"])
    export_label_tiff(output_dir / f"{stem}_labels.tif", result["labels"])
    export_measurements_csv(
        output_dir / f"{stem}_measurements.csv",
        result["measurements"],
    )
    export_qc_artifacts(
        image,
        output_dir,
        stem,
        mask=result["mask"],
        labels=result["labels"],
    )

    return int(result["labels"].max())


def main() -> None:
    output_dir = Path("output") / "touching_objects"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = make_touching_circles_image()
    save_tiff(output_dir / "touching_input.tif", image)

    connected_count = run_and_export(
        image,
        output_dir,
        "connected",
        labeling_method="connected",
    )
    watershed_count = run_and_export(
        image,
        output_dir,
        "watershed",
        labeling_method="watershed",
    )

    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Connected components detected: {connected_count} object(s)")
    print(f"Watershed splitting detected: {watershed_count} object(s)")
    print("Open the PNG overlays to compare labeling methods.")


if __name__ == "__main__":
    main()
