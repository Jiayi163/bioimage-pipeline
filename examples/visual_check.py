"""Visual validation workflow without a GUI."""

from pathlib import Path

import numpy as np

from bioimage_pipeline.export import (
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.measure import measure_objects
from bioimage_pipeline.preprocess import gaussian_blur
from bioimage_pipeline.qc import export_qc_artifacts
from bioimage_pipeline.segment import label_objects, remove_small_objects_from_mask
from bioimage_pipeline.threshold import otsu_threshold


def make_synthetic_image(shape: tuple[int, int] = (160, 160)) -> np.ndarray:
    """Create a test image with bright circular objects and noise."""
    image = np.random.randint(0, 15, size=shape, dtype=np.uint16)
    rows, cols = np.ogrid[: shape[0], : shape[1]]

    for center_y, center_x, radius, intensity in (
        (50, 50, 14, 230),
        (110, 105, 12, 210),
    ):
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        image[circle] = intensity

    return image


def main() -> None:
    output_dir = Path("output") / "visual_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = make_synthetic_image(shape=(160, 160))
    save_tiff(output_dir / "original.tif", image)

    processed = gaussian_blur(image, sigma=1)
    mask = otsu_threshold(processed)
    mask = remove_small_objects_from_mask(mask, min_size=20)
    labels = label_objects(mask)
    measurements = measure_objects(labels, intensity_image=image)

    export_mask_tiff(output_dir / "mask.tif", mask)
    export_label_tiff(output_dir / "labels.tif", labels)
    export_measurements_csv(output_dir / "measurements.csv", measurements)

    qc_paths = export_qc_artifacts(
        image,
        output_dir,
        "visual_check",
        mask=mask,
        labels=labels,
    )

    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Mask overlay: {qc_paths['mask_overlay']}")
    print(f"Label overlay: {qc_paths['label_overlay']}")
    print("Open the TIFF or PNG files in Fiji to visually inspect the result.")
    print("See docs/fiji_qc_workflow.md for a step-by-step Fiji checklist.")


if __name__ == "__main__":
    main()
