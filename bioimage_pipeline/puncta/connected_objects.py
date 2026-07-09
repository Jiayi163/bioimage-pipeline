"""Connected-component analysis for puncta declumping."""

from __future__ import annotations

import numpy as np
from skimage import measure

from bioimage_pipeline.puncta.types import ObjectInfo
from bioimage_pipeline.segment import label_objects


class ConnectedObjectAnalyzer:
    """Label connected foreground objects and extract punctum-relevant metrics."""

    def analyze(self, mask: np.ndarray, intensity_image: np.ndarray) -> tuple[np.ndarray, list[ObjectInfo]]:
        """Label mask objects and return per-object metadata."""
        mask_arr = np.asarray(mask).astype(bool)
        intensity = np.asarray(intensity_image)
        if mask_arr.shape != intensity.shape:
            raise ValueError("mask and intensity_image must have the same shape")

        labels = label_objects(mask_arr)
        objects: list[ObjectInfo] = []

        for region in measure.regionprops(labels, intensity_image=intensity):
            object_mask = labels == region.label
            coords = np.argwhere(object_mask)
            intensities = intensity[object_mask]
            brightest_index = int(np.argmax(intensities))
            brightest_row, brightest_col = coords[brightest_index]

            major = float(
                getattr(region, "axis_major_length", None)
                or getattr(region, "major_axis_length", 0.0)
                or 0.0
            )
            minor = float(
                getattr(region, "axis_minor_length", None)
                or getattr(region, "minor_axis_length", 0.0)
                or 0.0
            )
            elongation = major / max(minor, 1e-6) if minor > 0 else 1.0

            objects.append(
                ObjectInfo(
                    label=int(region.label),
                    area=float(region.area),
                    equivalent_diameter=float(region.equivalent_diameter_area),
                    bbox=tuple(int(v) for v in region.bbox),
                    centroid=(float(region.centroid[0]), float(region.centroid[1])),
                    brightest_row=float(brightest_row),
                    brightest_col=float(brightest_col),
                    brightest_intensity=float(intensities[brightest_index]),
                    eccentricity=float(getattr(region, "eccentricity", 0.0) or 0.0),
                    solidity=float(getattr(region, "solidity", 1.0) or 1.0),
                    major_axis_length=major,
                    minor_axis_length=minor,
                    elongation=float(elongation),
                )
            )

        return labels, objects
