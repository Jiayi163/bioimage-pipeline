"""Foreground mask generation for puncta declumping."""

from __future__ import annotations

import numpy as np
from skimage import measure

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.segment import clean_mask
from bioimage_pipeline.threshold import (
    adaptive_threshold,
    manual_threshold,
    otsu_threshold,
    sauvola_threshold,
)


class ThresholdMaskGenerator:
    """Generate a foreground mask from raw intensity or an external mask."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config

    def generate(
        self,
        image: np.ndarray,
        *,
        external_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return a cleaned boolean mask and metadata about thresholding."""
        image_arr = np.asarray(image)
        if image_arr.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported")

        metadata: dict[str, object] = {"method": self.config.threshold_method}

        if external_mask is not None:
            mask = np.asarray(external_mask).astype(bool)
            if mask.shape != image_arr.shape:
                raise ValueError("external_mask shape must match image shape")
            metadata["method"] = "external_mask"
        elif self.config.threshold_method == "external_mask":
            raise ValueError(
                "threshold_method is 'external_mask' but no external_mask was provided"
            )
        elif self.config.threshold_method == "manual":
            mask = manual_threshold(image_arr, self.config.manual_threshold_value)
            metadata["threshold_value"] = self.config.manual_threshold_value
        elif self.config.threshold_method == "adaptive":
            mask = adaptive_threshold(
                image_arr,
                block_size=self.config.adaptive_block_size,
                offset=self.config.adaptive_offset,
            )
            metadata["block_size"] = self.config.adaptive_block_size
            metadata["offset"] = self.config.adaptive_offset
        elif self.config.threshold_method == "sauvola":
            mask = sauvola_threshold(
                image_arr,
                block_size=self.config.sauvola_block_size,
                k=self.config.sauvola_k,
            )
            metadata["block_size"] = self.config.sauvola_block_size
            metadata["k"] = self.config.sauvola_k
        else:
            mask = otsu_threshold(image_arr)
            metadata["method"] = "otsu"

        mask = clean_mask(
            mask,
            min_size=max(1, self.config.min_object_area),
            fill_holes_in_mask=self.config.fill_holes,
            clear_border=self.config.clear_border,
        )
        mask = self._filter_by_area(mask)
        return mask.astype(bool), metadata

    def _filter_by_area(self, mask: np.ndarray) -> np.ndarray:
        """Remove objects outside min/max area bounds."""
        labels = measure.label(mask)
        if labels.max() == 0:
            return mask

        filtered = np.zeros_like(mask, dtype=bool)
        for region in measure.regionprops(labels):
            if self.config.min_object_area <= region.area <= self.config.max_object_area:
                filtered[labels == region.label] = True
        return filtered
