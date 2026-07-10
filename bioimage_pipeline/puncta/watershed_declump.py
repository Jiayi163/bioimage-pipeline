"""Watershed splitting of suspicious objects using accepted fit centers."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import watershed

from bioimage_pipeline.puncta.types import ObjectInfo, PunctumCandidate


def apply_watershed_declump(
    labels: np.ndarray,
    image: np.ndarray,
    objects: list[ObjectInfo],
    candidates_by_object: dict[int, list[PunctumCandidate]],
    *,
    next_label: int | None = None,
) -> tuple[np.ndarray, int]:
    """Split multi-center objects using watershed seeds at accepted centers."""
    label_arr = np.asarray(labels, dtype=np.int32).copy()
    image_arr = np.asarray(image, dtype=np.float64)
    if next_label is None:
        next_label = int(label_arr.max()) + 1

    for obj in objects:
        accepted = [
            c
            for c in candidates_by_object.get(obj.label, [])
            if c.accepted and c.fit_status in ("fit_ok", "fit_failed_fallback")
        ]
        if len(accepted) < 2:
            continue

        min_row, min_col, max_row, max_col = obj.bbox
        crop_mask = label_arr[min_row:max_row, min_col:max_col] == obj.label
        if not crop_mask.any():
            continue

        markers = np.zeros(crop_mask.shape, dtype=np.int32)
        for index, candidate in enumerate(accepted, start=1):
            local_row = int(round(candidate.final_row - min_row))
            local_col = int(round(candidate.final_col - min_col))
            local_row = max(0, min(local_row, crop_mask.shape[0] - 1))
            local_col = max(0, min(local_col, crop_mask.shape[1] - 1))
            if crop_mask[local_row, local_col]:
                markers[local_row, local_col] = index
            else:
                coords = np.argwhere(crop_mask)
                if coords.size == 0:
                    continue
                distances = (coords[:, 0] - local_row) ** 2 + (coords[:, 1] - local_col) ** 2
                nearest = coords[int(np.argmin(distances))]
                markers[int(nearest[0]), int(nearest[1])] = index

        if markers.max() < 2:
            continue

        intensity_crop = image_arr[min_row:max_row, min_col:max_col]
        ws_labels = watershed(
            -intensity_crop,
            markers,
            mask=crop_mask,
        )

        label_arr[min_row:max_row, min_col:max_col][crop_mask] = 0
        for marker_id in range(1, int(markers.max()) + 1):
            sub_mask = ws_labels == marker_id
            if not sub_mask.any():
                continue
            label_arr[min_row:max_row, min_col:max_col][sub_mask] = next_label
            next_label += 1

    return label_arr, next_label
