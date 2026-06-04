"""Object measurement helpers."""

import numpy as np
import pandas as pd
from skimage import measure as sk_measure


def measure_objects(
    label_image: np.ndarray,
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure labeled objects and return a table of properties.

    Args:
        label_image: Integer label image with background 0.
        intensity_image: Optional original intensity image for intensity stats.

    Returns:
        DataFrame with one row per object.
    """
    labels = np.asarray(label_image)
    properties = ["label", "area", "centroid", "bbox"]

    if intensity_image is not None:
        properties.extend(["mean_intensity", "max_intensity"])
        table = sk_measure.regionprops_table(
            labels,
            intensity_image=np.asarray(intensity_image),
            properties=properties,
        )
    else:
        table = sk_measure.regionprops_table(labels, properties=properties)

    return pd.DataFrame(table)
