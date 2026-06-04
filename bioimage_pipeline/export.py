"""Export helpers for Fiji-friendly TIFF and CSV output."""

from pathlib import Path

import numpy as np
import pandas as pd

from bioimage_pipeline.io import save_tiff


def export_mask_tiff(path: str | Path, mask: np.ndarray) -> None:
    """Save a boolean mask as a Fiji-friendly 0/255 uint8 TIFF."""
    mask_arr = np.asarray(mask).astype(bool)
    fiji_mask = mask_arr.astype(np.uint8) * 255
    save_tiff(path, fiji_mask)


def export_label_tiff(path: str | Path, labels: np.ndarray) -> None:
    """Save a labeled image as an integer TIFF."""
    label_arr = np.asarray(labels).astype(np.int32)
    save_tiff(path, label_arr)


def export_measurements_csv(path: str | Path, dataframe: pd.DataFrame) -> None:
    """Save measurement results to a CSV file readable by Excel."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
