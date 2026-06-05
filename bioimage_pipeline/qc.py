"""Quality control visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from skimage import color
from skimage import io as skio

from bioimage_pipeline.io import read_tiff, save_tiff


def _to_2d_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[0] <= 4:
            array = array[0]
        else:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image plane, got shape {array.shape}")
    return array


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Scale a grayscale image to uint8 using robust percentiles."""
    plane = _to_2d_grayscale(image).astype(np.float32)
    low = float(np.percentile(plane, 1))
    high = float(np.percentile(plane, 99))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Create an RGB overlay of a binary mask on a grayscale image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = normalize_for_display(image)
    grayscale_rgb = np.stack([base, base, base], axis=-1)
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.shape != base.shape:
        raise ValueError(
            f"Mask shape {mask_arr.shape} must match image shape {base.shape}"
        )

    overlay = grayscale_rgb.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask_arr] = (1.0 - alpha) * overlay[mask_arr] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_label_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.5,
    background_label: int = 0,
) -> np.ndarray:
    """Create an RGB overlay of labeled objects on a grayscale image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = normalize_for_display(image)
    grayscale_rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    label_arr = np.asarray(labels)
    if label_arr.shape != base.shape:
        raise ValueError(
            f"Label shape {label_arr.shape} must match image shape {base.shape}"
        )

    colored_labels = color.label2rgb(
        label_arr,
        image=grayscale_rgb / 255.0,
        bg_label=background_label,
        alpha=alpha,
        image_alpha=1.0 - alpha,
    )
    return (np.clip(colored_labels, 0, 1) * 255).astype(np.uint8)


def save_qc_figure(path: str | Path, figure: np.ndarray) -> Path:
    """Save an RGB QC figure as PNG or TIFF based on the file extension."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(figure)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("QC figure must be an RGB image with shape (H, W, 3)")

    if output_path.suffix.lower() in {".tif", ".tiff"}:
        save_tiff(output_path, rgb.astype(np.uint8))
    elif output_path.suffix.lower() == ".png":
        skio.imsave(output_path, rgb.astype(np.uint8))
    else:
        raise ValueError(
            f"Unsupported QC figure format: {output_path.suffix}. "
            "Use .png, .tif, or .tiff."
        )

    return output_path.resolve()


def export_qc_artifacts(
    image: np.ndarray,
    output_dir: str | Path,
    stem: str,
    *,
    mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    image_format: str = "png",
) -> dict[str, Path]:
    """Save mask and label QC overlay figures for one image."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = image_format.lower().lstrip(".")
    artifacts: dict[str, Path] = {}

    if mask is not None:
        mask_overlay = create_mask_overlay(image, mask)
        mask_path = output_path / f"{stem}_qc_mask_overlay.{suffix}"
        artifacts["mask_overlay"] = save_qc_figure(mask_path, mask_overlay)

    if labels is not None:
        label_overlay = create_label_overlay(image, labels)
        label_path = output_path / f"{stem}_qc_label_overlay.{suffix}"
        artifacts["label_overlay"] = save_qc_figure(label_path, label_overlay)

    if not artifacts:
        raise ValueError("At least one of mask or labels must be provided")

    return artifacts


def _collect_image_paths(input_folder: Path) -> list[Path]:
    paths = sorted(input_folder.glob("*.tif"))
    paths.extend(sorted(input_folder.glob("*.tiff")))
    return [path for path in paths if path.is_file()]


def _find_matching_tiff(image_stem: str, search_dir: Path) -> Path | None:
    candidates = sorted(search_dir.glob("*.tif")) + sorted(search_dir.glob("*.tiff"))
    for candidate in candidates:
        if candidate.stem == image_stem:
            return candidate
    for candidate in candidates:
        if image_stem in candidate.stem:
            return candidate
    return None


def generate_qc_for_cellprofiler_results(
    input_dir: str | Path,
    masks_dir: str | Path,
    labels_dir: str | Path,
    qc_dir: str | Path,
    processed_images: Sequence[str],
    *,
    image_format: str = "png",
) -> dict[str, dict[str, Path]]:
    """Create mask/label QC overlays from organized CellProfiler workflow outputs."""
    input_path = Path(input_dir)
    masks_path = Path(masks_dir)
    labels_path = Path(labels_dir)
    qc_path = Path(qc_dir)
    qc_path.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, Path]] = {}
    for filename in processed_images:
        image_path = input_path / filename
        if not image_path.exists():
            continue

        image = read_tiff(image_path)
        image_stem = image_path.stem
        mask_file = _find_matching_tiff(image_stem, masks_path)
        label_file = _find_matching_tiff(image_stem, labels_path)

        mask = read_tiff(mask_file) > 0 if mask_file is not None else None
        labels = read_tiff(label_file) if label_file is not None else None
        if mask is None and labels is None:
            continue

        image_artifacts = export_qc_artifacts(
            image,
            qc_path,
            image_stem,
            mask=mask,
            labels=labels,
            image_format=image_format,
        )
        artifacts[filename] = image_artifacts

    return artifacts


def generate_qc_for_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    image_format: str = "png",
) -> dict[str, Any]:
    """Generate QC overlays for batch pipeline outputs in a folder.

    Expects pipeline outputs named ``<stem>_mask.tif`` and/or
    ``<stem>_labels.tif`` alongside the original input TIFFs.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        raise ValueError(f"Input folder does not exist: {input_path}")

    generated: list[str] = []
    skipped: list[str] = []
    artifacts: dict[str, dict[str, str]] = {}

    for image_path in _collect_image_paths(input_path):
        stem = image_path.stem
        mask_path = output_path / f"{stem}_mask.tif"
        labels_path = output_path / f"{stem}_labels.tif"

        if not mask_path.exists() and not labels_path.exists():
            skipped.append(image_path.name)
            continue

        image = read_tiff(image_path)
        mask = read_tiff(mask_path) > 0 if mask_path.exists() else None
        labels = read_tiff(labels_path) if labels_path.exists() else None

        image_artifacts = export_qc_artifacts(
            image,
            output_path,
            stem,
            mask=mask,
            labels=labels,
            image_format=image_format,
        )
        generated.append(image_path.name)
        artifacts[image_path.name] = {
            key: str(path) for key, path in image_artifacts.items()
        }

    return {"generated": generated, "skipped": skipped, "artifacts": artifacts}


def view_in_napari(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    title: str = "bioimage-pipeline QC",
) -> Any:
    """Open an optional Napari viewer for interactive QC.

    Requires the optional ``qc`` extra: ``pip install bioimage-pipeline[qc]``.
    """
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed. Install with: pip install bioimage-pipeline[qc]"
        ) from exc

    viewer = napari.Viewer(title=title)
    viewer.add_image(_to_2d_grayscale(image), name="image")
    if mask is not None:
        viewer.add_labels(np.asarray(mask).astype(np.uint8), name="mask", opacity=0.5)
    if labels is not None:
        viewer.add_labels(np.asarray(labels), name="labels", opacity=0.5)
    napari.run()
    return viewer
