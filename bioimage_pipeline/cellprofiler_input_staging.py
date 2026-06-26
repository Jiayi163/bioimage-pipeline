"""Stage classifier outputs for CellProfiler measurement."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from bioimage_pipeline.io import read_tiff


@dataclass
class StagedInputPair:
    original_path: Path
    probability_path: Path | None
    mask_path: Path | None
    staged_original_path: Path
    staged_companion_path: Path


@dataclass
class StagingManifest:
    measurement_mode: str
    pairs: list[StagedInputPair] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "measurement_mode": self.measurement_mode,
            "pairs": [asdict(pair) for pair in self.pairs],
            "warnings": list(self.warnings),
        }


def _normalize_probability_map(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _write_probability_tiff(path: Path, array: np.ndarray) -> None:
    normalized = _normalize_probability_map(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, normalized)


def _write_mask_tiff(path: Path, array: np.ndarray) -> None:
    mask = (np.asarray(array) > 0).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, mask)


def stage_cellprofiler_input(
    originals_dir: str | Path,
    classifier_output_dir: str | Path,
    staging_dir: str | Path,
    *,
    measurement_mode: str = "binary_mask",
    image_pattern: str = "*.tif",
) -> StagingManifest:
    """Copy originals and classifier outputs into a CellProfiler input folder."""
    originals_path = Path(originals_dir).resolve()
    classifier_path = Path(classifier_output_dir).resolve()
    staging_path = Path(staging_dir).resolve()
    staging_path.mkdir(parents=True, exist_ok=True)

    probability_dir = classifier_path / "probability_maps"
    masks_dir = classifier_path / "masks"
    manifest = StagingManifest(measurement_mode=measurement_mode)

    for image_path in sorted(originals_path.glob(image_pattern)):
        if not image_path.is_file():
            continue
        stem = image_path.stem
        staged_original = staging_path / image_path.name
        shutil.copy2(image_path, staged_original)

        companion_source: Path | None = None
        staged_companion: Path
        if measurement_mode == "probability_map":
            for candidate in (
                probability_dir / f"{stem}_ev_probability.tif",
                probability_dir / f"{stem}_prob.tif",
            ):
                if candidate.is_file():
                    companion_source = candidate
                    break
            if companion_source is None:
                manifest.warnings.append(f"Missing probability map for {image_path.name}")
                continue
            staged_companion = staging_path / f"{stem}_prob.tif"
            _write_probability_tiff(staged_companion, read_tiff(companion_source))
        else:
            for candidate in (
                masks_dir / f"{stem}_ev_mask.tif",
                masks_dir / f"{stem}_mask.tif",
            ):
                if candidate.is_file():
                    companion_source = candidate
                    break
            if companion_source is None:
                manifest.warnings.append(f"Missing mask for {image_path.name}")
                continue
            staged_companion = staging_path / f"{stem}_mask.tif"
            _write_mask_tiff(staged_companion, read_tiff(companion_source))

        original_image = read_tiff(image_path)
        companion_image = read_tiff(companion_source)
        if original_image.shape[:2] != companion_image.shape[:2]:
            raise ValueError(
                f"Dimension mismatch for {image_path.name}: "
                f"{original_image.shape[:2]} vs {companion_image.shape[:2]}"
            )

        manifest.pairs.append(
            StagedInputPair(
                original_path=image_path,
                probability_path=companion_source if measurement_mode == "probability_map" else None,
                mask_path=companion_source if measurement_mode == "binary_mask" else None,
                staged_original_path=staged_original,
                staged_companion_path=staged_companion,
            )
        )
    return manifest


def save_cellprofiler_input_manifest(manifest: StagingManifest, logs_dir: str | Path) -> Path:
    path = Path(logs_dir) / "cellprofiler_input_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path
