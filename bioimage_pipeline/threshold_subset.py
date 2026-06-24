"""Select and stage representative image subsets for threshold recommender trials."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from bioimage_pipeline.gui.workflow_editor import scan_detected_images
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.validation import inspect_image

SubsetMode = Literal["auto", "manual"]
SubsetSampleMethod = Literal["first", "even", "random"]

DEFAULT_SUBSET_COUNT = 5
DEFAULT_SAMPLE_METHOD: SubsetSampleMethod = "even"
SUBSET_MANIFEST_FILENAME = "subset_manifest.json"
SUBSET_CHARACTERIZATION_FILENAME = "subset_characterization.json"
SUBSET_CHARACTERIZATION_CSV = "subset_characterization.csv"


@dataclass(frozen=True)
class ThresholdSubsetSelection:
    """Configuration for choosing a trial image subset."""

    mode: SubsetMode = "auto"
    sample_count: int = DEFAULT_SUBSET_COUNT
    sample_method: SubsetSampleMethod = DEFAULT_SAMPLE_METHOD
    random_seed: int | None = None
    selected_paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ThresholdSubsetManifest:
    """Record of which source images were staged for a trial run."""

    source_dir: Path
    staged_dir: Path
    mode: SubsetMode
    sample_count: int
    sample_method: SubsetSampleMethod
    image_names: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_dir": str(self.source_dir),
            "staged_dir": str(self.staged_dir),
            "mode": self.mode,
            "sample_count": self.sample_count,
            "sample_method": self.sample_method,
            "image_names": list(self.image_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ThresholdSubsetManifest:
        return cls(
            source_dir=Path(str(payload["source_dir"])),
            staged_dir=Path(str(payload["staged_dir"])),
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            sample_count=int(payload["sample_count"]),
            sample_method=str(payload["sample_method"]),  # type: ignore[arg-type]
            image_names=[str(name) for name in payload["image_names"]],
        )


def list_candidate_input_images(input_dir: str | Path) -> list[Path]:
    """Return image-like files under ``input_dir`` using workflow scan rules."""
    return scan_detected_images(input_dir)


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]

    last_index = length - 1
    indices: list[int] = []
    for index in range(count):
        position = round(index * last_index / (count - 1))
        if not indices or position != indices[-1]:
            indices.append(position)
    while len(indices) < count:
        for candidate in range(length):
            if candidate not in indices:
                indices.append(candidate)
                break
        else:
            break
    return sorted(indices)


def sample_image_paths(
    images: Sequence[Path],
    *,
    count: int = DEFAULT_SUBSET_COUNT,
    method: SubsetSampleMethod = DEFAULT_SAMPLE_METHOD,
    random_seed: int | None = None,
) -> list[Path]:
    """Return up to ``count`` image paths sampled from a sorted candidate list."""
    if not images:
        return []
    if count <= 0:
        raise ValueError("sample count must be positive.")
    if count >= len(images):
        return list(images)

    if method == "first":
        return list(images[:count])

    if method == "random":
        rng = random.Random(random_seed)
        return sorted(rng.sample(list(images), count), key=lambda path: path.name.lower())

    indices = _evenly_spaced_indices(len(images), count)
    return [images[index] for index in indices]


def select_input_subset(
    input_dir: str | Path,
    selection: ThresholdSubsetSelection | None = None,
    *,
    manual_image_names: Sequence[str] | None = None,
) -> list[Path]:
    """Choose image paths for a threshold recommender trial subset."""
    config = selection or ThresholdSubsetSelection()
    candidates = list_candidate_input_images(input_dir)
    if not candidates:
        raise ValueError(f"No input images found under: {input_dir}")

    by_name = {path.name: path for path in candidates}

    if manual_image_names:
        selected: list[Path] = []
        for name in manual_image_names:
            path = by_name.get(Path(name).name)
            if path is None:
                raise ValueError(f"Selected image not found in input folder: {name}")
            selected.append(path)
        return selected

    if config.mode == "manual" and config.selected_paths:
        return select_input_subset(
            input_dir,
            ThresholdSubsetSelection(mode="manual"),
            manual_image_names=config.selected_paths,
        )

    return sample_image_paths(
        candidates,
        count=config.sample_count,
        method=config.sample_method,
        random_seed=config.random_seed,
    )


def materialize_input_subset(
    source_dir: str | Path,
    dest_dir: str | Path,
    image_paths: Sequence[str | Path],
    *,
    mode: SubsetMode = "auto",
    sample_method: SubsetSampleMethod = DEFAULT_SAMPLE_METHOD,
    use_symlinks: bool = False,
) -> ThresholdSubsetManifest:
    """Copy or symlink selected images into a staging folder."""
    source_path = Path(source_dir).resolve()
    staged_path = Path(dest_dir).resolve()
    staged_path.mkdir(parents=True, exist_ok=True)

    staged_names: list[str] = []
    for item in image_paths:
        source_file = Path(item)
        if not source_file.is_file():
            source_file = source_path / source_file.name
        if not source_file.is_file():
            raise FileNotFoundError(f"Subset image not found: {item}")

        destination = staged_path / source_file.name
        if destination.exists():
            destination.unlink()

        if use_symlinks:
            destination.symlink_to(source_file)
        else:
            shutil.copy2(source_file, destination)
        staged_names.append(source_file.name)

    manifest = ThresholdSubsetManifest(
        source_dir=source_path,
        staged_dir=staged_path,
        mode=mode,
        sample_count=len(staged_names),
        sample_method=sample_method,
        image_names=staged_names,
    )
    save_subset_manifest(manifest, staged_path / SUBSET_MANIFEST_FILENAME)
    return manifest


def save_subset_manifest(manifest: ThresholdSubsetManifest, path: str | Path) -> Path:
    """Write ``subset_manifest.json``."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    return manifest_path.resolve()


def load_subset_manifest(path: str | Path) -> ThresholdSubsetManifest:
    """Load ``subset_manifest.json``."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ThresholdSubsetManifest.from_dict(payload)


@dataclass(frozen=True)
class SubsetImageCharacterization:
    """Intensity and histogram summary for one subset image."""

    image_name: str
    shape: tuple[int, ...]
    dtype: str
    min_value: float
    max_value: float
    mean_intensity: float
    p5_intensity: float
    p50_intensity: float
    p95_intensity: float
    background_mean: float
    dynamic_range: float
    estimated_snr: float | None
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_intensity": self.mean_intensity,
            "p5_intensity": self.p5_intensity,
            "p50_intensity": self.p50_intensity,
            "p95_intensity": self.p95_intensity,
            "background_mean": self.background_mean,
            "dynamic_range": self.dynamic_range,
            "estimated_snr": self.estimated_snr,
            "limitations": list(self.limitations),
        }


def characterize_subset_image(image_path: Path) -> SubsetImageCharacterization:
    """Summarize one staged subset image for pre-trial review."""
    array = np.asarray(read_tiff(image_path))
    properties = inspect_image(array)
    working = array[0] if array.ndim == 3 and array.shape[0] <= 4 else array
    if working.ndim != 2:
        working = np.squeeze(working)
    plane = np.asarray(working, dtype=np.float64)
    background = plane[plane <= np.percentile(plane, 25)]
    background_mean = float(background.mean()) if background.size else 0.0

    return SubsetImageCharacterization(
        image_name=image_path.name,
        shape=properties.shape,
        dtype=properties.dtype,
        min_value=properties.min_value,
        max_value=properties.max_value,
        mean_intensity=properties.mean_intensity,
        p5_intensity=float(np.percentile(plane, 5)),
        p50_intensity=float(np.percentile(plane, 50)),
        p95_intensity=float(np.percentile(plane, 95)),
        background_mean=background_mean,
        dynamic_range=properties.dynamic_range,
        estimated_snr=properties.estimated_snr,
        limitations=tuple(properties.limitations),
    )


def build_subset_characterization_report(
    staged_dir: str | Path,
    image_names: Sequence[str],
) -> list[SubsetImageCharacterization]:
    """Characterize each image in a staged subset folder."""
    staged_path = Path(staged_dir)
    report: list[SubsetImageCharacterization] = []
    for name in image_names:
        image_path = staged_path / name
        if not image_path.is_file():
            raise FileNotFoundError(f"Subset image not found for characterization: {image_path}")
        report.append(characterize_subset_image(image_path))
    return report


def save_subset_characterization_report(
    report: Sequence[SubsetImageCharacterization],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write subset characterization JSON and CSV under ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = (destination / SUBSET_CHARACTERIZATION_FILENAME).resolve()
    csv_path = (destination / SUBSET_CHARACTERIZATION_CSV).resolve()

    payload = [entry.to_dict() for entry in report]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import pandas as pd

    pd.DataFrame(payload).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def load_subset_characterization_report(path: str | Path) -> list[dict[str, Any]]:
    """Load subset characterization JSON."""
    report_path = Path(path)
    return json.loads(report_path.read_text(encoding="utf-8"))
