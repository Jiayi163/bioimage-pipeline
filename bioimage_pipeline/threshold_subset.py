"""Select and stage representative image subsets for threshold recommender trials."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from bioimage_pipeline.gui.workflow_editor import scan_detected_images

SubsetMode = Literal["auto", "manual"]
SubsetSampleMethod = Literal["first", "even", "random"]

DEFAULT_SUBSET_COUNT = 5
DEFAULT_SAMPLE_METHOD: SubsetSampleMethod = "even"
SUBSET_MANIFEST_FILENAME = "subset_manifest.json"


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
