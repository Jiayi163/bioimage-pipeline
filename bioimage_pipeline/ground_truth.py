"""Ground-truth mask catalog and pairing for threshold validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bioimage_pipeline.io import read_tiff

REFERENCE_MASK_SUFFIX = "_reference_mask"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class GroundTruthEntry:
    """One image paired with a lab-approved reference mask."""

    image_name: str
    image_path: Path
    reference_mask_path: Path
    notes: str | None = None
    difficulty_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_path"] = str(self.image_path)
        payload["reference_mask_path"] = str(self.reference_mask_path)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroundTruthEntry:
        return cls(
            image_name=str(payload["image_name"]),
            image_path=Path(payload["image_path"]),
            reference_mask_path=Path(payload["reference_mask_path"]),
            notes=payload.get("notes"),
            difficulty_tag=payload.get("difficulty_tag"),
        )


@dataclass
class GroundTruthManifest:
    """Catalog of reference masks available for a subset trial."""

    reference_dir: Path
    entries: list[GroundTruthEntry] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def annotated_image_names(self) -> list[str]:
        return [entry.image_name for entry in self.entries]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_dir": str(self.reference_dir.resolve()),
            "created_at": self.created_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroundTruthManifest:
        return cls(
            reference_dir=Path(payload["reference_dir"]),
            entries=[
                GroundTruthEntry.from_dict(entry)
                for entry in payload.get("entries", [])
            ],
            created_at=str(payload.get("created_at", "")),
        )


def reference_mask_path_for_image(
    reference_dir: str | Path,
    image_name: str,
) -> Path:
    """Return the expected reference mask path for one input image name."""
    stem = Path(image_name).stem
    return Path(reference_dir).resolve() / f"{stem}{REFERENCE_MASK_SUFFIX}.tif"


def discover_reference_masks(
    reference_dir: str | Path,
    image_names: Sequence[str],
) -> dict[str, Path]:
    """Map subset image names to existing reference mask files."""
    root = Path(reference_dir).resolve()
    if not root.is_dir():
        return {}

    discovered: dict[str, Path] = {}
    for image_name in image_names:
        candidate = reference_mask_path_for_image(root, image_name)
        if candidate.is_file():
            discovered[image_name] = candidate.resolve()
            continue

        stem = Path(image_name).stem
        for suffix in (".tif", ".tiff"):
            for pattern in (f"{stem}{REFERENCE_MASK_SUFFIX}{suffix}", f"{stem}*{suffix}"):
                matches = sorted(root.glob(pattern))
                for match in matches:
                    if match.is_file() and stem in match.stem:
                        discovered[image_name] = match.resolve()
                        break
                if image_name in discovered:
                    break
            if image_name in discovered:
                break
    return discovered


def validate_ground_truth_pair(
    image: np.ndarray,
    reference_mask: np.ndarray,
) -> list[str]:
    """Validate one image/reference mask pair and return warning messages."""
    warnings: list[str] = []
    image_array = np.asarray(image)
    mask_array = np.asarray(reference_mask)

    if image_array.ndim not in (2, 3):
        warnings.append(f"Image has unsupported dimensionality: {image_array.ndim}D")
    if mask_array.ndim not in (2, 3):
        warnings.append(f"Reference mask has unsupported dimensionality: {mask_array.ndim}D")

    image_plane = image_array
    mask_plane = mask_array
    if image_array.ndim == 3 and image_array.shape[0] <= 4:
        image_plane = image_array[0]
    elif image_array.ndim == 3:
        image_plane = image_array[..., 0]
    if mask_array.ndim == 3 and mask_array.shape[0] <= 4:
        mask_plane = mask_array[0]
    elif mask_array.ndim == 3:
        mask_plane = mask_array[..., 0]

    image_plane = np.squeeze(image_plane)
    mask_plane = np.squeeze(mask_plane)
    if image_plane.ndim == 2 and mask_plane.ndim == 2 and image_plane.shape != mask_plane.shape:
        warnings.append(
            f"Shape mismatch: image {image_plane.shape} vs reference mask {mask_plane.shape}"
        )

    unique_values = np.unique(mask_plane)
    if unique_values.size > 2:
        warnings.append(
            "Reference mask is not binary; non-zero values will be treated as foreground."
        )
    elif unique_values.size == 2 and not (
        0 in unique_values and max(unique_values) in {1, 255}
    ):
        warnings.append(
            "Reference mask uses non-standard foreground values; values > 0 count as object."
        )

    return warnings


def build_ground_truth_manifest(
    subset_dir: str | Path,
    reference_dir: str | Path,
    image_names: Sequence[str],
) -> GroundTruthManifest:
    """Pair subset images with available reference masks."""
    subset_path = Path(subset_dir).resolve()
    reference_path = Path(reference_dir).resolve()
    discovered = discover_reference_masks(reference_path, image_names)

    entries: list[GroundTruthEntry] = []
    for image_name in image_names:
        mask_path = discovered.get(image_name)
        if mask_path is None:
            continue
        image_path = subset_path / image_name
        entries.append(
            GroundTruthEntry(
                image_name=image_name,
                image_path=image_path,
                reference_mask_path=mask_path,
            )
        )
    return GroundTruthManifest(reference_dir=reference_path, entries=entries)


def save_ground_truth_manifest(
    manifest: GroundTruthManifest,
    output_dir: str | Path,
    *,
    basename: str = MANIFEST_FILENAME,
) -> Path:
    """Write the ground-truth manifest JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = (destination / basename).resolve()
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load_ground_truth_manifest(path: str | Path) -> GroundTruthManifest:
    """Load a ground-truth manifest JSON file."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return GroundTruthManifest.from_dict(payload)


def load_reference_mask(path: str | Path) -> np.ndarray:
    """Load a reference mask as a boolean 2D array."""
    array = np.asarray(read_tiff(path))
    if array.ndim == 3 and array.shape[0] <= 4:
        array = array[0]
    elif array.ndim == 3:
        array = array[..., 0]
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Reference mask must reduce to 2D, got shape {array.shape}")
    return array > 0
