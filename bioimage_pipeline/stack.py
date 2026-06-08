"""Image stack data model and loaders (Phase S.2).

Provides a unified :class:`ImageStack` abstraction that can be populated from:

* A **single multi-page TIFF** file (Z-stack, time series, hyperstack).
* A **folder of 2D TIFF files** — each file becomes one frame, sorted
  alphabetically (Fiji-style virtual stack from folder).

Use :func:`load_stack` to auto-detect file vs folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bioimage_pipeline.io import (
    AxisInfo,
    StackFrame,
    extract_2d_plane,
    interpret_tiff_axes,
    iter_stack_frames,
    read_tiff,
)


def _collect_tiff_paths(folder: Path, pattern: str) -> list[Path]:
    paths = sorted(folder.glob(pattern))
    if pattern == "*.tif":
        extra = sorted(folder.glob("*.tiff"))
        seen = {p.name for p in paths}
        paths = paths + [p for p in extra if p.name not in seen]
        paths = sorted(paths)
    return [p for p in paths if p.is_file()]


@dataclass
class ImageStack:
    """An ordered sequence of 2D frames from a TIFF file or a folder.

    Attributes:
        frames: Ordered list of :class:`~bioimage_pipeline.io.StackFrame` objects.
        source: Path to the originating file or folder.
        axis_info: Optional axis metadata (Z / T / C counts). Populated from
            ImageJ TIFF metadata when loading from a single file; ``None`` for
            folder-based stacks.
    """

    frames: list[StackFrame]
    source: Path
    axis_info: AxisInfo | None = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total number of 2D frames in the stack."""
        return len(self.frames)

    @property
    def shape(self) -> tuple[int, int] | None:
        """``(height, width)`` of one frame, or ``None`` for an empty stack."""
        if not self.frames:
            return None
        arr = self.frames[0].array
        return (arr.shape[0], arr.shape[1])

    # ------------------------------------------------------------------
    # Iteration and indexing
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[StackFrame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> StackFrame:
        return self.frames[index]


def load_stack_from_tiff(path: str | Path) -> ImageStack:
    """Load all pages of a TIFF file as an :class:`ImageStack`.

    Single-page TIFFs produce a one-frame stack.  Multi-page TIFFs (Z-stacks,
    time series, hyperstacks) produce one :class:`StackFrame` per page.
    ImageJ axis metadata is used when present.

    Args:
        path: Path to the TIFF file.

    Returns:
        :class:`ImageStack` with one frame per page.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file.
        OSError: If the TIFF cannot be opened.
    """
    tiff_path = Path(path)
    frames = list(iter_stack_frames(tiff_path))
    axis_info: AxisInfo | None = None
    if frames:
        axis_info = frames[0].metadata.get("axes")
    return ImageStack(frames=frames, source=tiff_path, axis_info=axis_info)


def load_stack_from_folder(
    folder: str | Path,
    pattern: str = "*.tif",
) -> ImageStack:
    """Load one frame per TIFF file in a folder as an :class:`ImageStack`.

    Files are discovered with ``pattern`` (plus ``*.tiff`` when
    ``pattern="*.tif"``) and sorted alphabetically — matching Fiji's
    *virtual stack from folder* behavior.  If a file contains multiple pages
    only its first plane is used; use :func:`load_stack_from_tiff` on
    individual files to extract all pages.

    Args:
        folder: Path to the folder containing TIFF images.
        pattern: Glob pattern for image discovery (default ``"*.tif"``).

    Returns:
        :class:`ImageStack` with one frame per file.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If no TIFF files are found.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Expected a directory, got a file: {folder_path}")

    paths = _collect_tiff_paths(folder_path, pattern)
    if not paths:
        raise ValueError(
            f"No TIFF files matching {pattern!r} found in {folder_path}"
        )

    frames: list[StackFrame] = []
    for i, p in enumerate(paths):
        raw = read_tiff(p)
        plane = extract_2d_plane(raw, frame_index=0)
        frames.append(
            StackFrame(
                index=i,
                array=plane,
                z_index=i,
                t_index=0,
                c_index=0,
                source_path=p,
            )
        )

    return ImageStack(frames=frames, source=folder_path)


def load_stack(source: str | Path) -> ImageStack:
    """Load an :class:`ImageStack` from a TIFF file or a folder of TIFFs.

    Automatically dispatches to :func:`load_stack_from_tiff` when ``source``
    is a file, or :func:`load_stack_from_folder` when it is a directory.

    Args:
        source: Path to a ``.tif`` / ``.tiff`` file or a folder.

    Returns:
        :class:`ImageStack`.

    Raises:
        FileNotFoundError: If the source path does not exist.
    """
    path = Path(source)
    if path.is_file():
        return load_stack_from_tiff(path)
    if path.is_dir():
        return load_stack_from_folder(path)
    raise FileNotFoundError(f"Source not found: {path}")
