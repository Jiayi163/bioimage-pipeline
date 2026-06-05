# Fiji / ImageJ TIFF Export (Phase 12 — Fallback / Intermediate)

Phase 12 provides **Python-based** ImageJ-compatible TIFF writing. This is a
**fallback and intermediate** export path — not the production final-export path.

For production workflows, **Phase 14** adds headless Fiji/ImageJ export to
produce final TIFFs. See [fiji_headless_export.md](fiji_headless_export.md).

## Project roles

| Layer | Role |
|-------|------|
| **CellProfiler** | Primary analysis engine — run any `.cppipe` pipeline headlessly |
| **Fiji/ImageJ** | Primary export engine (Phase 14) — headless final TIFF production |
| **Python TIFF export** | Fallback — ImageJ-compatible writes via `tifffile` when Fiji is unavailable |
| **Python analysis engine** | Optional lightweight workflow for teaching and quick tests |

CellProfiler functionality is supported by **running CellProfiler**, not by
reimplementing its modules in Python.

## When to use Python TIFF export

| Scenario | Export path |
|----------|-------------|
| Fiji installed, production workflow | **One Fiji run** + batch folder macro (Phase 14) |
| Fiji not installed | Python in-process fallback (`export.py`, `fiji_tiff.py`) |
| Unit tests and CI | Python in-process fallback (fast, no JVM) |
| Python-only prototyping | Python in-process fallback |

Python fallback loops per file **in-process** — this is acceptable because it
does not relaunch CellProfiler or Fiji. Per-image **Fiji subprocess** launches
are anti-pattern unless batch macros cannot handle the case.

## Export formats

### Ordinary TIFF

`bioimage_pipeline.io.save_tiff(..., imagej_compatible=False)`

- Plain TIFF written with `tifffile`
- Good for internal round-trips and simple storage
- No ImageJ-specific metadata tags

### ImageJ-compatible TIFF (Python fallback)

`export_mask_tiff`, `export_label_tiff`, `export_intensity_tiff`, and
`save_fiji_compatible_tiff`

- Writes ImageJ-compatible tags via `tifffile` (`imagej=True`)
- Optional metadata:
  - pixel size (`pixel_size_x`, `pixel_size_y`)
  - unit (default `um`)
  - channel name
  - image description
- Used for masks, labels, and intensity when Fiji headless export is not run

### OME-TIFF (future)

OME-TIFF with full microscopy metadata is **not implemented yet**. Use Fiji
headless export (Phase 14) or ImageJ-compatible Python TIFF today.

## Dtype conventions

| Output type | Dtype | Notes |
|-------------|-------|-------|
| Binary mask | `uint8` | Values `0` and `255` |
| Label image | `uint16` or `uint32` | `uint16` + ImageJ tags when max label ≤ 65535; `uint32` uses standard TIFF (ImageJ format does not support 32-bit labels) |
| Intensity image | preserved integer dtype | `float32` for unsupported float types |

## Example

```python
from bioimage_pipeline.export import export_label_tiff, export_mask_tiff
from bioimage_pipeline.fiji_tiff import TiffExportMetadata

metadata = TiffExportMetadata(
    pixel_size_x=0.65,
    pixel_size_y=0.65,
    unit="um",
    channel_name="Nuclei",
    description="Watershed labels",
)

export_mask_tiff("output/mask.tif", mask, metadata=metadata)
export_label_tiff("output/labels.tif", labels, metadata=metadata)
```

Read metadata back in Python:

```python
from bioimage_pipeline.fiji_tiff import read_fiji_tiff_metadata

info = read_fiji_tiff_metadata("output/labels.tif")
print(info.get("pixel_size"), info.get("channel_name"))
```

## Workflow placement

```text
CellProfiler .cppipe — ONE run per folder
    ↓
CSV + raw TIFFs (cellprofiler_raw/) — all outputs collected
    ↓
[Fiji headless — ONE run + batch macro — Phase 14, production path]
    ↓
final masks/, labels/

    — or when Fiji unavailable —

[Python in-process fallback — Phase 12]
    ↓
organize_cellprofiler_tiffs_for_fiji() → masks/, labels/
```

## Tests

```bash
pytest tests/test_fiji_tiff.py tests/test_export.py -v
```
