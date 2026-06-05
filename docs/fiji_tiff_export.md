# Fiji / ImageJ TIFF Export

This project does **not** embed Fiji or ImageJ. Instead, it writes TIFF files
that open correctly in Fiji/ImageJ for visual QC and downstream workflows.

## Project roles

| Layer | Role |
|-------|------|
| **CellProfiler** | Full analysis engine — run any `.cppipe` pipeline headlessly |
| **Python engine** | Lightweight/simple workflow for teaching and quick tests |
| **Fiji/ImageJ** | Final inspection target — open exported TIFFs manually |

CellProfiler functionality is supported by **running CellProfiler**, not by
reimplementing its modules in Python.

## Export formats

### Ordinary TIFF

`bioimage_pipeline.io.save_tiff(..., imagej_compatible=False)`

- Plain TIFF written with `tifffile`
- Good for internal round-trips and simple storage
- No ImageJ-specific metadata tags

### ImageJ-compatible TIFF (default for export helpers)

`export_mask_tiff`, `export_label_tiff`, `export_intensity_tiff`, and
`save_fiji_compatible_tiff`

- Writes ImageJ-compatible tags via `tifffile` (`imagej=True`)
- Optional metadata:
  - pixel size (`pixel_size_x`, `pixel_size_y`)
  - unit (default `um`)
  - channel name
  - image description
- Recommended for masks, labels, and intensity images you plan to open in Fiji

### OME-TIFF (future)

OME-TIFF with full microscopy metadata is **not implemented yet**. Use
ImageJ-compatible TIFF today. OME-TIFF is listed as a later enhancement when
multi-channel calibration and richer metadata are required.

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

## When actual Fiji/ImageJ integration would be needed

Call Fiji/ImageJ directly only if you need features TIFF export cannot provide:

- Live macro execution inside ImageJ
- Proprietary import formats (some via Bio-Formats)
- Interactive plugin workflows

For this project, the default path is:

```text
CellProfiler .cppipe  →  CSV + mask/label TIFFs  →  open in Fiji manually
Python pipeline      →  Fiji-compatible TIFF export  →  open in Fiji manually
```

## Tests

```bash
pytest tests/test_fiji_tiff.py tests/test_export.py -v
```
