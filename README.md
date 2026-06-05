# bioimage-pipeline

A lightweight **CellProfiler-to-Fiji workflow tool**. CellProfiler performs the
full analysis through `.cppipe` pipelines. This project manages inputs, runs
CellProfiler headlessly, collects outputs, standardizes TIFFs for Fiji/ImageJ,
and organizes results into a clean folder layout. It does not import or copy
code from CellProfiler or Fiji.

| Layer | Role |
|-------|------|
| **CellProfiler** | Full analysis engine — all functionality via headless `.cppipe` runs |
| **This project** | Orchestration, output collection, Fiji-compatible TIFF export, QC overlays |
| **Fiji/ImageJ** | Manual QC/viewing target (no embedded Fiji runtime) |
| **Python engine** | Lightweight/simple fallback for teaching, tests, and quick prototypes |

**This project is not a replacement for CellProfiler.** It wires CellProfiler
outputs into a repeatable workflow ending in Fiji-inspectable TIFFs and CSV
measurements.

## Architecture

| Layer | Role | Modules |
|-------|------|---------|
| CellProfiler-to-Fiji workflow | Primary end-to-end path | `analysis.py`, `cellprofiler_runner.py`, `export.py` |
| TIFF I/O and Fiji-friendly export | ImageJ-compatible masks, labels, intensity | `io.py`, `export.py`, `fiji_tiff.py` |
| QC visualization | Mask/label overlays and Fiji inspection workflow | `qc.py` |
| Lightweight Python pipeline | Simple fallback for teaching / prototyping | `preprocess.py`, `threshold.py`, `segment.py`, `measure.py`, `pipeline.py`, `batch.py` |

```text
input images + .cppipe
    ↓
CellProfiler (headless, full analysis engine)
    ↓
organized results: measurements/, masks/, labels/, qc/, logs/
    ↓
open TIFFs in Fiji/ImageJ for manual QC
```

**Implemented:** TIFF read/write, lightweight pipeline, CellProfiler CLI runner,
and example scripts.

**Roadmap:** Phases 0–10 are complete. Next work is split into smaller steps
before advanced algorithms:

| Phase | Focus |
|-------|--------|
| 10.1–10.2 | CellProfiler validation and CSV import |
| 10.3 | Unified mode (`analysis_engine`: `python` or `cellprofiler`) — complete |
| 10.4 | Real-data validation — complete |
| 10.5 | QC visualization — complete |
| 11 | Advanced segmentation (watershed, touching objects) — complete |
| 12 | Fiji/ImageJ-compatible TIFF export — complete |
| 13 | CellProfiler-to-Fiji workflow integration — complete |
| 14 | GUI (Streamlit / PyQt) |
| 15 | Advanced CellProfiler support (templates, presets, batch jobs) |
| 16 | Optional Python enhancements (adaptive thresholding, etc.) |

Phases **0–13** are complete. **Phase 14** (GUI) is next.
See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for full phase details and status.

## Project Layout

```text
bioimage_pipeline/
    __init__.py
    io.py
    preprocess.py
    threshold.py
    segment.py
    measure.py
    export.py
    pipeline.py
    batch.py
    cellprofiler_runner.py
    analysis.py
    validation.py
    qc.py
    fiji_tiff.py
examples/
    run_basic_pipeline.py
    run_analysis.py
    visual_check.py
    validate_cellprofiler.py
    run_cellprofiler_workflow.py
    validate_real_data.py
    touching_objects_demo.py
docs/
    cellprofiler_validation.md
    cellprofiler_workflow.md
    real_data_validation.md
    fiji_qc_workflow.md
    fiji_tiff_export.md
tests/
```

## Development

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run the basic example:

```bash
python examples/run_basic_pipeline.py
```

Run the visual validation script (outputs and QC overlays for Fiji inspection):

```bash
python examples/visual_check.py
pytest tests/test_qc.py -v
```

See [docs/fiji_qc_workflow.md](docs/fiji_qc_workflow.md) for the step-by-step
Fiji inspection checklist. Optional interactive viewing:

```bash
pip install -e ".[qc]"
```

Validate CellProfiler integration (requires a local `.cppipe` file):

```bash
python examples/validate_cellprofiler.py --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/output
```

See [docs/cellprofiler_validation.md](docs/cellprofiler_validation.md) for the
Phase 10.1 checklist.

## CellProfiler integration

Install [CellProfiler](https://cellprofiler.org/) separately. By default the
runner uses the `cellprofiler` command on your PATH. On Windows you can pass the
full path to `CellProfiler.exe` via `cellprofiler_executable`.

```python
from bioimage_pipeline.cellprofiler_runner import (
    load_cellprofiler_measurements,
    merge_cellprofiler_tables,
    read_cellprofiler_csv,
    run_cellprofiler_pipeline,
)

output_dir = run_cellprofiler_pipeline(
    cppipe_path="path/to/pipeline.cppipe",
    input_dir="path/to/images",
    output_dir="path/to/cellprofiler_output",
)

# Windows example with explicit executable path:
output_dir = run_cellprofiler_pipeline(
    cppipe_path="path/to/pipeline.cppipe",
    input_dir="path/to/images",
    output_dir="path/to/cellprofiler_output",
    cellprofiler_executable=r"C:\Program Files\CellProfiler\CellProfiler.exe",
)

tables = load_cellprofiler_measurements(output_dir)
merged = merge_cellprofiler_tables(tables)
```

Headless command equivalent:

```text
cellprofiler -c -r -p pipeline.cppipe -i path/to/images -o path/to/cellprofiler_output
```

The built-in lightweight pipeline (`Pipeline`, `batch.py`) remains available for
small workflows that do not require CellProfiler.

### Unified analysis mode (Phase 10.3)

Use one function to switch between engines:

```python
from bioimage_pipeline.analysis import run_analysis

# Python engine (default built-in pipeline)
result = run_analysis("path/to/images", "path/to/output", analysis_engine="python")
print(result["processed"])

# CellProfiler engine
result = run_analysis(
    "path/to/images",
    "path/to/cellprofiler_output",
    analysis_engine="cellprofiler",
    cppipe_path="path/to/pipeline.cppipe",
    cellprofiler_executable=r"C:\Program Files\CellProfiler\CellProfiler.exe",
)
print(result["measurements"])
```

CLI example:

```bash
python examples/run_analysis.py --input-dir path/to/images --output-dir path/to/output
python examples/run_analysis.py --engine cellprofiler --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/output
```

### CellProfiler-to-Fiji workflow (Phase 13)

Primary workflow: `.cppipe` → headless run → organized results for Fiji QC:

```python
from bioimage_pipeline.analysis import run_cellprofiler_workflow

result = run_cellprofiler_workflow(
    "path/to/images",
    "path/to/results",
    "path/to/pipeline.cppipe",
    cellprofiler_executable=r"C:\Program Files\CellProfiler\CellProfiler.exe",
)
print(result.measurements_dir)
print(result.mask_exports)
print(result.qc_artifacts)
```

```bash
python examples/run_cellprofiler_workflow.py --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/results
```

Results layout: `measurements/`, `masks/`, `labels/`, `qc/`, `logs/`,
`cellprofiler_raw/`. See [docs/cellprofiler_workflow.md](docs/cellprofiler_workflow.md).

### Real data validation (Phase 10.4)

Validate microscopy TIFFs and compare Python outputs against reference masks or
CellProfiler results:

```bash
python examples/validate_real_data.py --input-dir path/to/real_images --output-dir path/to/validation_output
```

See [docs/real_data_validation.md](docs/real_data_validation.md) for fixture
details, limitations, and failure cases.

### Advanced segmentation (Phase 11)

Split touching objects with watershed labeling:

```python
from bioimage_pipeline.analysis import run_analysis

run_analysis("path/to/images", "path/to/output", labeling_method="watershed")
```

Compare connected vs watershed on a synthetic touching-objects demo:

```bash
python examples/touching_objects_demo.py
```

### Roadmap summary

| Phase | Goal | Status |
|-------|------|--------|
| 10.1 | CellProfiler integration validation | Complete |
| 10.2 | CellProfiler output import | Complete |
| 10.3 | Unified analysis mode | Complete |
| 10.4 | Real data validation | Complete |
| 10.5 | Visualization and QC | Complete |
| 11 | Advanced segmentation | Complete |
| 12 | Fiji/ImageJ-compatible TIFF export | Complete |
| 13 | CellProfiler-to-Fiji workflow integration | Complete |
| 14 | GUI (Streamlit / PyQt) | Not complete |
| 15 | Advanced CellProfiler support | Not complete |
| 16 | Optional Python enhancements | Not complete |

### Fiji/ImageJ-compatible TIFF export (Phase 12)

Masks, labels, and intensity images are exported with ImageJ-compatible tags:

```python
from bioimage_pipeline.export import export_label_tiff, export_mask_tiff
from bioimage_pipeline.fiji_tiff import TiffExportMetadata

metadata = TiffExportMetadata(
    pixel_size_x=0.65,
    pixel_size_y=0.65,
    unit="um",
    channel_name="Nuclei",
)
export_mask_tiff("output/mask.tif", mask, metadata=metadata)
```

See [docs/fiji_tiff_export.md](docs/fiji_tiff_export.md) for ordinary TIFF vs
ImageJ TIFF vs future OME-TIFF.

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for phased development status,
acceptance criteria, and checklists.
