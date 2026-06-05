# bioimage-pipeline

A **workflow orchestration layer** that **exposes CellProfiler functionality**
through pipeline management and a future GUI — without reimplementing
CellProfiler algorithms. CellProfiler is the primary analysis engine. Fiji/ImageJ
handles export/QC when required. This project manages configuration, subprocess
runs, output organization, logs, and QC artifacts.

It does not import or copy code from CellProfiler or Fiji/ImageJ.

| Layer | Role |
|-------|------|
| **CellProfiler** | Primary analysis engine — segmentation, measurement, feature extraction via `.cppipe` |
| **Fiji/ImageJ** | Export/QC engine — headless final TIFF export when required (Phase 14) |
| **This project** | Orchestration + future GUI front-end — pipeline management, config, logs, QC, results |
| **Python TIFF export** | Fallback / intermediate — ImageJ-compatible writes when Fiji is unavailable |
| **Python analysis engine** | Optional lightweight fallback for teaching, tests, and prototypes |

**This project is not a replacement for CellProfiler or Fiji.** It exposes their
capabilities through workflow control and a GUI that manages `.cppipe` pipelines:

```text
GUI → .cppipe config → CellProfiler (once/folder) → Fiji (optional, once/folder) → results
```

## Performance: batch-first execution

External tools (CellProfiler, Fiji) are expensive to start. The workflow is
designed for **batch invocation**, not one launch per image:

1. **One CellProfiler run** per input folder (`-i input_dir -o output_dir`).
2. **Collect all outputs** from `cellprofiler_raw/` before export.
3. **One Fiji headless run** per folder with a batch macro (Phase 14).
4. **Per-image export** only when batch macros cannot handle the case.
5. **Python TIFF fallback** stays in-process (no JVM startup) for tests and when
   Fiji is unavailable.
6. **Stage timings** logged in `workflow_summary.json` (Phase 14): CellProfiler,
   Fiji export, and QC generation runtime.

## Architecture

| Layer | Role | Modules |
|-------|------|---------|
| Workflow orchestration | End-to-end CP → Fiji path | `analysis.py`, `cellprofiler_runner.py`, `qc.py` |
| CellProfiler integration | Headless `.cppipe` runs, CSV import | `cellprofiler_runner.py` |
| Fiji headless export (Phase 14) | Export/QC — final TIFF production | `fiji_runner.py` (planned) |
| Python TIFF fallback | Intermediate / no-Fiji export | `fiji_tiff.py`, `export.py`, `io.py` |
| QC visualization | Overlays and inspection helpers | `qc.py` |
| GUI (Phase 15) | CP/Fiji front-end — test UI (15.0), shell (15.1), builder (15.2) | `workflow_test_ui.py`, `gui/`, `cppipe_io.py` (planned) |
| Python analysis fallback | Teaching / prototyping only | `preprocess.py`, `threshold.py`, `segment.py`, `measure.py`, `pipeline.py`, `batch.py` |

```text
input images + .cppipe
    ↓
CellProfiler (one headless run per folder — primary analysis)
    ↓  measurements/, cellprofiler_raw/  (all images collected)
Fiji/ImageJ (one headless run per folder — final TIFF export)   ← Phase 14
    ↓  masks/, labels/ (final TIFFs)
QC overlays + logs + organized results
    ↓
optional manual inspection in Fiji GUI
```

**Implemented today:** TIFF I/O, Python TIFF fallback, CellProfiler CLI runner,
CellProfiler workflow orchestration (Phase 13), QC overlays, and example scripts.

**Next priority:** Phase 14 — Fiji/ImageJ headless export integration.

| Phase | Focus | Status |
|-------|--------|--------|
| 0–11 | Core Python pipeline, CP validation, segmentation | Complete |
| 12 | Python TIFF export (fallback / intermediate) | Complete |
| 13 | CellProfiler workflow integration | Complete |
| 14 | Fiji/ImageJ headless export integration | **Next** |
| 15.0 | Temporary Streamlit workflow test UI | Complete |
| 15.1 | GUI workflow shell (run, logs, preview) | Not started |
| 15.2 | GUI pipeline builder & CP module exposure | Not started |
| 16 | Optional Python analysis enhancements | Not started |
| 17 | Self-adaptive threshold at import (hybrid CP) — core differentiator | Deferred — prototype only |

### GUI direction (Phase 15)

Phase **15.0** ships first: a temporary Streamlit UI to click-run workflows and
see logs, overlays, and measurements — not the final GUI. **15.1** is the proper
workflow shell; **15.2** adds pipeline building and CellProfiler module exposure.
The GUI **exposes CellProfiler functionality** without reimplementing algorithms
like `IdentifyPrimaryObjects` or `MeasureObjectSizeShape`. See
[docs/gui_direction.md](docs/gui_direction.md).

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
    export.py              # Python TIFF fallback + CP output organization
    pipeline.py
    batch.py
    cellprofiler_runner.py # CellProfiler headless runner
    analysis.py            # Workflow orchestration
    validation.py
    qc.py
    fiji_tiff.py           # Python ImageJ-compatible TIFF writer
    fiji_runner.py         # (Phase 14) Fiji headless export — planned
    cppipe_io.py           # (Phase 15.2) Load/save .cppipe — planned
    pipeline_catalog.py    # (Phase 15.2) CP module metadata — planned
examples/
    run_basic_pipeline.py
    run_analysis.py
    visual_check.py
    validate_cellprofiler.py
    run_cellprofiler_workflow.py
    validate_real_data.py
app/
    workflow_test_ui.py    # Phase 15.0 Streamlit test UI
    touching_objects_demo.py
docs/
    cellprofiler_validation.md
    cellprofiler_workflow.md
    fiji_headless_export.md
    fiji_tiff_export.md
    fiji_qc_workflow.md
    gui_direction.md
    real_data_validation.md
tests/
```

## Development

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Run the Phase 15.0 workflow test UI (local browser app — requires CellProfiler):

```bash
pip install -e ".[ui]"
streamlit run app/workflow_test_ui.py
```

Run tests:

```bash
pytest
```

Run the basic example:

```bash
python examples/run_basic_pipeline.py
```

Run the visual validation script (outputs and QC overlays):

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

load_result = load_cellprofiler_measurements(output_dir)
merged, warnings = merge_cellprofiler_tables(
    load_result.tables,
    metadata=load_result.metadata,
)
```

Headless command equivalent:

```text
cellprofiler -c -r -p pipeline.cppipe -i path/to/images -o path/to/cellprofiler_output
```

The built-in lightweight pipeline (`Pipeline`, `batch.py`) remains available as
an optional fallback for workflows that do not require CellProfiler.

### Unified analysis mode (Phase 10.3)

Use one function to switch between engines:

```python
from bioimage_pipeline.analysis import run_analysis

# Python engine (optional fallback)
result = run_analysis("path/to/images", "path/to/output", analysis_engine="python")
print(result["processed"])

# CellProfiler engine (primary)
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

### CellProfiler workflow (Phase 13)

Orchestrates **one CellProfiler headless run per input folder** and organizes
results. Today, mask/label TIFFs are written via the **Python in-process fallback**
exporter; Phase 14 will add **one Fiji batch run per folder** for final TIFFs.

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

### Fiji headless export (Phase 14 — planned)

Install [Fiji](https://fiji.sc/) separately. Phase 14 will run Fiji **once per
folder** with batch macros to produce final TIFF outputs from CellProfiler
results. Python TIFF export remains a fast in-process fallback when Fiji is
unavailable. Per-image Fiji launches are fallback-only, not the default path.

See [docs/fiji_headless_export.md](docs/fiji_headless_export.md) for the
integration plan, export-path comparison, and planned API.

### Real data validation (Phase 10.4)

Validate microscopy TIFFs and compare Python outputs against reference masks or
CellProfiler results:

```bash
python examples/validate_real_data.py --input-dir path/to/real_images --output-dir path/to/validation_output
```

See [docs/real_data_validation.md](docs/real_data_validation.md) for fixture
details, limitations, and failure cases.

### Advanced segmentation (Phase 11 — optional Python engine)

Split touching objects with watershed labeling:

```python
from bioimage_pipeline.analysis import run_analysis

run_analysis("path/to/images", "path/to/output", labeling_method="watershed")
```

Compare connected vs watershed on a synthetic touching-objects demo:

```bash
python examples/touching_objects_demo.py
```

### Python TIFF fallback (Phase 12)

When Fiji is not used, masks, labels, and intensity images can be exported with
ImageJ-compatible tags via Python:

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

See [docs/fiji_tiff_export.md](docs/fiji_tiff_export.md) for format details and
the fallback vs Fiji headless export comparison.

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for phased development status,
acceptance criteria, and checklists.
