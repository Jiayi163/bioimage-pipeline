# bioimage-pipeline

A **lightweight bioimage analysis pipeline** inspired by Fiji/ImageJ and
CellProfiler. It provides a Python-native stack and batch-processing workflow,
a CellProfiler integration layer, and a Fiji-compatible TIFF export path.
This project manages pipeline configuration, subprocess orchestration, output
organization, logs, and QC artifacts.

| Layer | Role |
|-------|------|
| **CellProfiler** | Primary analysis engine — segmentation, measurement, feature extraction via `.cppipe` |
| **Fiji/ImageJ** | Export/QC engine — headless final TIFF export when required (Phase 14) |
| **This project** | Orchestration + future GUI front-end — pipeline management, config, logs, QC, results |
| **Python TIFF export** | Fallback / intermediate — ImageJ-compatible writes when Fiji is unavailable |
| **Python analysis engine** | Optional lightweight fallback for teaching, tests, and prototypes |

It can run fully standalone with its own Python pipeline, or use CellProfiler
and Fiji/ImageJ as external engines via headless subprocess calls:

```text
Python pipeline (standalone)  →  stack/folder of TIFFs  →  masks, labels, CSV
CellProfiler (external)       →  .cppipe headless run   →  measurements, exports
Fiji/ImageJ (external)        →  headless macro export  →  final TIFFs
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
| Fiji headless export (Phase 14) | Export/QC — final TIFF production | `fiji_runner.py` |
| Python TIFF fallback | Intermediate / no-Fiji export | `fiji_tiff.py`, `export.py`, `io.py` |
| QC visualization | Overlays and inspection helpers | `qc.py` |
| GUI (Phase 15) | CP/Fiji front-end — shell (15.1), builder (15.2) | `gui/`, `cppipe_io.py`, `pipeline_catalog.py` |
| Python analysis fallback | Teaching / prototyping only | `preprocess.py`, `threshold.py`, `segment.py`, `measure.py`, `pipeline.py`, `batch.py`, `stack.py`, `stack_batch.py` |

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

**Next priority:** Phase 16 — optional Python analysis enhancements.

| Phase | Focus | Status |
|-------|--------|--------|
| 0–11 | Core Python pipeline, CP validation, segmentation | Complete |
| 12 | Python TIFF export (fallback / intermediate) | Complete |
| 13 | CellProfiler workflow integration | Complete |
| 14 | Fiji/ImageJ headless export integration | Complete |
| 15.1 | GUI workflow shell (run, logs, preview) | Complete |
| 15.2 | GUI pipeline builder & CP module exposure | Complete |
| 16 | Optional Python analysis enhancements | Not started |
| 17 | Self-adaptive threshold at import (hybrid CP) — core differentiator | Deferred — prototype only |

### GUI direction (Phase 15)

Phase **15.1** is the proper workflow shell; **15.2** adds pipeline building and
CellProfiler module exposure.
The GUI exposes CellProfiler and Fiji functionality through workflow controls,
pipeline configuration, and result inspection. See
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
    fiji_runner.py         # (Phase 14) Fiji headless export
    gui/
        workflow_shell.py  # (Phase 15.1) Workflow shell
    cppipe_io.py           # (Phase 15.2) Load/save/edit .cppipe
    pipeline_catalog.py    # (Phase 15.2) CP module metadata
examples/
    run_basic_pipeline.py
    run_analysis.py
    visual_check.py
    validate_cellprofiler.py
    run_cellprofiler_workflow.py
    run_gui.py
    validate_real_data.py
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

Launch the Phase 15.1 workflow shell:

```bash
python examples/run_gui.py
```

Run tests:

```bash
pytest
```

Run the basic example:

```bash
python examples/run_basic_pipeline.py
```

### Stack / batch processing (Python engine)

Process a folder of TIFFs or a multi-page stack (Fiji-style workflow):

```bash
python examples/run_stack_batch.py --demo --output output/demo_test
python examples/run_stack_example.py
```

With a JSON recipe:

```bash
python examples/run_stack_batch.py --recipe examples/stack_batch_recipe.json
```

See [docs/stack_batch_workflow.md](docs/stack_batch_workflow.md) for the full
API, output layout, and Fiji workflow mapping.

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

### Fiji headless export (Phase 14)

Install [Fiji](https://fiji.sc/) separately. The workflow runs Fiji **once per
folder** with batch macros to produce final TIFF outputs from CellProfiler
results. Python TIFF export remains a fast in-process fallback when Fiji is
unavailable. Per-image Fiji launches are fallback-only, not the default path.

See [docs/fiji_headless_export.md](docs/fiji_headless_export.md) for the
integration details and export-path comparison.

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
