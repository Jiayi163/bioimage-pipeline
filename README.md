# bioimage-pipeline

A lightweight bioimage analysis framework inspired by **Fiji/ImageJ** (TIFF I/O and
visual inspection) and **CellProfiler** (modular analysis workflow). It does not
import or copy code from either project.

**This project is not a replacement for CellProfiler.** For heavy image-analysis
pipelines, use CellProfiler as an external engine via `.cppipe` files. This package
provides a thin Python wrapper for batch workflow, TIFF I/O, and reading results.

## Architecture

| Layer | Role | Modules |
|-------|------|---------|
| TIFF I/O and Fiji-friendly export | Read/write and inspect in Fiji | `io.py`, `export.py` |
| Lightweight Python pipeline | Simple teaching / prototyping workflow | `preprocess.py`, `threshold.py`, `segment.py`, `measure.py`, `pipeline.py`, `batch.py` |
| CellProfiler integration | Run `.cppipe` pipelines headless via CLI | `cellprofiler_runner.py` |

```text
Python wrapper / batch workflow
    ↓
calls CellProfiler .cppipe (subprocess, headless)
    ↓
reads CellProfiler CSV / mask outputs
    ↓
exports or visualizes results (TIFF, pandas, Fiji)
```

**Implemented:** TIFF read/write, lightweight pipeline, CellProfiler CLI runner,
and example scripts.

**Roadmap:** Phases 0–10 are complete. Next work is split into smaller steps
before advanced algorithms:

| Phase | Focus |
|-------|--------|
| 10.1–10.2 | CellProfiler validation and CSV import |
| 10.3 | Unified mode (`analysis_engine`: `python` or `cellprofiler`) |
| 10.4–10.5 | Real-data validation and QC visualization |
| 11 | Advanced segmentation (watershed, touching objects) |
| 12 | Adaptive thresholding improvements |
| 13 | GUI |

Advanced thresholding is **Phase 12**, not the immediate next step after Phase 10.
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
examples/
    run_basic_pipeline.py
    visual_check.py
    validate_cellprofiler.py
docs/
    cellprofiler_validation.md
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

Run the visual validation script (outputs for Fiji inspection):

```bash
python examples/visual_check.py
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

### Roadmap summary

| Phase | Goal | Status |
|-------|------|--------|
| 10.1 | CellProfiler integration validation | Complete |
| 10.2 | CellProfiler output import | Complete |
| 10.3 | Unified analysis mode | Not complete |
| 10.4 | Real data validation | Not complete |
| 10.5 | Visualization and QC | Not complete |
| 11 | Advanced segmentation | Not complete |
| 12 | Adaptive thresholding | Not complete |
| 13 | GUI | Not complete |

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for phased development status,
acceptance criteria, and checklists.
