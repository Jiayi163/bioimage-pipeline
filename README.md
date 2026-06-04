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

**Planned later:** advanced self-adaptive thresholding (see `DEVELOPMENT_PLAN.md` Phase 11).

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

## CellProfiler integration

Install [CellProfiler](https://cellprofiler.org/) separately and ensure the
`cellprofiler` command is on your PATH. Prepare a `.cppipe` pipeline file and
folders of input images.

```python
from bioimage_pipeline.cellprofiler_runner import (
    read_cellprofiler_csv,
    run_cellprofiler_pipeline,
)

output_dir = run_cellprofiler_pipeline(
    "pipelines/basic.cppipe",
    "input_images",
    "cellprofiler_output",
)

measurements = read_cellprofiler_csv(output_dir / "MyMeasurements.csv")
```

Headless command equivalent:

```text
cellprofiler -c -r -p pipeline.cppipe -i input_images -o cellprofiler_output
```

The built-in lightweight pipeline (`Pipeline`, `batch.py`) remains available for
small workflows that do not require CellProfiler.

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for phased development status.
