# CellProfiler-to-Fiji Workflow (Phase 13)

This project is a **lightweight CellProfiler-to-Fiji workflow tool**. CellProfiler
performs the full analysis through `.cppipe` pipelines. This package manages
inputs, runs CellProfiler headlessly, collects outputs, standardizes TIFFs for
Fiji/ImageJ, and organizes everything into a clean results folder.

Fiji is the manual QC/viewing target — it is **not** embedded in this project.
The built-in Python engine remains available as a simple fallback for teaching
and tests, but it is not the main feature.

## What the workflow does

1. Accept a `.cppipe` pipeline file and an input image folder.
2. Run CellProfiler headlessly (`-c -r -p -i -o`).
3. Capture stdout, stderr, command, and exit code under `logs/`.
4. Locate CellProfiler CSV and TIFF outputs in `cellprofiler_raw/`.
5. Copy measurement CSVs into `measurements/` and optionally merge them.
6. Convert mask/label TIFFs into Fiji-friendly files under `masks/` and `labels/`.
7. Generate QC overlay PNGs under `qc/`.
8. Write `logs/workflow_summary.json` with a full job summary.

## Results folder layout

```text
results/
  measurements/        # CSV tables (+ merged_measurements.csv)
  masks/               # Fiji-compatible mask TIFFs (0/255 uint8)
  labels/              # Fiji-compatible label TIFFs (uint16/uint32)
  qc/                  # Mask/label overlay PNGs for quick inspection
  logs/                # CellProfiler stdout/stderr and workflow summary
  cellprofiler_raw/    # Unmodified CellProfiler output files
```

## Required `.cppipe` modules

| Module | Purpose |
|--------|---------|
| **ExportToSpreadsheet** | Write measurement CSV files |
| **SaveImages** (recommended) | Export masks and labeled object images as TIFF |

Typical CSV exports:

- `MyExpt_Image.csv` — image metadata (`FileName`, `Image_Number`, …)
- `MyExpt_IdentifyPrimaryObjects.csv` — object measurements
- `MyExpt_Experiment.csv` — experiment metadata

Processed input image names are read from any table containing a `FileName` column.

## Python API

```python
from bioimage_pipeline.analysis import run_cellprofiler_workflow

result = run_cellprofiler_workflow(
    "path/to/images",
    "path/to/results",
    "path/to/pipeline.cppipe",
    cellprofiler_executable=r"C:\Program Files\CellProfiler\CellProfiler.exe",
)

print(result.processed_images)
print(result.mask_exports)
print(result.label_exports)
print(result.qc_artifacts)
print(result.to_dict())
```

## CLI example

```bash
python examples/run_cellprofiler_workflow.py ^
  --cppipe path\to\pipeline.cppipe ^
  --input-dir path\to\images ^
  --output-dir path\to\results ^
  --executable "C:\Program Files\CellProfiler\CellProfiler.exe"
```

Skip Fiji TIFF conversion or QC overlays:

```bash
python examples/run_cellprofiler_workflow.py ^
  --cppipe path\to\pipeline.cppipe ^
  --input-dir path\to\images ^
  --output-dir path\to\results ^
  --no-fiji-export ^
  --no-qc
```

## Error reporting

When CellProfiler fails, the workflow raises `RuntimeError` with stderr details
and writes logs before raising:

- `logs/cellprofiler_stdout.log`
- `logs/cellprofiler_stderr.log`
- `logs/cellprofiler_command.txt`
- `logs/cellprofiler_exit_code.txt`
- `logs/workflow_summary.json` (even on failure)

Common causes:

- Missing or invalid `cellprofiler_executable` on Windows
- Pipeline file not found
- Input directory missing or empty
- CellProfiler module errors (see stderr log)

## Fiji QC

Open TIFFs from `masks/` and `labels/` in Fiji/ImageJ. Use `qc/` overlays for
quick visual checks before opening Fiji. See
[docs/fiji_qc_workflow.md](fiji_qc_workflow.md) for the manual inspection
checklist.

## Lightweight Python fallback

For teaching or quick tests without CellProfiler installed:

```python
from bioimage_pipeline.analysis import run_analysis

result = run_analysis("path/to/images", "path/to/output", analysis_engine="python")
```

This path does not replace CellProfiler analysis — it is a simple built-in
pipeline for prototyping and unit tests.
