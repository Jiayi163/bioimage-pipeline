# CellProfiler Workflow Integration (Phase 13)

Phase 13 orchestrates **CellProfiler as the primary analysis engine** with
**one headless run per input folder**. It does not reimplement CellProfiler
modules. Final TIFF export through Fiji/ImageJ headless is **Phase 14** (one Fiji
run per folder); this phase currently uses Python TIFF export as an in-process
fallback.

## Performance: batch-first

| Stage | Invocations | Implementation |
|-------|-------------|----------------|
| CellProfiler | **1 per folder** | `run_cellprofiler_pipeline_logged()` with `-i input_dir` |
| Output collection | After CP completes | Scan `cellprofiler_raw/` for all CSVs and TIFFs |
| Python TIFF fallback | In-process (no CP/Fiji relaunch) | `organize_cellprofiler_tiffs_for_fiji()` |
| QC overlays | In-process per image | `generate_qc_for_cellprofiler_results()` |
| Stage timing | Phase 14 | `timing.*_seconds` in `workflow_summary.json` |

**Never** spawn CellProfiler once per image file from the workflow API.

## OIR projection cache (output folder)

When the input folder contains Olympus `.oir` stacks, the workflow runs a
**prepare_input** step: Fiji (or Python) Z-max projection into
`{output_dir}/oir_projection/` before CellProfiler sees TIFFs.

Re-runs to the **same output folder** reuse existing projected TIFFs when:

- the expected `{stem}.tif` exists under `oir_projection/`,
- it is non-empty, and
- its modification time is not older than the source `.oir` (with a small
  **2-second tolerance** for Windows/FAT32 timestamp rounding).

When all pairs hit the cache, Fiji is **not** launched for OIR projection
(`action: oir_projection_cache_hit` in `logs/prepare_input_profile.txt`).

Debug artifacts (per-pair paths, sizes, mtimes, cache decision):

- `logs/oir_projection_cache_debug.txt`
- `logs/oir_projection_cache_debug.json`

To force reprojection (ignore cache): pass `force_oir_reproject=True` to
`run_cellprofiler_workflow()` / `CellProfilerWorkflowConfig`.

Standalone batch tool: `examples/run_oir_zmax_batch.py` (same cache logic in
`bioimage_pipeline/oir_zmax_batch.py`).

## Full workflow (target architecture)

```text
Input Images (folder)
    ↓
CellProfiler — ONE run (.cppipe, headless)     ← Phase 13 (this doc)
    ↓  collect all outputs
Fiji/ImageJ — ONE run (batch macro)            ← Phase 14
    ↓
Organized results + QC + logs (with timings)
```
## What Phase 13 does today

1. Accept a `.cppipe` pipeline file and an input image folder.
2. Run CellProfiler headlessly **once** over the folder (`-c -r -p -i -o`).
3. Capture stdout, stderr, command, and exit code under `logs/`.
4. Locate **all** CellProfiler CSV and TIFF outputs in `cellprofiler_raw/`.
5. Copy measurement CSVs into `measurements/` and optionally merge them.
6. Convert mask/label TIFFs via **Python in-process fallback** under `masks/` and
   `labels/` (until Phase 14 batch Fiji export is wired in).
7. Generate QC overlay PNGs under `qc/` (in-process, no external relaunch).
8. Write `logs/workflow_summary.json` with a full job summary (timings in Phase 14).
## Results folder layout

```text
results/
  measurements/        # CSV tables (+ merged_measurements.csv)
  masks/               # Mask TIFFs (Python fallback today; Fiji final in Phase 14)
  labels/              # Label TIFFs (Python fallback today; Fiji final in Phase 14)
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

Skip Python fallback TIFF conversion or QC overlays:

```bash
python examples/run_cellprofiler_workflow.py ^
  --cppipe path\to\pipeline.cppipe ^
  --input-dir path\to\images ^
  --output-dir path\to\results ^
  --no-fiji-export ^
  --no-qc
```

Note: `--no-fiji-export` disables the Python fallback TIFF organization step
(naming retained for backward compatibility). Phase 14 will add a separate
Fiji headless export flag.

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

## QC and manual inspection

Open TIFFs from `masks/` and `labels/` in Fiji/ImageJ for manual QC. Use `qc/`
overlays for quick visual checks. See [fiji_qc_workflow.md](fiji_qc_workflow.md).

When Phase 14 is complete, `masks/` and `labels/` will contain Fiji-exported
final TIFFs by default.

## Optional Python analysis fallback

For teaching or quick tests without CellProfiler installed:

```python
from bioimage_pipeline.analysis import run_analysis

result = run_analysis("path/to/images", "path/to/output", analysis_engine="python")
```

This path does not replace CellProfiler analysis — it is a simple built-in
pipeline for prototyping and unit tests.

## Related docs

- [gui_direction.md](gui_direction.md) — Phase 15 GUI: expose CP modules, do not reimplement
- [fiji_headless_export.md](fiji_headless_export.md) — Phase 14 Fiji export plan
- [fiji_tiff_export.md](fiji_tiff_export.md) — Python fallback TIFF format details
