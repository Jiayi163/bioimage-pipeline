# Fiji / ImageJ Headless Export (Phase 14)

Phase 14 adds **headless Fiji/ImageJ export** as a first-class workflow step.
CellProfiler performs analysis; Fiji/ImageJ produces the **final TIFF outputs**.

## Performance: batch-first export

**Do not spawn Fiji once per image.** JVM/process startup dominates runtime.

| Approach | Default? | When |
|----------|----------|------|
| **One Fiji run + batch folder macro** | Yes | Production workflows |
| **Per-image Fiji subprocess** | No | Only when batch macro cannot handle format/layout |
| **Python TIFF fallback (in-process loop)** | Fallback | Fiji missing, CI/tests, simple cases |

Preferred invocation pattern:

```text
# One subprocess — macro processes entire folders
Fiji.app/ImageJ-win64.exe /headless -macro export_folder.ijm cellprofiler_raw/ masks/ labels/
```

The macro receives **input and output directory paths**, iterates files
internally, and applies Fiji-native export settings to all TIFFs in one session.

## Architecture

```text
Input Images (folder)
    ↓
CellProfiler — ONE run per folder (.cppipe, headless)
    ↓  measurements, masks, labels, raw TIFFs (all collected)
Fiji/ImageJ — ONE run per folder (headless batch macro)
    ↓  final TIFFs + Fiji metadata
Organized results (measurements/, masks/, labels/, qc/, logs/)
```

| Component | Role |
|-----------|------|
| **CellProfiler** | Primary analysis — one `.cppipe` run per folder |
| **Fiji/ImageJ** | Primary export — headless TIFF writing with native metadata and export behavior |
| **This project** | Orchestration — subprocess management, config, logs, timing, QC, future GUI |
| **Python TIFF (`fiji_tiff.py`)** | Fast in-process fallback — no JVM startup |

## Planned module: `fiji_runner.py`

Responsibilities (not yet implemented):

- Resolve Fiji/ImageJ executable path (config + auto-detect common install locations)
- Run **one headless subprocess per batch** with captured stdout/stderr
- Execute **batch folder macros** (input dir → output dirs), not one launch per file
- Fall back to per-image Fiji export only when documented as necessary
- Fall back to `export.organize_cellprofiler_tiffs_for_fiji()` when Fiji is missing
- Record `timing.fiji_export_seconds` in workflow summary
- Write export logs under `logs/fiji_*.log`

Planned API:

```python
run_fiji_batch_export(
    input_dir,          # e.g. cellprofiler_raw/ or staging dir
    masks_dir,
    labels_dir,
    macro_path="export_folder.ijm",
    fiji_executable=None,
) -> FijiExportResult
```

## Headless invocation (reference)

Platform-specific commands vary. Batch pattern:

```text
# Linux / macOS (Fiji) — one run, macro loops over folder
/path/to/Fiji.app/ImageJ-linux64 --headless -macro export_folder.ijm /path/in /path/masks /path/labels

# Windows — one run
"C:\Program Files\Fiji.app\ImageJ-win64.exe" /headless -macro export_folder.ijm C:\in C:\masks C:\labels
```

Macros will live under `examples/fiji_macros/` (planned). Each macro should:

- Accept folder paths as arguments (not single file paths)
- Process all matching TIFFs in the input folder
- Write final exports to the output folders with consistent naming

## Workflow integration

`run_cellprofiler_workflow()` (Phase 13) currently:

1. Runs CellProfiler **once** per folder (complete)
2. Collects all outputs from `cellprofiler_raw/` (complete)
3. Python fallback TIFF organization — in-process per file (acceptable fallback)
4. QC overlays — in-process per image (no external relaunch)

Phase 14 will insert batch Fiji export between steps 2 and 4:

1. Run CellProfiler once (unchanged)
2. Collect CP outputs (unchanged)
3. **Run Fiji once** with batch macro → final TIFFs in `masks/`, `labels/`
4. If Fiji fails or is not installed → Python in-process fallback + logged warning
5. Generate QC overlays from final TIFFs
6. Write `logs/workflow_summary.json` with **stage timings**

Planned timing fields in `workflow_summary.json`:

```json
{
  "timing": {
    "cellprofiler_seconds": 142.3,
    "fiji_export_seconds": 18.7,
    "qc_seconds": 4.1,
    "total_seconds": 165.1
  }
}
```

Planned config flags on `CellProfilerWorkflowConfig`:

- `fiji_executable: str | None`
- `fiji_export_enabled: bool = True`
- `fiji_macro_path: Path | None` — batch folder macro (default)
- `fiji_fallback_to_python: bool = True`
- `fiji_per_image_fallback: bool = False` — opt-in only

## Export path comparison

| Aspect | Fiji batch (Phase 14, production) | Python TIFF (Phase 12, fallback) |
|--------|-------------------------------------|-------------------------------------|
| Invocations | **One subprocess per folder** | In-process loop (no JVM) |
| Macro | Folder batch macro | N/A — `tifffile` writes |
| Metadata | Native Fiji/ImageJ export tags | ImageJ-compatible tags from Python |
| When used | Fiji installed, export enabled | Fiji missing, tests, quick prototypes |
| Speed | Best for large batches | Fast for small sets / CI |

## What stays in Python (fallback only)

These remain useful but are **not** the production final-export path:

| Function | Module | Purpose |
|----------|--------|---------|
| `save_fiji_compatible_tiff` | `fiji_tiff.py` | ImageJ-tag TIFF writes without Fiji runtime |
| `export_mask_tiff` / `export_label_tiff` / `export_intensity_tiff` | `export.py` | Typed export helpers |
| `organize_cellprofiler_tiffs_for_fiji` | `export.py` | In-process re-export of CP TIFFs (current Phase 13 behavior) |
| `export_cellprofiler_tiff_for_fiji` | `export.py` | Single-file Python re-export |
| `io.save_tiff(..., imagej_compatible=True)` | `io.py` | Optional ImageJ mode on generic save |

## Logs and failure handling

Planned log files under `logs/`:

- `fiji_stdout.log`
- `fiji_stderr.log`
- `fiji_command.txt`
- `fiji_exit_code.txt`

When Fiji is unavailable:

- Workflow continues if `fiji_fallback_to_python=True`
- `workflow_summary.json` records `export_engine: "python_fallback"`
- User sees a clear message pointing to Fiji installation docs

When batch macro fails but per-image fallback is enabled:

- Log `export_mode: "per_image_fallback"` and increased `fiji_export_seconds`
- Document why batch export was not used

## Tests (planned)

- Mock subprocess verifies **single Fiji invocation** per workflow run
- Mock batch macro success path
- Fiji failure → Python fallback
- Missing executable detection
- Timing fields present in workflow summary

```bash
pytest tests/test_fiji_runner.py -v
```

## Related docs

- [cellprofiler_workflow.md](cellprofiler_workflow.md) — Phase 13 batch CP orchestration
- [fiji_tiff_export.md](fiji_tiff_export.md) — Python fallback TIFF format details
- [gui_direction.md](gui_direction.md) — Phase 15 GUI front-end for CP & Fiji
- [fiji_qc_workflow.md](fiji_qc_workflow.md) — manual QC checklist in Fiji GUI
