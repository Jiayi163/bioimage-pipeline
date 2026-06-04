# CellProfiler Integration Validation (Phase 10.1)

This document records Phase 10.1 validation for the Python wrapper around
CellProfiler headless execution.

## Environment

| Item | Value |
|------|--------|
| OS | Windows 10 |
| CellProfiler | 4.2.8 |
| Executable | `C:\Program Files\CellProfiler\CellProfiler.exe` |
| Python package | `bioimage_pipeline.cellprofiler_runner` |

## Automated checks (always run)

```bash
pytest tests/test_cellprofiler_runner.py
```

These tests cover executable validation, CLI command construction, subprocess
error handling, CSV reading, and output-directory loading (fixtures).

## End-to-end validation (real `.cppipe` required)

Use any local folders for input TIFFs and CellProfiler output (they are not
stored in this repository). Then run:

```bash
python examples/validate_cellprofiler.py ^
  --cppipe path\to\your\pipeline.cppipe ^
  --input-dir path\to\images ^
  --output-dir path\to\cellprofiler_output ^
  --executable "C:\Program Files\CellProfiler\CellProfiler.exe"
```

### Manual validation checklist

- [x] CellProfiler executable detection (`_validate_cellprofiler_executable`)
- [x] `.cppipe` path validation before launch
- [x] Input directory validation
- [x] Output directory creation (`mkdir(parents=True)`)
- [x] Headless CLI flags (`-c -r -p -i -o`)
- [x] `read_cellprofiler_csv` and `load_cellprofiler_measurements`
- [x] Wrapper unit tests pass (`pytest tests/test_cellprofiler_runner.py`)
- [ ] Full E2E run with your `.cppipe` (run `examples/validate_cellprofiler.py`)

Re-run the script above when the pipeline file is available to refresh this
checklist.

### Expected outputs after E2E run

- `MyExpt_Image.csv` (image-level measurements)
- `MyExpt_IdentifyPrimaryObjects.csv` (object-level)
- Additional tables depending on the pipeline (Experiment, secondary objects)

## Phase 10.1 status

Wrapper behavior is implemented, tested, and marked `PHASE COMPLETE` in
`DEVELOPMENT_PLAN.md`. Re-run `validate_cellprofiler.py` with your `.cppipe`
whenever you change pipelines or CellProfiler versions.
