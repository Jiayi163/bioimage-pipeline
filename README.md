# bioimage-pipeline

A lightweight bioimage analysis framework inspired by Fiji and CellProfiler.

This project is being built phase by phase. TIFF input/output and basic
preprocessing are implemented; the remaining modules are placeholders for
future phases.

## Planned Features

- TIFF image input/output
- Gaussian blur, median filtering, and intensity normalization
- Modular image processing pipelines
- Batch processing of folders
- Otsu thresholding
- Adaptive thresholding
- Object segmentation
- Object measurements
- CSV export

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
examples/
    run_basic_pipeline.py
tests/
    test_io.py
    test_preprocess.py
    test_threshold.py
    test_segment.py
    test_measure.py
    test_pipeline.py
    test_batch.py
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
