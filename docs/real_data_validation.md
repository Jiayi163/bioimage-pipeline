# Real Data Validation (Phase 10.4)

This document records validation on microscopy TIFF images and the
comparison helpers used to evaluate Python and CellProfiler outputs.

## Validation helpers

`bioimage_pipeline/validation.py` provides:

- `inspect_image` — summarize image properties and flag limitations
- `compare_masks` — IoU, Dice, and object-count agreement
- `compare_measurements` — compare object counts and mean areas
- `build_validation_report` — structured JSON report for one image

## Manual validation on your images

```bash
python examples/validate_real_data.py ^
  --input-dir path\to\real_images ^
  --output-dir path\to\validation_output
```

Optional reference masks:

```bash
python examples/validate_real_data.py ^
  --input-dir path\to\real_images ^
  --output-dir path\to\validation_output ^
  --reference-mask-dir path\to\reference_masks
```

Compare Python and CellProfiler on the same folder:

```bash
python examples/validate_real_data.py ^
  --input-dir path\to\real_images ^
  --output-dir path\to\validation_output ^
  --engine both ^
  --cppipe path\to\pipeline.cppipe ^
  --executable "C:\Program Files\CellProfiler\CellProfiler.exe"
```

## Observed limitations

| Condition | Effect |
|-----------|--------|
| Low SNR | Thresholding becomes unstable; object counts may drift |
| Multi-channel images | Lightweight Python mode analyzes one 2D plane only |
| Large images (>4096 px) | Slower processing and higher memory use |
| Uneven illumination | Otsu threshold may under-segment dim regions |
| Touching objects | Connected components merge without watershed splitting (Phase 11) |

## Failure cases to watch

- **Empty segmentation:** image dynamic range too low or blur/threshold mismatch.
- **Over-segmentation:** noise speckles counted as objects when `min_object_size` is too small.
- **Engine mismatch:** Python and CellProfiler pipelines use different preprocessing, so mask IoU is expected to be below 1.0 even on the same image.
- **Missing CellProfiler CSV columns:** `merge_cellprofiler_tables` requires `Image_Number` and `ObjectNumber` on object tables.

## Acceptance checklist

- [x] Validation helpers implemented (`validation.py`)
- [x] Manual validation script (`examples/validate_real_data.py`)
- [x] Limitations and failure cases documented
- [ ] E2E run on your own lab images with Python and/or CellProfiler

Re-run `examples/validate_real_data.py` whenever you change preprocessing,
thresholding, or CellProfiler pipelines.

## Threshold recommender E2E (Phase 17.6)

Subset trial smoke test with a real CellProfiler install:

```bash
python examples/validate_threshold_recommender_e2e.py ^
  --cppipe path\to\your\assay.cppipe ^
  --input-dir path\to\real_images ^
  --output-dir path\to\e2e_output ^
  --apply
```

With synthetic spot TIFFs (pipeline generation smoke test only):

```bash
python examples/validate_threshold_recommender_e2e.py ^
  --output-dir path\to\e2e_output
```

The script stages a subset, runs the threshold recommender trial, checks ranking
artifacts, and optionally applies the top variant to the full input folder.

**Recommended:** use your real assay `.cppipe` and lab images. Synthetic
fixtures help test pipeline generation, but some CellProfiler installs may still
report `Empty image set list` until real assay data is used.

GUI equivalent: open `python examples/run_gui.py`, set paths, click **Threshold
Recommender**, run subset trial, review ranking, then apply to the full dataset.
