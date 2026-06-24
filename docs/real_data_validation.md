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

## Threshold Parameter Assistant E2E (Phase 17.6)

The threshold workflow is a **human-in-the-loop assistant**, not an automatic
optimizer. Heuristic scores and QC warnings help reject obviously bad candidates
and sort plausible ones for review. They do **not** prove biological optimality
without ground truth.

### When heuristics are enough

Use heuristic screening when:

- You have no manually approved reference masks yet
- You need to reject failed runs, extreme tiny/huge fractions, or object-count
  ratios far from baseline
- You will review QC overlay previews and per-image metrics before applying

The assistant can say: *this parameter looks plausible based on heuristic QC*.

### When ground truth is needed

Upgrade to reference-mask scoring when:

- You have 5–10 representative images with lab-approved masks or outlines
- You need to claim which parameter best matches expert judgment
- Heuristic size fractions look good but object counts are biologically suspicious

See [lab ground-truth annotation guide](lab_ground_truth_annotations.md) for mask
format and naming (`<image_stem>_reference_mask.tif`).

### Ground-truth scoring workflow

When reference masks are available, run the assistant or CLI with
`--reference-mask-dir`:

```bash
python examples/run_threshold_variants.py trial ^
  --cppipe path\to\assay.cppipe ^
  --input-dir path\to\images ^
  --output-dir path\to\output ^
  --reference-mask-dir path\to\reference_masks
```

The tool writes `threshold_recommender/ground_truth/` with:

- `manifest.json` — paired subset images and reference masks
- `threshold_variant_gt_comparison.csv/json` — per variant × image metrics
- `threshold_variant_gt_comparison_ranking.csv/json` — GT-ranked candidates

Metrics include object-level precision/recall/F1, pixel Dice/IoU, and object
count error. Heuristic screening still runs in parallel; use GT rank as the
primary review starting point when masks exist.

Until reference masks are provided, treat heuristic rank `#1` as a **starting
point for review**, not the final answer.

### Consistent preprocessing

Apply the same preprocessing to all comparable images in an experiment (e.g.
Z-max projection, illumination correction, background subtraction). Avoid
per-image manual brightness tweaks inside the assistant workflow.

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

The script stages a subset, writes a subset characterization report, runs
candidate variants, saves aggregate and per-image comparison tables, and
optionally applies the user-selected variant to the full input folder.

**Recommended:** use your real assay `.cppipe` and lab images. Synthetic
fixtures help test pipeline generation, but some CellProfiler installs may still
report `Empty image set list` until real assay data is used.

Ensure the pipeline exports mask or label images (e.g. SaveImages) so QC overlay
previews can be generated.

GUI equivalent: open `python examples/run_gui.py`, set paths, click **Threshold
Parameter Assistant**, run subset trial, review screening table, per-image QC,
and compare previews, then apply to the full dataset.
