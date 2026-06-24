# Lab Ground-Truth Annotations Guide

This guide describes what to provide when validating threshold parameter
candidates against expert segmentation. Ground-truth masks let the Threshold
Parameter Assistant rank candidates by agreement with lab-approved objects, not
only by heuristic screening.

## How many images?

| Goal | Recommended count |
|------|-------------------|
| First feasibility check | 5–10 images |
| Stronger parameter selection | 15–30 images |
| ML training (future, only if needed) | 20–50+ depending on method |

Start with 5–10 representative images. Expand once the workflow is working.

## Preferred format: binary mask TIFF

For each annotated image, provide one binary mask TIFF:

- **White (non-zero)** = true object (EV spot / particle)
- **Black (zero)** = background
- Same pixel width and height as the analysis input image
- Single 2D plane (not a multi-page stack unless each page matches one input)

## Naming convention

Place masks in one folder and name each file:

```text
<input_stem>_reference_mask.tif
```

Examples:

| Input image | Reference mask |
|-------------|----------------|
| `BSA_spots_01.tif` | `BSA_spots_01_reference_mask.tif` |
| `sample.tif` | `sample_reference_mask.tif` |

The assistant and CLI resolve masks by matching the input filename stem.

## Image diversity checklist

Do **not** annotate only the cleanest images. Include cases such as:

- [ ] Clean image with low background
- [ ] High background or uneven illumination
- [ ] Weak-signal image
- [ ] Many-object image
- [ ] Few-object image
- [ ] At least one image where the current threshold clearly fails
- [ ] Different treatment/control conditions (if applicable)

## Optional later: ROI outlines

ROI sets or object outlines (e.g. from ImageJ) can be converted to binary masks
in a later pass. For the first validation round, binary masks are preferred.

## How annotations are used

1. You run a subset trial in the Threshold Parameter Assistant (or CLI).
2. You point the tool at your reference mask folder.
3. Each candidate CellProfiler setting is scored against your masks (object-level
   precision/recall/F1, pixel Dice/IoU, count error).
4. You review GT rankings **and** heuristic screening, previews, and per-image QC.
5. You explicitly confirm before applying a setting to the full dataset.

Heuristic screening still runs when reference masks are absent. Ground-truth
scoring is an optional upgrade when annotations exist.

## Pipeline requirement

The imported CellProfiler pipeline must export segmentation masks (e.g. via
SaveImages) so predicted masks can be compared to your reference masks. Without
mask export, ground-truth scoring cannot run.

## Consistent preprocessing

Apply the same preprocessing to all images in an experiment (e.g. Z-max
projection, background subtraction). Avoid per-image manual brightness tweaks
inside the assistant workflow so comparisons remain fair.
