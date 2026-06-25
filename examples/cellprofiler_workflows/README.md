# CellProfiler workflow templates

Reference pipelines for use with **native CellProfiler** (hybrid authoring).

## OIR Z-max + projection (MakeProjection)

**Goal:** Load Olympus `.oir` stacks, compute Z-max projection, then analyze 2D images.

**Modules (in order):**

| # | Module | Notes |
|---|--------|--------|
| 1 | Images | Drag-drop `.oir` files or folder |
| 2 | Metadata | Optional: extract from folder/file name |
| 3 | NamesAndTypes | Assign image names; Process as 3D = No |
| 4 | Groups | Optional: group by plate/well metadata |
| 5 | MakeProjection | Type = Maximum; output image e.g. `ZMax` |
| 6+ | Analysis | IdentifyPrimaryObjects, Measure*, etc. |
| n | ExportToSpreadsheet | CSV measurements |
| n+1 | SaveImages | Optional mask/label export |

**Alternative:** Pre-project with `examples/run_oir_zmax_batch.py`, then use a simpler
pipeline starting at projected `.tif` files in **Images**.

## Post-CP Fiji export (custom app)

The bioimage-pipeline app runs CellProfiler headlessly, then batch-exports masks/labels
via Fiji macros. That step is **not** a standard CellProfiler module — use the custom
editor **Run workflow** panel after saving your `.cppipe`.

See [cellprofiler_authoring.md](../docs/cellprofiler_authoring.md).

## Weka ML assay template (Phase 18 — planned)

**Goal:** CellProfiler pipeline for `weka_ml` segmentation mode — original
fluorescence + single-channel foreground probability maps (`*_prob.tif`,
normalized to 0–1 by Python staging before CP runs).

**Planned file:** `weka_assay_template.cppipe` (authored once in CellProfiler
desktop, tested, then committed).

**Expected modules:**

| # | Module | Notes |
|---|--------|--------|
| 1 | Images | Reads staged `cellprofiler_input/` |
| 2 | Metadata | Optional |
| 3 | NamesAndTypes | `EV` = originals; `Prob` = `*_prob.tif` |
| 4 | Groups | Optional |
| 5 | IdentifyPrimaryObjects | Input `Prob`; manual threshold on 0–1 scale |
| 6+ | Measure*, SaveImages, ExportToSpreadsheet | Same export goals as standard workflow |

Weka classifier **training** happens in Fiji Trainable Weka Segmentation, not in
this repository. See [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) Phase 18.
