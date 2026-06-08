# Stack / Batch Workflow

Fiji-style stack and batch processing on the **Python pipeline engine**.
Load a folder of TIFFs or one multi-page TIFF, run the same
preprocess → segment → measure steps on every frame, and export TIFFs + CSV.

## Fiji analogy

| Fiji concept | This project |
|--------------|--------------|
| Open folder as virtual stack | `load_stack_from_folder()` / `load_stack()` |
| Open multi-page TIFF (Z-stack) | `load_stack_from_tiff()` / `iter_stack_frames()` |
| Process Stack / macro on all slices | `run_pipeline_on_stack()` |
| Save results per slice | `{stem}_f{idx:03d}_mask.tif`, `_labels.tif`, `_measurements.csv` |
| Results table | `all_measurements.csv` with `stack_id`, `frame_index`, `z_index` |
| Visual QC | `generate_qc_for_stack()` → PNG overlays |
| Repeatable macro | JSON recipe + `examples/run_stack_batch.py` |

## Quick start (no real data)

```bash
python examples/run_stack_batch.py --demo --output output/demo_test
python examples/run_stack_example.py
```

## Real data

**Folder of 2D TIFFs** (one frame per file, sorted alphabetically):

```bash
python examples/run_stack_batch.py \
    --input C:/path/to/images \
    --output C:/path/to/results \
    --export-processed \
    --generate-qc
```

**Single multi-page TIFF** (Z-stack or time series):

```bash
python examples/run_stack_batch.py \
    --input C:/path/to/stack.tif \
    --output C:/path/to/results \
    --labeling watershed
```

## JSON batch recipe

Save pipeline settings in a JSON file for reproducible runs (like a Fiji macro):

```json
{
  "blur_sigma": 1.0,
  "min_object_size": 20,
  "labeling_method": "connected",
  "export_processed": true,
  "generate_qc": true,
  "input": "path/to/images",
  "output": "path/to/results"
}
```

Run with a recipe (CLI flags override recipe values):

```bash
python examples/run_stack_batch.py --recipe examples/stack_batch_recipe.json
python examples/run_stack_batch.py --recipe my_recipe.json --labeling watershed
```

Demo recipe (no input path needed):

```bash
python examples/run_stack_batch.py --recipe examples/stack_batch_recipe.json
```

See `examples/stack_batch_recipe.json` for a working template.

## Python API

```python
from bioimage_pipeline.stack import load_stack
from bioimage_pipeline.stack_batch import run_stack_batch_workflow

# Folder or multi-page TIFF — auto-detected
stack = load_stack("path/to/images_or_stack.tif")

result = run_stack_batch_workflow(
    "path/to/images_or_stack.tif",
    "path/to/output",
    blur_sigma=1.0,
    min_object_size=20,
    labeling_method="connected",  # or "watershed"
    export_processed=True,
    generate_qc=True,
)

print(result.processed)
print(result.measurements)
print(result.qc_artifacts)
```

Lower-level control:

```python
from bioimage_pipeline.analysis import build_default_pipeline
from bioimage_pipeline.batch import run_pipeline_on_stack
from bioimage_pipeline.stack import load_stack_from_tiff

stack = load_stack_from_tiff("stack.tif")
pipeline = build_default_pipeline(labeling_method="watershed")
batch_result = run_pipeline_on_stack(pipeline, stack, "output/", export_processed=True)
```

## Output layout

```text
output/
    img_00_f000_mask.tif
    img_00_f000_labels.tif
    img_00_f000_processed.tif      # when --export-processed
    img_00_f000_measurements.csv
    img_00_f000_qc_mask_overlay.png   # when --generate-qc
    img_00_f000_qc_label_overlay.png
    ...
    all_measurements.csv
```

### Combined CSV columns

| Column | Description |
|--------|-------------|
| `stack_id` | Source file or folder name |
| `frame_index` | Sequential frame index (0-based) |
| `z_index` | Z position when known |
| `t_index` | Time index (ImageJ hyperstacks only) |
| `c_index` | Channel index (ImageJ hyperstacks only) |
| `filename` | Original source filename |
| `label`, `area`, … | Standard measurement columns |

## Pipeline steps (default)

1. Gaussian blur (`blur_sigma`)
2. Otsu threshold
3. Remove small objects (`min_object_size`)
4. Label objects (`connected` or `watershed`)
5. Measure region properties (`area`, `centroid`, intensities)

## Limitations

- **2D per frame** — each slice is analyzed as a single 2D plane.
- **Folder mode** — one frame per file; multi-page files in a folder use the first plane only.
- **Multi-channel** — basic axis metadata is read; full multi-channel analysis is not implemented.
- **Output** — per-frame TIFFs, not a single combined hyperstack TIFF.

## Tests

```bash
pytest tests/test_stack.py tests/test_stack_recipe.py tests/test_stack_batch_cli.py -v
pytest tests/test_batch.py tests/test_io.py -v
python -m pytest -v
```

## Related modules

| Module | Role |
|--------|------|
| `bioimage_pipeline/io.py` | `iter_stack_frames`, `StackFrame`, `AxisInfo` |
| `bioimage_pipeline/stack.py` | `ImageStack`, `load_stack()` |
| `bioimage_pipeline/batch.py` | `run_pipeline_on_stack()` |
| `bioimage_pipeline/stack_batch.py` | `run_stack_batch_workflow()` |
| `bioimage_pipeline/stack_recipe.py` | JSON recipe load/save |
| `bioimage_pipeline/qc.py` | `generate_qc_for_stack()` |
