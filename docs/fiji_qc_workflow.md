# Fiji QC Workflow (Phase 10.5)

Use this workflow to inspect segmentation results in Fiji/ImageJ without writing
custom scripts. For export format details, see
[docs/fiji_tiff_export.md](fiji_tiff_export.md).

## 1. Run analysis and generate QC overlays

```bash
python examples/visual_check.py
```

This writes:

| File | Purpose |
|------|---------|
| `original.tif` | Input image |
| `mask.tif` | Binary mask (0/255) |
| `labels.tif` | Integer label image |
| `measurements.csv` | Object measurements |
| `qc_mask_overlay.png` | Red mask overlay for quick review |
| `qc_label_overlay.png` | Color-coded label overlay |

For batch outputs, generate overlays after `run_pipeline_on_folder`:

```python
from bioimage_pipeline.qc import generate_qc_for_folder

generate_qc_for_folder("path/to/images", "path/to/output")
```

## 2. Open images in Fiji

1. Launch Fiji.
2. **File → Open** and select `original.tif`.
3. **File → Open** and select `mask.tif` or `labels.tif`.

## 3. Inspect the binary mask

1. Select the `mask.tif` window.
2. **Image → Duplicate** so you keep an untouched copy.
3. **Image → Color → Merge Channels**:
   - C1 (red): `original.tif`
   - C2 (green): duplicated mask
   - Keep **Create composite** checked.
4. Use the brightness/contrast tool on the mask channel if needed.

The generated `qc_mask_overlay.png` provides the same view without manual merging.

## 4. Inspect labeled objects

1. Open `labels.tif`.
2. **Image → Lookup Tables → Glasbey on dark** (or another LUT).
3. **Image → Overlay → Add image...** and choose `original.tif` with low opacity.

The generated `qc_label_overlay.png` shows colored labels on the grayscale image.

## 5. Review measurements

1. Open `measurements.csv` in Excel or Fiji's table tools.
2. Sort by `area` or `mean_intensity` to find outliers.
3. Cross-check suspicious rows against the overlay images.

## 6. Optional interactive QC with Napari

```bash
pip install bioimage-pipeline[qc]
```

```python
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.qc import view_in_napari

image = read_tiff("output/visual_check/original.tif")
mask = read_tiff("output/visual_check/mask.tif") > 0
labels = read_tiff("output/visual_check/labels.tif")
view_in_napari(image, mask=mask, labels=labels)
```

## Quick checklist

- [ ] Foreground objects match biological structures
- [ ] Background is mostly excluded from the mask
- [ ] Touching objects are not over-merged (see Phase 11 for splitting)
- [ ] Measurement table row count matches visible objects
- [ ] Low-intensity or dim regions are not systematically missed
