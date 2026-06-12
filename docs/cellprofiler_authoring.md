# CellProfiler Authoring (Import-Only Workflow)

Author pipelines in **native CellProfiler**. This application imports the saved
``.cppipe`` file and runs it headlessly with batch orchestration (OIR projection,
Fiji export, QC, organized results).

## Workflow

1. Create or edit a pipeline in CellProfiler (all module settings, file lists, exports).
2. **File → Save** in CellProfiler.
3. In this app: **File → Open Pipeline...** and select the ``.cppipe`` file.
4. Set **Default Input Folder** and **Default Output Folder** in the workflow panel.
5. **Analyze Images** — the imported file is passed to CellProfiler as-is (no rewrites).

Use **Tools → Open in CellProfiler...** to jump back to CellProfiler for edits, then
re-import the saved pipeline.

## Required pipeline outputs (your responsibility)

The app does not auto-insert modules. Include whatever you need:

| Goal | Typical CellProfiler module |
|------|----------------------------|
| Measurement CSVs | **ExportToSpreadsheet** |
| Mask/label TIFFs for Fiji/QC | **SaveImages** (mask or objects) |

Before each run, the GUI shows **advisories** (not auto-fixes) when export modules
are missing or when embedded input paths do not exist on this machine.

Input images are passed via CellProfiler's ``-i`` CLI flag from **Default Input
Folder**, not by rewriting the pipeline file.

## OIR Z-max via native CellProfiler (MakeProjection)

CellProfiler can import Olympus `.oir` files through Bio-Formats and project Z-stacks
with **MakeProjection** instead of a custom fork module.

Suggested pipeline (after input modules):

1. **Images** — add `.oir` files or parent folder (in CP file list when authoring in CellProfiler).
2. **Metadata** — optional well/plate extraction from folder names.
3. **NamesAndTypes** — assign channel names; keep **Process as 3D?** = No for 2D projection workflow.
4. **Groups** — group by metadata if batching multiple files.
5. **MakeProjection** — type: **Maximum** (Z-max); assign output image name.
6. Downstream analysis modules (e.g. **IdentifyPrimaryObjects**).
7. **ExportToSpreadsheet** / **SaveImages**.

For batch OIR→TIFF preprocessing outside CellProfiler, continue using:

```bash
python examples/run_oir_zmax_batch.py --input-dir ... --output-dir ...
```

Then point the **Images** module input folder at the projected TIFF folder.

When using the GUI/headless workflow (not the standalone batch CLI), projected
TIFFs are cached under `{output}/oir_projection/` on re-runs — see
[cellprofiler_workflow.md — OIR projection cache](cellprofiler_workflow.md#oir-projection-cache-output-folder).

## When to fork CellProfiler

Fork only if you need **in-app** Bio-Formats preview and full ModuleView fidelity
without switching applications. See [architecture_decision.md](architecture_decision.md).
