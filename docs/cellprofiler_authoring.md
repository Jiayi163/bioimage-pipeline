# CellProfiler Authoring (Hybrid Workflow)

Use native CellProfiler when the custom editor's catalog stubs are not enough for
complex module settings (Filters, Joiners, measurement pickers, etc.).

## Workflow

1. Build the setup block in the custom editor: **Images → Metadata → NamesAndTypes → Groups**.
2. Add analysis modules from the catalog, or start from a saved `.cppipe`.
3. **Tools → Open in CellProfiler...** (materializes the current pipeline to a working file and launches CellProfiler).
4. Edit settings in CellProfiler; **File → Save** in CellProfiler.
5. **File → Open Pipeline...** in the custom editor to reload the updated `.cppipe`.
6. Set the **Images → Input folder path** in the custom editor and **Run Pipeline**.

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

## When to fork CellProfiler

Fork only if you need **in-app** Bio-Formats preview and full ModuleView fidelity
without switching applications. See [architecture_decision.md](architecture_decision.md).
