# Architecture Decision: Custom App vs CellProfiler Fork

**Date:** 2026-06-10 (updated 2026-06-11)  
**Status:** Accepted  
**Decision:** **Import-only orchestration shell + native CellProfiler authoring**

## Context

The bioimage-pipeline project needs a CellProfiler-style pipeline editor while preserving
lab-specific Fiji/stack/OIR workflows. Two options were evaluated:

| Option | Summary |
|--------|---------|
| **Option 1** | Continue the custom Tkinter application (~7k LOC) |
| **Option 2** | Fork CellProfiler (~110k LOC) and embed Fiji/stack features |

See the feasibility analysis for full comparison.

## Decision

Adopt an **import-only orchestration shell**:

1. **Author pipelines in native CellProfiler** — the GUI imports a ``.cppipe`` path and
   runs it headlessly without rewriting module settings.
2. **Continue this app** as the batch orchestration shell: **CP → Fiji → QC**, OIR
   projection, logs, and organized results folders.
3. **Do not fork CellProfiler** for an in-app module editor.
4. **Deprecate the Phase 15.2 catalog builder** for production use; keep parse/validate
   utilities and legacy tests only.

## Consequences

### In scope (custom app)

- Import-only GUI: pipeline path, read-only module list, input/output folders, run/logs/QC
- Headless CellProfiler + Fiji orchestration (existing)
- OIR Z-max and stack batch tools (existing Python/Fiji paths)

### Out of scope (for now)

- In-app CellProfiler module editor / catalog builder (deprecated)
- Full CellProfiler ModuleView parity (92 modules × all setting types)
- Maintaining a CellProfiler fork
- Bio-Formats preview inside the Tkinter shell
- Automatic pipeline rewrites (SaveImages normalization, auto ExportToSpreadsheet)

### Import-only workflow

1. Author and save pipeline in CellProfiler.
2. **File → Open Pipeline...** in this app; optional **Open in CellProfiler** for edits.
3. Set input/output folders; **Analyze Images** runs the imported file via ``-p`` and ``-i``.

## References

- [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) — Phase 15.3
- [gui_direction.md](gui_direction.md)
- CellProfiler source: `C:\Users\Administrator\Desktop\CellProfiler`
