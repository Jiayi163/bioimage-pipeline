# Architecture Decision: Custom App vs CellProfiler Fork

**Date:** 2026-06-10  
**Status:** Accepted  
**Decision:** **Hybrid strategy (Option 1 + native CellProfiler authoring)**

## Context

The bioimage-pipeline project needs a CellProfiler-style pipeline editor while preserving
lab-specific Fiji/stack/OIR workflows. Two options were evaluated:

| Option | Summary |
|--------|---------|
| **Option 1** | Continue the custom Tkinter application (~7k LOC) |
| **Option 2** | Fork CellProfiler (~110k LOC) and embed Fiji/stack features |

See the feasibility analysis for full comparison.

## Decision

Adopt a **hybrid strategy**:

1. **Continue Option 1** as the primary product shell for pipeline editing (lightweight)
   and integrated **CP → Fiji → QC execution**.
2. **Do not fork CellProfiler** for the editor unless full module-setting fidelity
   and Bio-Formats preview become hard blockers.
3. **Use native CellProfiler** for advanced pipeline authoring when catalog stubs are
   insufficient (`.cppipe` round-trip via **Pipeline → Open in CellProfiler**).
4. **Defer full CP fork**; document CP-native workflows (OIR Z-max via MakeProjection)
   instead of porting the entire stack into a fork.

## Consequences

### In scope (custom app)

- Phases C–F: file menu, Images-centric input, output path surfacing, editor-first layout
- Headless CellProfiler + Fiji orchestration (existing)
- OIR Z-max and stack batch tools (existing Python/Fiji paths)

### Out of scope (for now)

- Full CellProfiler ModuleView parity (92 modules × all setting types)
- Maintaining a CellProfiler fork
- Bio-Formats preview inside the Tkinter shell

### Hybrid authoring workflow

1. Build or edit pipeline in the custom editor (setup modules + catalog modules).
2. For complex module settings, use **Open in CellProfiler**, save `.cppipe`, reload.
3. Run from the custom app (materializes working pipeline, no separate file pick required).

## References

- [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) — Phase 15.3
- [gui_direction.md](gui_direction.md)
- CellProfiler source: `C:\Users\Administrator\Desktop\CellProfiler`
