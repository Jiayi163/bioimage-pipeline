"""Launch the Phase 15.1 workflow shell."""

from __future__ import annotations

from bioimage_pipeline.gui import launch_workflow_shell


def main() -> int:
    launch_workflow_shell()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
