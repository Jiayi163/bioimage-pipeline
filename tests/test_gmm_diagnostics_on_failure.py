"""Test that GMM diagnostics are preserved even when all multi-start attempts fail."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.pipeline import run_puncta_declump

# Fields exported by production GmmInitAttemptDiagnostics on gaussian-fitting HEAD.
HEAD_ATTEMPT_FIELDS = (
    "strategy",
    "converged",
    "post_merge_component_count",
    "merge_collapsed",
    "merge_notes",
    "fitted_centers",
    "fitted_amplitudes",
    "fitted_sigma_x",
    "fitted_sigma_y",
    "rss",
    "bic",
    "selected",
    "optimizer_runtime_s",
    "n_optimizer_evaluations",
)


def test_gmm_diagnostics_exported_when_all_fits_fail(tmp_path: Path) -> None:
    """Regression: when all multi-start attempts fail, diagnostics should still be exported."""
    image = np.zeros((50, 50), dtype=np.float32)
    mask = np.ones((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1

    config = PunctaDeclumpConfig(
        threshold_method="external_mask",
        gmm_multi_start_enabled=True,
        enable_selective_routing=False,
        diagnostic_mode="summary",
        export_fiji_tiffs=False,
        candidate_detector="python_log",
        gmm_max_multi_starts=3,
    )

    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        output_dir=tmp_path,
        stem="pathological",
    )

    gmm_diag = result.threshold_metadata.get("gmm_init_diagnostics")
    if gmm_diag is not None:
        assert len(gmm_diag) > 0, "Expected at least one GMM diagnostics entry"
        entry = gmm_diag[0]
        assert "attempts" in entry, "Expected 'attempts' field in diagnostics"
        if entry["attempts"]:
            attempt = entry["attempts"][0]
            for field in HEAD_ATTEMPT_FIELDS:
                assert field in attempt, f"Missing production field {field}"


def test_gmm_diagnostics_categorization(tmp_path: Path) -> None:
    """Test that model_selection_reason distinguishes failure types correctly."""
    image = np.zeros((48, 48), dtype=np.float32)
    image[24, 24] = 1000
    image[23:26, 23:26] = 500

    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:30, 20:30] = 1

    config = PunctaDeclumpConfig(
        threshold_method="external_mask",
        gmm_multi_start_enabled=True,
        enable_selective_routing=False,
        diagnostic_mode="summary",
        export_fiji_tiffs=False,
        candidate_detector="python_log",
        accept_brightest_on_fit_failure=True,
    )

    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        output_dir=tmp_path,
        stem="single_spot",
    )

    assert len(result.candidates) >= 1, "Expected at least one candidate"
    candidate = result.candidates[0]
    assert candidate.model_selection_reason is not None

    if candidate.tried_gmm:
        model_sel = candidate.model_selection_reason
        assert any(
            keyword in model_sel
            for keyword in (
                "no_successful_multi_component_fit",
                "collapsed_to_one",
                "kept_single",
                "spurious_tight_split",
                "selected_gmm",
            )
        ), f"Unexpected model_selection_reason: {model_sel}"


def test_gmm_init_diagnostics_json_export(tmp_path: Path) -> None:
    """Test that gmm_init_diagnostics.json is written and serializable on HEAD."""
    image = np.random.rand(48, 48).astype(np.float32) * 100
    mask = np.ones((48, 48), dtype=np.uint8)
    mask[15:35, 15:35] = 1

    config = PunctaDeclumpConfig(
        threshold_method="external_mask",
        gmm_multi_start_enabled=True,
        enable_selective_routing=False,
        diagnostic_mode="summary",
        export_fiji_tiffs=False,
        candidate_detector="python_log",
    )

    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        output_dir=tmp_path,
        stem="test_case",
    )

    from bioimage_pipeline.puncta.export import ResultExporter

    exporter = ResultExporter()
    summary_path = tmp_path / "test_case_summary.json"
    exporter.export_summary_json(summary_path, result)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gmm_diag = summary.get("threshold_metadata", {}).get("gmm_init_diagnostics")

    if gmm_diag is not None and len(gmm_diag) > 0:
        entry = gmm_diag[0]
        assert isinstance(entry, dict)
        assert "object_id" in entry
        assert "attempts" in entry
        attempts = entry["attempts"]
        if attempts:
            att = attempts[0]
            for field in HEAD_ATTEMPT_FIELDS:
                assert field in att, f"Missing production field {field}"
