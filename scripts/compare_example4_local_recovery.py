#!/usr/bin/env python3
"""Compare example4 local-peak-recovery run against the phase_b_d baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio
from skimage.draw import disk

from bioimage_pipeline.qc import normalize_for_display

BASE = Path(r"C:\Users\Administrator\Desktop\example4")
BASELINE_DIR = BASE / "phase_b_d"
NEW_DIR = BASE / "phase_b_d_recovery_fast"
IMAGE_PATH = BASE / "input" / "MAX_10% BSA block with EV_0007.tif"
OUT_DIR = NEW_DIR / "diagnostics" / "investigation" / "dense_cluster"

# Dense ROI from the prior investigation.
ROI = (258, 198, 552, 490)

# Original 36 zero-maxima fallback objects in the dense ROI.
ZERO_MAXIMA_IDS = [
    429, 467, 475, 483, 516, 524, 543, 546, 556, 574, 585, 619, 629, 634, 644,
    680, 693, 734, 740, 744, 774, 784, 827, 831, 835, 856, 870, 874, 904, 908,
    941, 958, 959, 961, 968, 979,
]


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    disp = normalize_for_display(gray)
    return np.stack([disp, disp, disp], axis=-1).astype(np.uint8)


def _draw_cross(rgb: np.ndarray, row: float, col: float, color: tuple[int, int, int], size: int = 3) -> None:
    h, w = rgb.shape[:2]
    r, c = int(round(row)), int(round(col))
    for dr in range(-size, size + 1):
        rr = r + dr
        if 0 <= rr < h and 0 <= c < w:
            rgb[rr, c] = color
        cc = c + dr
        if 0 <= r < h and 0 <= cc < w:
            rgb[r, cc] = color


def _in_roi(row: float, col: float) -> bool:
    r0, c0, r1, c1 = ROI
    return r0 <= row < r1 and c0 <= col < c1


def _load(stem_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meas = pd.read_csv(stem_dir / f"{stem}_measurements.csv")
    diag = pd.read_csv(stem_dir / f"{stem}_object_diagnostics.csv")
    summary = json.loads((stem_dir / f"{stem}_summary.json").read_text(encoding="utf-8"))
    return meas, diag, summary


def _object_status(diag: pd.DataFrame, meas: pd.DataFrame, object_id: int) -> dict[str, object]:
    drow = diag.loc[diag.object_id == object_id]
    mrows = meas.loc[meas.object_id == object_id]
    if drow.empty:
        return {"object_id": object_id, "missing": True}
    row = drow.iloc[0]
    accepted = int((mrows["accepted"].fillna(False) & (mrows["fit_status"] == "fit_ok")).sum()) if not mrows.empty else 0
    yellow = 0
    if not mrows.empty:
        yellow = int((mrows["fit_status"] == "fit_failed_fallback").sum())
    return {
        "object_id": object_id,
        "path": row.get("path"),
        "fit_status": row.get("fit_status"),
        "n_accepted_fit_ok": int(row.get("n_accepted_fit_ok", accepted) or 0),
        "n_filtered_local_maxima": row.get("n_filtered_local_maxima"),
        "tried_gmm": bool(row.get("tried_gmm")),
        "peak_source": row.get("peak_source") if "peak_source" in row.index else None,
        "local_peak_recovery_attempted": row.get("local_peak_recovery_attempted")
        if "local_peak_recovery_attempted" in row.index
        else None,
        "local_peak_recovery_filtered_count": row.get("local_peak_recovery_filtered_count")
        if "local_peak_recovery_filtered_count" in row.index
        else None,
        "yellow_fallback_markers": yellow,
    }


def _yellow_count(meas: pd.DataFrame, object_ids: list[int] | None = None) -> int:
    sub = meas
    if object_ids is not None:
        sub = meas[meas.object_id.isin(object_ids)]
    return int((sub["fit_status"] == "fit_failed_fallback").sum())


def _accepted_centers(meas: pd.DataFrame, object_ids: list[int] | None = None) -> int:
    sub = meas
    if object_ids is not None:
        sub = meas[meas.object_id.isin(object_ids)]
    ok = sub["accepted"].fillna(False) & (sub["fit_status"] == "fit_ok")
    return int(ok.sum())


def _save_overlay(image: np.ndarray, meas: pd.DataFrame, path: Path) -> None:
    r0, c0, r1, c1 = ROI
    roi = _to_rgb(image[r0:r1, c0:c1])
    for _, row in meas.iterrows():
        fr = float(row.get("final_row", row["initial_row"]))
        fc = float(row.get("final_col", row["initial_col"]))
        if not _in_roi(fr, fc):
            continue
        rr, cc = fr - r0, fc - c0
        if bool(row.get("accepted")) and row.get("fit_status") == "fit_ok":
            yy, xx = disk((int(round(rr)), int(round(cc))), 2, shape=roi.shape[:2])
            roi[yy, xx] = (0, 255, 0)
        elif row.get("fit_status") == "fit_failed_fallback":
            _draw_cross(roi, rr, cc, (255, 220, 0), size=3)
        else:
            _draw_cross(roi, rr, cc, (255, 80, 80), size=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(path, roi)


def main() -> None:
    stem = "EV_0007_phaseBD"
    base_meas, base_diag, base_summary = _load(BASELINE_DIR, stem)
    new_meas, new_diag, new_summary = _load(NEW_DIR, stem)

    image = np.asarray(skio.imread(IMAGE_PATH), dtype=np.float64)
    if image.ndim > 2:
        image = image[..., 0]

    outcomes = {"single": [], "suspicious": [], "fallback": [], "other": []}
    per_object = []
    for oid in ZERO_MAXIMA_IDS:
        status = _object_status(new_diag, new_meas, oid)
        per_object.append(status)
        path = str(status.get("path") or "")
        if path in {"single", "fast_single"}:
            outcomes["single"].append(oid)
        elif path in {"gmm", "declump"} or bool(status.get("tried_gmm")):
            outcomes["suspicious"].append(oid)
        elif path == "fallback":
            outcomes["fallback"].append(oid)
        else:
            outcomes["other"].append(oid)

    # Isolated controls: ordinary fast_single objects outside ROI that were fit_ok in baseline.
    cent = base_meas.groupby("object_id").first()[["initial_row", "initial_col"]]
    r0, c0, r1, c1 = ROI
    outside = cent[
        ~((cent.initial_row >= r0) & (cent.initial_row < r1) & (cent.initial_col >= c0) & (cent.initial_col < c1))
    ].index
    control_ids = (
        base_diag[
            (base_diag.object_id.isin(outside))
            & (base_diag.path == "fast_single")
            & (base_diag.n_accepted_fit_ok == 1)
        ]
        .object_id.head(30)
        .tolist()
    )
    control_changed = []
    for oid in control_ids:
        b = base_diag.loc[base_diag.object_id == oid].iloc[0]
        n = new_diag.loc[new_diag.object_id == oid]
        if n.empty:
            control_changed.append({"object_id": oid, "reason": "missing"})
            continue
        nrow = n.iloc[0]
        if str(nrow["path"]) != str(b["path"]) or int(nrow["n_accepted_fit_ok"]) != int(b["n_accepted_fit_ok"]):
            control_changed.append(
                {
                    "object_id": oid,
                    "before_path": b["path"],
                    "after_path": nrow["path"],
                    "before_accepted": int(b["n_accepted_fit_ok"]),
                    "after_accepted": int(nrow["n_accepted_fit_ok"]),
                }
            )

    report = {
        "baseline_summary": {
            "objects": base_summary["summary"]["total_mask_objects"],
            "suspicious": base_summary["summary"]["suspicious_objects"],
            "fallback": base_summary["summary"]["fallback_objects"],
            "runtime_s": base_summary["summary"]["total_runtime_seconds"],
        },
        "new_summary": {
            "objects": new_summary["summary"]["total_mask_objects"],
            "suspicious": new_summary["summary"]["suspicious_objects"],
            "fallback": new_summary["summary"]["fallback_objects"],
            "fast": new_summary["summary"]["fast_path_objects"],
            "single": new_summary["summary"]["single_path_objects"],
            "gmm_triggered": new_summary["summary"]["gmm_triggered_objects"],
            "runtime_s": new_summary["summary"]["total_runtime_seconds"],
            "local_peak_recovery_attempts": new_summary["summary"].get("local_peak_recovery_attempts"),
            "local_peak_recovery_success": new_summary["summary"].get("local_peak_recovery_success"),
            "local_peak_recovery_one_peak": new_summary["summary"].get("local_peak_recovery_one_peak"),
            "local_peak_recovery_multi_peak": new_summary["summary"].get("local_peak_recovery_multi_peak"),
        },
        "timing": new_summary.get("timing", {}),
        "original_36": {
            "n": len(ZERO_MAXIMA_IDS),
            "recovered_single": outcomes["single"],
            "recovered_suspicious": outcomes["suspicious"],
            "remain_fallback": outcomes["fallback"],
            "other": outcomes["other"],
            "n_single": len(outcomes["single"]),
            "n_suspicious": len(outcomes["suspicious"]),
            "n_fallback": len(outcomes["fallback"]),
            "n_other": len(outcomes["other"]),
            "objects": per_object,
        },
        "dense_roi": {
            "yellow_markers_before": _yellow_count(base_meas, ZERO_MAXIMA_IDS),
            "yellow_markers_after": _yellow_count(new_meas, ZERO_MAXIMA_IDS),
            "accepted_centers_before": _accepted_centers(base_meas),
            "accepted_centers_after": _accepted_centers(new_meas),
            "dense_accepted_centers_before": _accepted_centers(base_meas, list(base_diag.object_id)),
        },
        "isolated_controls_n": len(control_ids),
        "isolated_controls_changed": control_changed,
    }

    # Dense ROI object IDs from centroids.
    new_cent = new_meas.groupby("object_id").first()[["initial_row", "initial_col"]]
    dense_ids = [
        int(oid)
        for oid, row in new_cent.iterrows()
        if _in_roi(float(row.initial_row), float(row.initial_col))
    ]
    report["dense_roi"]["n_objects"] = len(dense_ids)
    report["dense_roi"]["yellow_all_dense_before"] = _yellow_count(base_meas, dense_ids)
    report["dense_roi"]["yellow_all_dense_after"] = _yellow_count(new_meas, dense_ids)
    report["dense_roi"]["accepted_fit_ok_dense_before"] = _accepted_centers(base_meas, dense_ids)
    report["dense_roi"]["accepted_fit_ok_dense_after"] = _accepted_centers(new_meas, dense_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "local_recovery_comparison.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(per_object).to_csv(OUT_DIR / "original_36_after_recovery.csv", index=False)
    _save_overlay(image, new_meas, OUT_DIR / "dense_cluster_roi_overlay_after.png")
    _save_overlay(image, base_meas, OUT_DIR / "dense_cluster_roi_overlay_before.png")
    print(json.dumps({k: v for k, v in report.items() if k != "original_36" or True}, indent=2, default=str)[:4000])
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
