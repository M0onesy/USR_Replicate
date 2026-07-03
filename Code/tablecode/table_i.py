from __future__ import annotations

from pathlib import Path

from prepareCore.config import RunConfig
from prepareCore.engine import ReplicationResult
from prepareCore.logging_utils import log_info, log_render, log_step
from prepareCore.runner import run_standalone
from prepareCore.submission_fast import build_submission_table_i_aligned
from tableCode._common import diagnostics_dir, output_path, write_csv, write_json, write_table_with_fallback


TAG = "table_i"
ROMAN = "I"


def export_fast(cfg: RunConfig) -> Path:
    diag_dir = diagnostics_dir(cfg)
    table = build_submission_table_i_aligned(cfg)
    diagnostics_path = diag_dir / "table_i_aligned_with_figures.csv"
    write_csv(diagnostics_path, table)
    path = write_table_with_fallback(output_path(cfg, ROMAN), table, "aligned", TAG)
    write_json(
        diag_dir / "table_i_aligned_summary.json",
        {
            "table": "Table I",
            "mode": "submission_fast_aligned",
            "output_path": str(path),
            "diagnostics_path": str(diagnostics_path),
            "panels": sorted(table["panel_block"].dropna().unique().tolist()) if "panel_block" in table else [],
            "rows": int(table.shape[0]),
        },
    )
    log_info(TAG, f"Generated aligned Table I with {table.shape[0]} rows.")
    return path


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    log_step(TAG, "Build Table I from submission Figure 1/2 panel definitions.")
    path = export_fast(cfg)
    log_render(TAG, f"Wrote {path.name}")
    return path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
