from __future__ import annotations

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.logging_utils import log_info, log_render, log_step
from core.runner import run_standalone
from core.submission_fast import build_submission_table_iii
from tableCode._common import diagnostics_dir, output_path, write_csv, write_json, write_table_with_fallback


TAG = "table_iii"
ROMAN = "III"


def export_fast(cfg: RunConfig) -> Path:
    diag_dir = diagnostics_dir(cfg)
    table = build_submission_table_iii(cfg)
    diagnostics_path = diag_dir / "table_iii_submission_gc.csv"
    write_csv(diagnostics_path, table)
    path = write_table_with_fallback(output_path(cfg, ROMAN), table, "submission", TAG)
    write_json(
        diag_dir / "table_iii_submission_summary.json",
        {
            "table": "Table III",
            "mode": "submission_fast_gc",
            "output_path": str(path),
            "diagnostics_path": str(diagnostics_path),
            "rows": int(table.shape[0]),
            "comparisons": table["comparison"].tolist() if "comparison" in table else [],
        },
    )
    log_info(TAG, f"Generated submission Table III with {table.shape[0]} rows.")
    return path


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    log_step(TAG, "Build Table III generalized-correlation summary.")
    path = export_fast(cfg)
    log_render(TAG, f"Wrote {path.name}")
    return path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
