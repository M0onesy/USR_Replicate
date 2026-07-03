from __future__ import annotations

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.logging_utils import log_info, log_render, log_step
from core.runner import run_standalone
from core.submission_fast import build_submission_table_v
from tableCode._common import diagnostics_dir, output_path, write_csv, write_json, write_table_with_fallback


TAG = "table_v"
ROMAN = "V"


def export_fast(cfg: RunConfig) -> Path:
    diag_dir = diagnostics_dir(cfg)
    table = build_submission_table_v(cfg)
    diagnostics_path = diag_dir / "table_v_submission_sharpe_ratios.csv"
    write_csv(diagnostics_path, table)
    path = write_table_with_fallback(output_path(cfg, ROMAN), table, "submission", TAG)
    write_json(
        diag_dir / "table_v_submission_summary.json",
        {
            "table": "Table V",
            "mode": "submission_fast_sharpe_ratios",
            "output_path": str(path),
            "diagnostics_path": str(diagnostics_path),
            "rows": int(table.shape[0]),
            "portfolios": table["portfolio"].tolist() if "portfolio" in table else [],
            "note": "PCA rows use standardized return increments recovered from Figure 13 normalized cumulative returns.",
        },
    )
    log_info(TAG, f"Generated submission Table V with {table.shape[0]} rows.")
    return path


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    log_step(TAG, "Build Table V Sharpe-ratio summary.")
    path = export_fast(cfg)
    log_render(TAG, f"Wrote {path.name}")
    return path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
