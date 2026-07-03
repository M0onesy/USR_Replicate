from __future__ import annotations

from pathlib import Path

from prepareCore.config import RunConfig
from prepareCore.engine import ReplicationResult
from prepareCore.logging_utils import log_info, log_render, log_step
from prepareCore.runner import run_standalone
from prepareCore.submission_fast import build_submission_table_ii_paper_style
from tableCode._common import diagnostics_dir, output_path, write_csv, write_json, write_table_with_fallback


TAG = "table_ii"
ROMAN = "II"


def export_fast(cfg: RunConfig) -> Path:
    diag_dir = diagnostics_dir(cfg)
    wide, long = build_submission_table_ii_paper_style(cfg, jump_a=float(cfg.jump_a))
    diagnostics_wide_path = diag_dir / "table_ii_paper_style_fixed_k.csv"
    diagnostics_long_path = diag_dir / "table_ii_paper_style_fixed_k_long.csv"
    write_csv(diagnostics_wide_path, wide)
    write_csv(diagnostics_long_path, long)
    path = write_table_with_fallback(output_path(cfg, ROMAN), wide, "paper_style_fixed_k", TAG)
    write_json(
        diag_dir / "table_ii_paper_style_fixed_k_summary.json",
        {
            "table": "Table II",
            "mode": "submission_fast_paper_style_fixed_k",
            "output_path": str(path),
            "diagnostics_wide_path": str(diagnostics_wide_path),
            "diagnostics_long_path": str(diagnostics_long_path),
            "rows": int(wide.shape[0]),
            "long_rows": int(long.shape[0]),
            "blocks": wide["block"].dropna().unique().tolist() if "block" in wide else [],
        },
    )
    log_info(TAG, f"Generated paper-style fixed-K Table II with {wide.shape[0]} rows.")
    return path


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    log_step(TAG, "Build Table II from fixed-intersection and yearly balanced panels.")
    path = export_fast(cfg)
    log_render(TAG, f"Wrote {path.name}")
    return path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
