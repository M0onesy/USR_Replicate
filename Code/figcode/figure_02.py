from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from prepareCore.config import RunConfig
from prepareCore.engine import ReplicationResult, load_or_build_submission_figure_factor_counts
from prepareCore.io_utils import figure_path, figure_title
from prepareCore.logging_utils import log_render, log_step
from prepareCore.runner import run_standalone
from figCode.figure_01 import _plot_er_panel, _write_significance_summary

TAG = "figure_02"
FIGURE_NUMBER = 2
PANEL_BLOCK = "Fixed-intersection yearly slices"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    log_step(TAG, "Loading submission-specific fixed-intersection yearly diagnostics for Figure 2.")
    df = load_or_build_submission_figure_factor_counts(result, FIGURE_NUMBER)
    log_step(TAG, f"Figure 2 diagnostic rows: {len(df)}")

    log_render(TAG, "Rendering Figure 2 with fixed full-sample intersection yearly slices.")
    summary_df = _plot_er_panel(df, PANEL_BLOCK, title, output_path)
    _write_significance_summary(result, PANEL_BLOCK, summary_df, tag=TAG)
    if not summary_df.empty:
        gt_one = int((summary_df["K_hat"] > 1).sum())
        n_symbols = summary_df["n_symbols"].dropna().astype(int).unique().tolist() if "n_symbols" in summary_df.columns else []
        log_step(
            TAG,
            f"Figure 2 years with K>1: {gt_one}/{len(summary_df)}; yearly N values: {n_symbols or ['unknown']}",
        )
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
