from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import (
    _save_line_plot,
    _save_placeholder_figure,
    figure_path,
    figure_title,
    gc_columns,
    get_rolling_frames,
)
from core.logging_utils import log_render, log_step
from core.runner import run_standalone

TAG = "figure_07"
FIGURE_NUMBER = 7


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    try:
        from figCode._timevar import render_fig7

        log_render(TAG, "Rendering Figure 7 with the local-vs-global weight generalized-correlation implementation.")
        render_fig7(result, cfg, output_path, title)
        return output_path
    except Exception as exc:
        log_render(TAG, f"Paper-style renderer failed; falling back to cached rolling GC lines: {exc!r}")

    log_step(TAG, "Loading rolling generalized-correlation diagnostics.")
    rolling_gc_df, _ = get_rolling_frames(result)
    gc_cols = gc_columns(rolling_gc_df)
    top_cols = gc_cols[: min(7, len(gc_cols))]
    log_step(TAG, f"Fallback uses {len(top_cols)} continuous-factor GC lines from a 21-trading-day window summary.")

    if rolling_gc_df.empty or not top_cols:
        _save_placeholder_figure(output_path, title, "No rolling generalized-correlation data are available.")
        return output_path

    log_render(TAG, "Drawing the top 7 local-vs-global continuous-factor generalized-correlation lines.")
    _save_line_plot(rolling_gc_df, "window_index", top_cols, title, output_path, ylabel="Generalized correlation")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
