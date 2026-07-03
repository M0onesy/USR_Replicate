from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from prepareCore.config import RunConfig
from prepareCore.engine import ReplicationResult
from prepareCore.io_utils import figure_path, figure_title
from prepareCore.logging_utils import log_render, log_step
from prepareCore.paper_tail import render_figure_15
from prepareCore.runner import run_standalone


TAG = "figure_15"
FIGURE_NUMBER = 15


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    output_path = figure_path(result, FIGURE_NUMBER)
    title = figure_title(FIGURE_NUMBER)
    log_step(TAG, "使用 supplement-backed paper_tail 数据重建 Figure 15。")
    log_render(TAG, f"输出 {output_path.name}")
    render_figure_15(result, output_path, title)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
