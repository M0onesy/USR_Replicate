from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title
from core.logging_utils import log_render, log_step
from core.paper_tail import render_figure_12
from core.runner import run_standalone


TAG = "figure_12"
FIGURE_NUMBER = 12


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    output_path = figure_path(result, FIGURE_NUMBER)
    title = figure_title(FIGURE_NUMBER)
    log_step(TAG, "使用 supplement-backed paper_tail 数据重建 Figure 12。")
    log_render(TAG, f"输出 {output_path.name}")
    render_figure_12(result, output_path, title)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
