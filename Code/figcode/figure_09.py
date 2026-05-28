"""
figcode/figure_09.py
====================

Figure 9 —— Time-Varying Explained Variation（随时间变化的解释方差）

论文含义：
  画出滚动窗口下前 K 个因子能解释的方差占比随时间的变化。解释度越高，说明系统性
  因子在该时段越主导；解释度的波动反映系统性风险结构本身的时变性。

数据来源：
  由 result.rolling_explained_variation 还原出的滚动解释度表
  （列：window_index, explained_variation）。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import (
    figure_path, figure_title, get_rolling_frames,
    _save_line_plot, _save_placeholder_figure,
)
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_09"
FIGURE_NUMBER = 9


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "从 ReplicationResult 还原滚动解释度表")
    _, rolling_explained_df = get_rolling_frames(result)
    log_step(TAG, f"解释度窗口数: {len(rolling_explained_df)}")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制解释方差随滚动窗口的变化曲线")
    if rolling_explained_df.empty:
        _save_placeholder_figure(output_path, title, "No rolling explained-variation data are available.")
        return output_path
    _save_line_plot(
        rolling_explained_df, "window_index", ["explained_variation"],
        title, output_path, ylabel="Explained variation",
    )
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
