"""
figcode/figure_13.py
====================

Figure 13 —— Cumulative Factor Returns（因子累计收益）

论文含义：
  画出第一个连续 PCA 因子的累计日内、累计隔夜、累计日度收益曲线。累计视角能直观
  看出"盘中-隔夜反转"的长期效果：日内与隔夜累计曲线常常一升一降，而日度是两者之和。

数据来源：
  result.cumulative_factor_returns（列：date, factor, cum_intraday, cum_overnight,
  cum_daily），按论文展示前 4 个连续因子。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title, _save_cumulative_factor_grid_plot, _save_placeholder_figure
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_13"
FIGURE_NUMBER = 13


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    base_title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 cumulative_factor_returns（{len(result.cumulative_factor_returns)} 行）")
    if result.cumulative_factor_returns.empty:
        log_render(TAG, "无累计收益数据，输出占位图")
        _save_placeholder_figure(output_path, base_title, "No cumulative factor return data are available.")
        return output_path
    df = result.cumulative_factor_returns.copy()
    log_step(TAG, f"累计收益覆盖 {df['factor'].nunique()} 个因子、{len(df)} 行")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制多因子累计日内/隔夜/日度收益曲线")
    _save_cumulative_factor_grid_plot(df, base_title, output_path, ylabel="Cumulative log return")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
