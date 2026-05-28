"""
figcode/figure_12.py
====================

Figure 12 —— Expected Intraday and Overnight Returns（预期日内与隔夜收益）

论文含义：
  这是论文最重要的发现之一——"盘中-隔夜反转"。把每个连续 PCA 因子的平均日内、
  隔夜、日度收益并排画出。论文发现某些因子的日内与隔夜收益方向相反（盘中赚的、
  隔夜还回去，或反之），这是低频数据完全看不到的高频结构。

数据来源：
  result.factor_return_summary（列：factor, mean_intraday, mean_overnight, mean_daily）。

数据处理 vs 图表输出：
  - 数据处理：把宽表 melt 成 (factor, segment, mean_return) 长表。
  - 图表输出：按因子分组的分段柱状图。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title, _save_bar_plot, _save_placeholder_figure
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_12"
FIGURE_NUMBER = 12


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 factor_return_summary（{len(result.factor_return_summary)} 行）")
    if result.factor_return_summary.empty:
        log_render(TAG, "无因子收益摘要，输出占位图")
        _save_placeholder_figure(output_path, title, "No factor return summary is available.")
        return output_path
    long_df = result.factor_return_summary.melt(id_vars="factor", var_name="segment", value_name="mean_return")
    log_step(TAG, f"melt 后得到 {len(long_df)} 行（因子 × 时段）")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制日内/隔夜/日度平均收益分组柱状图")
    _save_bar_plot(long_df, "factor", "mean_return", title, output_path, group_col="segment", ylabel="Mean return")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
