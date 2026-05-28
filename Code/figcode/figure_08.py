"""
figcode/figure_08.py
====================

Figure 8 —— Time-Varying Portfolio Weights（随时间变化的组合权重）

论文含义：
  追踪第一个因子（最重要的市场型因子）在滚动窗口下，少数核心股票的组合权重如何
  随时间漂移。用于说明"即便是主因子，其构成也不是一成不变的"。

数据来源：
  result.rolling_weight_summary（列含 start_day, symbol, weight_factor_1）。

数据处理 vs 图表输出：
  - 数据处理：pivot 成 start_day × symbol 的 Factor-1 权重矩阵。
  - 图表输出：每只股票一条随时间变化的折线。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title, _save_line_plot, _save_placeholder_figure
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_08"
FIGURE_NUMBER = 8


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 rolling_weight_summary（{len(result.rolling_weight_summary)} 行）")
    if result.rolling_weight_summary.empty:
        log_render(TAG, "无滚动权重摘要，输出占位图")
        _save_placeholder_figure(output_path, title, "No rolling weight summary is available.")
        return output_path
    pivot = (
        result.rolling_weight_summary
        .pivot_table(index="start_day", columns="symbol", values="weight_factor_1", aggfunc="first")
        .reset_index()
    )
    y_cols = [col for col in pivot.columns if col != "start_day"]
    log_step(TAG, f"透视得到 {len(pivot)} 个窗口起点 × {len(y_cols)} 只股票的 Factor-1 权重")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制 Factor-1 权重随时间变化的折线图")
    _save_line_plot(pivot, "start_day", y_cols, title, output_path, ylabel="Factor 1 weight")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
