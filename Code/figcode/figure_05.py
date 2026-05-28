"""
figcode/figure_05.py
====================

Figure 5 —— Monthly PCA Factor Portfolio Weights（月频 PCA 因子组合权重热图）

论文含义：
  把日收益聚合到月频后再做 PCA，得到低频因子的组合权重。与高频 PCA（Figure 4）对比，
  用来说明"低频会丢失哪些信息"——这是论文强调高频方法价值的关键对照之一。

数据来源：
  result.monthly_pca_weights（列：factor, rank, symbol, weight）。

绘图逻辑复用 figure_03.plot_weight_heatmap。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title
from core.logging_utils import log_step, log_render
from core.runner import run_standalone
from figcode.figure_03 import plot_weight_heatmap

TAG = "figure_05"
FIGURE_NUMBER = 5


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 monthly_pca_weights（{len(result.monthly_pca_weights)} 行），准备透视为因子×股票矩阵")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制月频 PCA 因子权重热图")
    plot_weight_heatmap(result.monthly_pca_weights, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
