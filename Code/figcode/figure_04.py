"""
figcode/figure_04.py
====================

Figure 4 —— Continuous PCA Factor Portfolio Weights（连续 PCA 因子组合权重热图）

论文含义：
  与 Figure 3 同构，但画的是"连续部分"PCA 因子（剔除跳跃后的收益做 PCA）得到的
  组合权重。对比 3、4 可看出代理因子是否抓住了 PCA 因子的主结构。

数据来源：
  result.pca_weights（列：factor, rank, symbol, weight）。

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

TAG = "figure_04"
FIGURE_NUMBER = 4


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 pca_weights（{len(result.pca_weights)} 行），准备透视为因子×股票矩阵")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制连续 PCA 因子权重热图")
    plot_weight_heatmap(result.pca_weights, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
