"""
figcode/figure_03.py
====================

Figure 3 —— Proxy Factor Portfolio Weights（代理因子组合权重热图）

论文含义：
  论文把统计 PCA 因子用"少量股票的稀疏组合"去近似（proxy factors），考察这些代理
  组合的权重在股票层面的分布。热图行是因子、列是股票，颜色表示该股票在该因子代理
  组合里的权重。直观地展示因子的经济含义（哪些股票主导了某个因子）。

数据来源：
  result.proxy_weights（列：factor, rank, symbol, weight）。

数据处理 vs 图表输出：
  - 数据处理：把长表 pivot 成 因子 × 股票 的权重矩阵。
  - 图表输出：imshow 热图。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title, _save_heatmap, _save_placeholder_figure
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_03"
FIGURE_NUMBER = 3


def plot_weight_heatmap(df, title: str, output_path: Path) -> None:
    """复刻 engine 内 weight_heatmap：长表 -> 因子×股票矩阵 -> 热图。"""
    if df.empty:
        _save_placeholder_figure(output_path, title, "No portfolio-weight data are available.")
        return
    pivot = df.pivot_table(index="factor", columns="symbol", values="weight", aggfunc="first").fillna(0.0)
    _save_heatmap(
        pivot.to_numpy(),
        pivot.columns.tolist(),
        [f"Factor {idx}" for idx in pivot.index],
        title,
        output_path,
    )


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 proxy_weights（{len(result.proxy_weights)} 行），准备透视为因子×股票矩阵")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制代理因子权重热图")
    plot_weight_heatmap(result.proxy_weights, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
