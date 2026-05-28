"""
figcode/figure_01.py
====================

Figure 1 —— Number of High-Frequency Factors, Unbalanced Panel
（非平衡面板下的高频因子个数诊断）

论文含义：
  论文用"扰动特征值比 (perturbed eigenvalue ratio)"来估计因子个数 K。把相关矩阵的
  特征值排序后做扰动比，比值出现明显拐点 / 落差的位置就是估计出的因子数。这张图把
  各年份非平衡面板的 ER 曲线叠在一起，曲线标签里标注每年估计出的 K。

数据来源：
  result.paper_factor_counts（逐年、按 balanced/unbalanced 与 hf/cont/jump 分块的
  扰动特征值比诊断表），取 panel_block = "Unbalanced panel"、return_component = "hf"。

数据处理 vs 图表输出：
  - 数据处理：从 paper_factor_counts 过滤出非平衡面板的 HF 行，提取 er_* 列。
  - 图表输出：逐年画一条 ER 折线，横轴 k，纵轴扰动特征值比。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

import numpy as np

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title, _save_placeholder_figure, _atomic_save_figure
from core.logging_utils import log_step, log_render
from core.runner import run_standalone, run_generator

TAG = "figure_01"
FIGURE_NUMBER = 1
PANEL_BLOCK = "Unbalanced panel"


def _plot_er_panel(df, panel_block: str, title: str, output_path: Path) -> None:
    """复刻 engine 内 _save_er_panel 的绘图逻辑（逐年 ER 折线）。"""
    import matplotlib.pyplot as plt

    if df.empty:
        _save_placeholder_figure(output_path, title, "No factor-count diagnostics are available.")
        return
    sub = df.loc[df["panel_block"].eq(panel_block) & df["return_component"].eq("hf")].copy()
    er_cols = [col for col in sub.columns if col.startswith("er_")]
    if sub.empty or not er_cols:
        _save_placeholder_figure(
            output_path, title,
            f"No HF perturbed eigenvalue-ratio data are available for {panel_block.lower()}.",
        )
        return
    x = np.arange(1, len(er_cols) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for _, row in sub.sort_values("year").iterrows():
        y = [float(row[col]) for col in er_cols]
        label = f"{int(row['year'])} (K={int(row['K_hat'])})"
        ax.plot(x, y, marker="o", linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("Perturbed eigenvalue ratio")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 paper_factor_counts，过滤 {PANEL_BLOCK} / HF 行")
    df = result.paper_factor_counts.copy()
    n_rows = int(df.loc[df.get("panel_block", "").eq(PANEL_BLOCK)].shape[0]) if not df.empty else 0
    log_step(TAG, f"非平衡面板候选行数: {n_rows}")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制逐年扰动特征值比折线图")
    _plot_er_panel(df, PANEL_BLOCK, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
