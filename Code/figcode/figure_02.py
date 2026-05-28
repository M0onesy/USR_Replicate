"""
figcode/figure_02.py
====================

Figure 2 —— Number of High-Frequency Factors, Balanced Panel
（严格平衡面板下的高频因子个数诊断）

与 Figure 1 完全同构，只是把面板从"非平衡"换成"严格平衡"。绘图逻辑直接复用
figure_01 里的 _plot_er_panel，避免重复实现。

数据来源：
  result.paper_factor_counts，取 panel_block = "Balanced panel"、return_component = "hf"。
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
from figcode.figure_01 import _plot_er_panel

TAG = "figure_02"
FIGURE_NUMBER = 2
PANEL_BLOCK = "Balanced panel"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, f"读取 paper_factor_counts，过滤 {PANEL_BLOCK} / HF 行")
    df = result.paper_factor_counts.copy()
    n_rows = int(df.loc[df.get("panel_block", "").eq(PANEL_BLOCK)].shape[0]) if not df.empty else 0
    log_step(TAG, f"平衡面板候选行数: {n_rows}")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制逐年扰动特征值比折线图")
    _plot_er_panel(df, PANEL_BLOCK, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
