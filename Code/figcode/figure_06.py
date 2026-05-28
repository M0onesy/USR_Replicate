"""
figcode/figure_06.py
====================

Figure 6 —— Time Variation in Loadings（载荷的时间变化）

论文含义：
  论文用滚动窗口 PCA 估计局部因子，并和全局因子算"广义相关性 (generalized
  correlation, GC)"。GC 接近 1 表示局部因子空间与全局一致；明显偏离 1 表示载荷
  在该时间段发生了结构性变化。这张图把每个因子的 GC 随滚动窗口的轨迹都画出来。

数据来源：
  由 result.rolling_gc 还原出的滚动 GC 表（列：window_index, gc_1, gc_2, ...）。

数据处理 vs 图表输出：
  - 数据处理：从 ReplicationResult 还原滚动 GC 表，取出全部 gc_* 列。
  - 图表输出：以 window_index 为横轴的多线折线图。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import (
    figure_path, figure_title, get_rolling_frames, gc_columns,
    _save_line_plot, _save_placeholder_figure,
)
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_06"
FIGURE_NUMBER = 6


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "从 ReplicationResult 还原滚动广义相关性 (GC) 表")
    rolling_gc_df, _ = get_rolling_frames(result)
    gc_cols = gc_columns(rolling_gc_df)
    log_step(TAG, f"窗口数 {len(rolling_gc_df)}，GC 序列数 {len(gc_cols)}")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制 GC 随滚动窗口的时间变化曲线")
    if rolling_gc_df.empty or not gc_cols:
        _save_placeholder_figure(output_path, title, "No rolling generalized-correlation data are available.")
        return output_path
    _save_line_plot(rolling_gc_df, "window_index", gc_cols, title, output_path, ylabel="Generalized correlation")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
