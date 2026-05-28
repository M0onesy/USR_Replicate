"""
figcode/figure_11.py
====================

Figure 11 —— Continuous Factor-Structure Decomposition（连续因子结构分解）

论文含义：
  同时画出滚动窗口里 GC 的"最小值 (min_gc)"与"均值 (mean_gc)"。min_gc 反映
  最不稳定的那个因子方向，mean_gc 反映整体平均稳定性。两者拉开差距，说明因子
  之间稳定性差异很大——某些方向稳定、某些方向漂移明显。

数据来源：
  滚动 GC 表，对 gc_* 列分别求 min 与 mean。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

import numpy as np

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import (
    figure_path, figure_title, get_rolling_frames, gc_columns,
    _save_line_plot, _save_placeholder_figure,
)
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "figure_11"
FIGURE_NUMBER = 11


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "还原滚动 GC 表，并计算 min_gc / mean_gc")
    rolling_gc_df, _ = get_rolling_frames(result)
    if rolling_gc_df.empty:
        log_render(TAG, "无滚动 GC 数据，输出占位图")
        _save_placeholder_figure(output_path, title, "No rolling generalized-correlation data are available.")
        return output_path
    gc_cols = gc_columns(rolling_gc_df)
    df = rolling_gc_df.copy()
    df["min_gc"] = df[gc_cols].min(axis=1) if gc_cols else np.nan
    df["mean_gc"] = df[gc_cols].mean(axis=1) if gc_cols else np.nan
    log_step(TAG, f"窗口数 {len(df)}，已构造 min_gc / mean_gc")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制连续因子 min/mean GC 分解图")
    _save_line_plot(df, "window_index", ["min_gc", "mean_gc"], title, output_path, ylabel="Generalized correlation")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
