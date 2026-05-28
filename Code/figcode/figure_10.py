"""
figcode/figure_10.py
====================

Figure 10 —— Factor-Structure Time Variation Decomposition
（因子结构的时间变化分解）

论文含义：
  把"平均广义相关性 (avg GC)"与"解释方差 (explained variation)"叠在同一张图上，
  从两个角度同时刻画因子结构的时变：GC 看因子空间方向是否稳定，解释度看系统性
  成分的强弱。两条线一起读，能区分"方向变了"还是"强度变了"。

数据来源：
  滚动 GC 表（取 gc_* 列求均值得到 avg_gc）+ 滚动解释度表，按 window_index 合并。
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

TAG = "figure_10"
FIGURE_NUMBER = 10


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "还原滚动 GC 与解释度表，并合并")
    rolling_gc_df, rolling_explained_df = get_rolling_frames(result)
    if rolling_gc_df.empty:
        log_render(TAG, "无滚动 GC 数据，输出占位图")
        _save_placeholder_figure(output_path, title, "No rolling generalized-correlation data are available.")
        return output_path
    gc_cols = gc_columns(rolling_gc_df)
    df = rolling_gc_df.copy()
    df["avg_gc"] = df[gc_cols].mean(axis=1) if gc_cols else np.nan
    if not rolling_explained_df.empty:
        df = df.merge(rolling_explained_df, on="window_index", how="left")
    log_step(TAG, f"合并后窗口数 {len(df)}，可用列: avg_gc / explained_variation")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制 avg_gc 与解释方差的结构分解图")
    y_cols = [col for col in ["avg_gc", "explained_variation"] if col in df.columns]
    _save_line_plot(df, "window_index", y_cols, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
