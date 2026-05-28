"""
figcode/figure_07.py
====================

Figure 7 —— Locally Estimated Continuous Factors（局部估计的连续因子）

论文含义：
  聚焦前几个（最多 4 个）主因子的局部 vs 全局广义相关性轨迹。相比 Figure 6 画全部
  GC 序列，这里只看主因子，更清楚地展示"最重要的几个连续因子在时间上有多稳定"。

数据来源：
  与 Figure 6 同源（滚动 GC 表），但只取前 min(4, 序列数) 条 gc_* 列。
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

TAG = "figure_07"
FIGURE_NUMBER = 7


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "还原滚动 GC 表，并取前 4 个主因子的 GC 序列")
    rolling_gc_df, _ = get_rolling_frames(result)
    gc_cols = gc_columns(rolling_gc_df)
    top_cols = gc_cols[: min(4, len(gc_cols))]
    log_step(TAG, f"窗口数 {len(rolling_gc_df)}，选取主因子 GC 序列 {len(top_cols)} 条")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "绘制前 4 个连续因子的局部-全局 GC 曲线")
    if rolling_gc_df.empty or not top_cols:
        _save_placeholder_figure(output_path, title, "No rolling generalized-correlation data are available.")
        return output_path
    _save_line_plot(rolling_gc_df, "window_index", top_cols, title, output_path, ylabel="Generalized correlation")
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
