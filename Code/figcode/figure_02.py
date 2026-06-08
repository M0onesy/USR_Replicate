"""
figcode/figure_02.py
====================

Figure 2 —— Number of High-Frequency Factors, Balanced Panel
（严格平衡面板下的高频因子个数诊断）
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title
from core.logging_utils import log_render, log_step
from core.runner import run_standalone
from figcode.figure_01 import _plot_er_panel, _write_significance_summary

TAG = "figure_02"
FIGURE_NUMBER = 2
PANEL_BLOCK = "Balanced panel"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    log_step(TAG, f"读取 paper_factor_counts，过滤 {PANEL_BLOCK} / HF 行")
    df = result.paper_factor_counts.copy()
    n_rows = int(df.loc[df.get("panel_block", "").eq(PANEL_BLOCK)].shape[0]) if not df.empty else 0
    log_step(TAG, f"平衡面板候选行数: {n_rows}")

    log_render(TAG, "绘制逐年扰动特征值比图，并突出 ER_k > 1+gamma 的显著因子个数")
    summary_df = _plot_er_panel(df, PANEL_BLOCK, title, output_path)
    _write_significance_summary(result, PANEL_BLOCK, summary_df, tag=TAG)
    if not summary_df.empty:
        gt_one = int((summary_df["K_hat"] > 1).sum())
        log_step(
            TAG,
            (
                f"Figure 2 使用的是逐年 balanced-panel K_hat；"
                f"{gt_one}/{len(summary_df)} 个年份的显著因子数大于 1，"
                f"不要与全样本 pipeline.K_hf_hat={int(result.pipeline.K_hf_hat)} 混淆。"
            ),
        )
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
