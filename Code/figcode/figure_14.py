"""
figcode/figure_14.py
====================

Figure 14 —— Asset Pricing of Industry Portfolios（行业组合的资产定价）

论文含义：
  论文用估计出的高频因子去对一组"行业组合"测试资产做资产定价检验，看因子能否
  解释这些组合的收益（定价误差是否接近 0）。

当前状态（外部数据缺口）：
  该图需要行业组合收益作为测试资产，当前 A 股复现仓库未提供这份外部数据，因此
  按既定策略输出一张明确的占位图，避免论文图编号出现"静默缺失"。要补全请提供
  行业组合收益 CSV，并接入 engine.load_test_asset_csv(...)。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title, _save_placeholder_figure
from core.logging_utils import log_step, log_render, log_warn
from core.runner import run_standalone

TAG = "figure_14"
FIGURE_NUMBER = 14


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "检查行业组合测试资产是否可用")
    log_warn(TAG, "缺少行业组合收益（外部数据），将输出占位图")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "输出 Figure 14 占位图")
    _save_placeholder_figure(
        output_path, title,
        "Industry portfolio returns are not available in the current repository. "
        "Provide external test-asset CSV files to replace this placeholder.",
    )
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
