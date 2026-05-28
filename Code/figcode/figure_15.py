"""
figcode/figure_15.py
====================

Figure 15 —— Asset Pricing of Size- and Value-Sorted Portfolios
（规模 / 价值排序组合的资产定价）

论文含义：
  与 Figure 14 同类，但测试资产换成按规模 (size) 和价值 (value) 双重排序构造的组合，
  检验高频因子对经典 size/value 异象组合的定价能力。

当前状态（外部数据缺口）：
  需要 size/value 排序组合收益作为测试资产，当前仓库未提供，按既定策略输出占位图。
  补全方式同 Figure 14：提供测试资产 CSV 并接入 engine.load_test_asset_csv(...)。
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

TAG = "figure_15"
FIGURE_NUMBER = 15


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "检查 size/value 排序组合测试资产是否可用")
    log_warn(TAG, "缺少 size/value 组合收益（外部数据），将输出占位图")

    # ---------------- 图表输出 ----------------
    log_render(TAG, "输出 Figure 15 占位图")
    _save_placeholder_figure(
        output_path, title,
        "Size/value sorted portfolio returns are not available in the current repository. "
        "Provide external test-asset CSV files to replace this placeholder.",
    )
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
