"""
tablecode/table_factor_returns.py
=================================

Factor Return Summary —— 因子收益摘要表（论文 Figure 12-13 的数据底座）

论文含义：
  逐个连续 PCA 因子给出其平均日内、平均隔夜、平均日度收益。这是 Figure 12（分段
  柱状图）的数值来源，也与 Figure 13（累计收益曲线）同源。用于量化"盘中-隔夜
  反转"在均值层面的表现。

数据来源：
  result.factor_return_summary（列：factor, mean_intraday, mean_overnight, mean_daily）。

输出位置：
  Table_14_factor_return_summary.csv（与原脚本一致，放在 tables/ 下）。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import tables_dir, _atomic_to_csv
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_factor_returns"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = tables_dir(result) / "Table_14_factor_return_summary.csv"

    # ---------------- 数据处理 ----------------
    df = result.factor_return_summary
    log_step(TAG, f"取出因子收益摘要（{df.shape[0]} 行 × {df.shape[1]} 列）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
