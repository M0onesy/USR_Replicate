"""
tablecode/table_coverage.py
===========================

Replication Coverage Report —— 复现覆盖度报告

含义：
  逐项列出论文的每张表 / 图在当前 A 股数据条件下的复现状态（已适配实现 /
  需外部数据），并附中文说明。这不是论文里的表，而是本复现项目自带的"清单"，
  方便一眼看出哪些已完成、哪些还差外部数据（Table III、Figure 14/15）。

数据来源：
  result.replication_coverage —— engine 的 build_replication_coverage_report()。

输出位置：
  diagnostics/replication_coverage_report.csv（与原脚本一致）。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import diagnostics_dir, _atomic_to_csv
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_coverage"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = diagnostics_dir(result) / "replication_coverage_report.csv"

    # ---------------- 数据处理 ----------------
    df = result.replication_coverage
    log_step(TAG, f"取出复现覆盖度报告（{df.shape[0]} 行）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
