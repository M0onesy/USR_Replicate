"""
tablecode/table_v.py
====================

Table V —— Intraday / Overnight / Daily Sharpe Ratios
（日内 / 隔夜 / 日度夏普比率）

论文含义：
  论文用切点组合 (tangency portfolio) 把因子组合成最大夏普比率组合，并分别在日内、
  隔夜、日度三个口径上报告夏普比率，同时列出各个连续 PCA 单因子的三段夏普。这张表
  量化了"盘中-隔夜反转"在风险调整收益上的体现。

数据来源 / 数据处理：
  复用 engine 的 build_paper_table_v（= build_factor_sharpe_table），输入是已算好的
  PelgerPipeline（含切点组合夏普与各因子日内/隔夜/日度收益）。

数据处理 vs 表格输出：
  - 数据处理：从 result.pipeline 计算切点组合与逐因子的三段夏普。
  - 表格输出：写入权威 CSV，并镜像旧别名 Table_10。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult, build_paper_table_v, _copy_alias_files
from core.io_utils import table_path, tables_dir, _atomic_to_csv
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_v"
ROMAN = "V"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)
    alias = tables_dir(result) / "Table_10_paper_style_factor_sharpes.csv"

    # ---------------- 数据处理 ----------------
    log_step(TAG, "基于 pipeline 计算切点组合与逐因子日内/隔夜/日度夏普")
    df = build_paper_table_v(result.pipeline)
    log_step(TAG, f"得到 Table V（{df.shape[0]} 行 × {df.shape[1]} 列）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name} 并镜像别名 {alias.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    _copy_alias_files(canonical, [alias])
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
