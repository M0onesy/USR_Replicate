"""
tablecode/table_iii.py
======================

Table III —— Generalized Correlations with Industry and FFC Factors
（与行业因子、Fama-French-Carhart 因子的广义相关性）

论文含义：
  论文把统计 PCA 因子与"行业组合因子"以及"FFC 四因子"做广义相关性比较，回答
  "纯统计的高频因子，能否被既有的特征驱动因子（行业、规模、价值、动量）所近似"。

当前状态（外部数据缺口）：
  这张表需要行业组合收益与 FFC 因子两份外部数据，当前仓库未提供。按既定策略输出
  一张"状态说明表"（明确写出缺哪份数据、应由哪个 loader 接入），而不是静默缺失。

数据来源：
  result.paper_table_iii —— engine 的 build_paper_table_iii() 生成的占位说明表。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import table_path, _atomic_to_csv
from core.logging_utils import log_step, log_render, log_warn
from core.runner import run_standalone

TAG = "table_iii"
ROMAN = "III"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)

    # ---------------- 数据处理 ----------------
    df = result.paper_table_iii
    log_step(TAG, f"取出 Table III 状态说明表（{df.shape[0]} 行）")
    log_warn(TAG, "Table III 依赖行业组合收益与 FFC 因子（外部数据），当前为占位说明")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
