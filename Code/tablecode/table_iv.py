"""
tablecode/table_iv.py
=====================

Table IV —— Time-Variation Decomposition（时间变化分解汇总）

论文含义：
  把滚动窗口分析得到的广义相关性 (GC) 与解释方差 (explained variation) 做跨窗口的
  统计汇总（均值、中位数、最小、最大），用一张表概括"因子结构随时间变化的幅度"。
  对应 Figure 6/7/9/10/11 背后的数字版本。

数据来源 / 数据处理：
  这张表不是从 ReplicationResult 直接取出的静态字段，而是由滚动结果即时聚合得到。
  这里复用 engine 的 build_paper_table_iv（口径与原脚本完全一致），输入是从
  ReplicationResult 还原出的滚动 GC / 解释度两张表。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult, build_paper_table_iv
from core.io_utils import table_path, get_rolling_frames, _atomic_to_csv
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_iv"
ROMAN = "IV"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)

    # ---------------- 数据处理 ----------------
    log_step(TAG, "从 ReplicationResult 还原滚动 GC / 解释度表")
    rolling_gc_df, rolling_explained_df = get_rolling_frames(result)
    log_step(TAG, f"GC 窗口数 {len(rolling_gc_df)}，解释度窗口数 {len(rolling_explained_df)}，开始聚合")
    df = build_paper_table_iv(rolling_gc_df, rolling_explained_df)
    log_step(TAG, f"聚合得到 Table IV（{df.shape[0]} 行 × {df.shape[1]} 列）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
