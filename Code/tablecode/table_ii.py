"""
tablecode/table_ii.py
=====================

Table II —— Balanced and Unbalanced Panel Results（平衡与非平衡面板结果）

论文含义：
  论文要论证"严格平衡面板"提取出的因子能代表更广的非平衡全样本。Table II 通过
  比较平衡面板因子空间与非平衡面板因子空间的"广义相关性 (generalized correlation)"
  来量化这种代表性——相关性高，说明用平衡面板做 PCA 不会丢失系统性结构。

数据来源：
  result.paper_table_ii（逐年、balanced vs unbalanced 的因子空间 GC 汇总）。

数据处理 vs 表格输出：
  - 数据处理：取出已算好的 Table II。
  - 表格输出：写入权威 CSV（无旧别名）。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import table_path, _atomic_to_csv
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_ii"
ROMAN = "II"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)

    # ---------------- 数据处理 ----------------
    df = result.paper_table_ii
    log_step(TAG, f"取出已算好的 Table II（{df.shape[0]} 行 × {df.shape[1]} 列）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
