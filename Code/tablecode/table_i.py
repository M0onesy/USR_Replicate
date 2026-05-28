"""
tablecode/table_i.py
====================

Table I —— Summary Statistics for Continuous and Jump Returns
（连续收益与跳跃收益的汇总统计）

论文含义：
  论文先用阈值法把高频收益拆成"连续部分"和"跳跃部分"，再在不同阈值倍数 a 下统计
  跳跃的频率、占总变差的比例等。Table I 就是这些跳跃统计的多阈值、按面板分块的汇总，
  用来回答"A 股高频收益里跳跃成分有多大、随阈值如何变化"。

数据来源：
  result.paper_table_i —— 已由 engine 的逐年 yearly_paper_outputs 计算完成（重活在
  pipeline 阶段已做完，这里只负责落表）。

数据处理 vs 表格输出：
  - 数据处理：从 ReplicationResult 取出已算好的 Table I，并准备兼容别名路径。
  - 表格输出：写入权威 CSV，并镜像到旧命名别名 Table_08。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import table_path, tables_dir, _atomic_to_csv
from core.engine import _copy_alias_files
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_i"
ROMAN = "I"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)
    # 旧命名别名（与 allcode_Need.py 一致），方便老脚本 / 老引用继续工作。
    alias = tables_dir(result) / "Table_08_paper_style_yearly_jump_stats.csv"

    # ---------------- 数据处理 ----------------
    df = result.paper_table_i
    log_step(TAG, f"取出已算好的 Table I（{df.shape[0]} 行 × {df.shape[1]} 列）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name} 并镜像别名 {alias.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    _copy_alias_files(canonical, [alias])
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
