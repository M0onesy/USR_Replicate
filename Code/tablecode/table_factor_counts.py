"""
tablecode/table_factor_counts.py
================================

Factor-Count Diagnostics —— 扰动特征值比诊断表（论文 Figure 1-2 的数据底座）

论文含义：
  这张诊断表逐年、按 balanced/unbalanced 面板与 hf/continuous/jump 收益分量，给出
  扰动特征值比 (er_1, er_2, ...) 以及由此估计的因子个数 K_hat。Figure 1 / Figure 2
  正是把其中的 HF 行画成 ER 折线。把它单独落表，便于核对图背后的具体数值。

数据来源：
  result.paper_factor_counts（pipeline 阶段已算好）。

输出位置：
  权威路径放在 diagnostics/ 下（与原脚本一致），同时镜像一份旧别名到 tables/。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult, _copy_alias_files
from core.io_utils import diagnostics_dir, tables_dir, _atomic_to_csv
from core.logging_utils import log_step, log_render
from core.runner import run_standalone

TAG = "table_factor_counts"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = diagnostics_dir(result) / "paper_factor_count_diagnostics.csv"
    alias = tables_dir(result) / "Table_09_paper_style_factor_count_diagnostics.csv"

    # ---------------- 数据处理 ----------------
    df = result.paper_factor_counts
    log_step(TAG, f"取出扰动特征值比诊断表（{df.shape[0]} 行 × {df.shape[1]} 列）")

    # ---------------- 表格输出 ----------------
    log_render(TAG, f"写入 {canonical.name} 并镜像别名 {alias.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    _copy_alias_files(canonical, [alias])
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
