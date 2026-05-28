"""
tablecode/table_weights.py
==========================

Factor Portfolio Weight Tables —— 因子组合权重表（论文 Figure 3-5 的数据底座）

包含三张相关联的权重表，一次性输出：
  - 连续 PCA 因子权重     -> Table_11_continuous_pca_weights.csv      （Figure 4 底座）
  - 代理因子权重          -> Table_12_proxy_factor_weights.csv        （Figure 3 底座）
  - 月频 PCA 因子权重     -> Table_13_monthly_pca_weights.csv         （Figure 5 底座）

论文含义：
  这些表给出每个因子组合在股票层面的权重（按 |权重| 排序的 top 股票），是 Figure 3-5
  热图的数值来源。把权重单独落表，方便核对"哪些股票主导了某个因子"。

数据来源：
  result.pca_weights / proxy_weights / monthly_pca_weights（pipeline 阶段已算好）。

返回值约定：
  本脚本输出多张表，generate 返回"第一张"（连续 PCA 权重）的路径，作为日志展示用；
  其余两张同样会被写出。
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

TAG = "table_weights"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    tdir = tables_dir(result)
    pca_path = tdir / "Table_11_continuous_pca_weights.csv"
    proxy_path = tdir / "Table_12_proxy_factor_weights.csv"
    monthly_path = tdir / "Table_13_monthly_pca_weights.csv"

    # ---------------- 数据处理 ----------------
    log_step(
        TAG,
        f"取出三张权重表：连续 PCA {len(result.pca_weights)} 行 / "
        f"代理 {len(result.proxy_weights)} 行 / 月频 {len(result.monthly_pca_weights)} 行",
    )

    # ---------------- 表格输出 ----------------
    log_render(TAG, "写入连续 PCA / 代理 / 月频三张权重表")
    _atomic_to_csv(result.pca_weights, pca_path, index=False, encoding="utf-8-sig")
    _atomic_to_csv(result.proxy_weights, proxy_path, index=False, encoding="utf-8-sig")
    _atomic_to_csv(result.monthly_pca_weights, monthly_path, index=False, encoding="utf-8-sig")
    return pca_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
