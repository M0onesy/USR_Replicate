"""
figcode/figure_05.py
====================

Figure 5 —— Monthly PCA Factor Portfolio Weights（月频 PCA 因子组合权重热图）

论文含义：
  把日收益聚合到月频后再做 PCA，得到低频因子的组合权重。与高频 PCA（Figure 4）对比，
  用来说明"低频会丢失哪些信息"——这是论文强调高频方法价值的关键对照之一。

数据来源：
  result.monthly_pca_weights（列：factor, rank, symbol, weight）。

绘图逻辑复用 figure_03.plot_weight_heatmap。
"""

from __future__ import annotations

# --- 允许 `python figcode/xxx.py` / `python tablecode/xxx.py` 直接运行时找到 core 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import figure_path, figure_title
from core.logging_utils import log_step, log_render
from core.runner import run_standalone
from figcode.figure_03 import plot_weight_heatmap, plot_industry_sorted_bars, _load_industry_lookup

TAG = "figure_05"
FIGURE_NUMBER = 5


def _full_monthly_weight_heatmap(result, cfg, title: str, output_path: Path) -> None:
    """1:1 论文 Figure 5：对【全部股票】重算月频 PCA 载荷（含 N12 符号定向），
    按行业排序画热图。异常回退到旧的 top-N 表热图。"""
    try:
        import core.engine as eng
        import numpy as np
        import pandas as pd

        panel = result.panel
        months = pd.Index([pd.Timestamp(d).to_period("M").to_timestamp() for d in panel.dates])
        labels = sorted(pd.unique(months))
        monthly = np.zeros((len(labels), panel.N), dtype=float)
        R_daily = np.asarray(panel.R_daily, dtype=float)
        for mi, m in enumerate(labels):
            mask = (months == m).to_numpy() if hasattr(months == m, "to_numpy") else np.asarray(months == m)
            monthly[mi] = np.nansum(R_daily[mask], axis=0)
        if monthly.shape[0] < 3:
            raise RuntimeError("月频样本过少")
        res = eng._panel_pca(monthly, K=min(4, panel.N), use_corr=True) if hasattr(eng, "_panel_pca") else eng.pca_factors(monthly, K=min(4, panel.N), use_corr=True)
        eng.orient_pca_result(res)  # N12：月频 PCA 也做确定性符号定向
        Lam = np.asarray(res.Lambda, dtype=float)
        W = Lam / np.sqrt(max(Lam.shape[0], 1))   # 特征向量尺度（与 Fig 4 一致）
        lookup = _load_industry_lookup(cfg)
        plot_industry_sorted_bars(W, panel.tickers, lookup, min(W.shape[1], 4), title, output_path)
    except Exception as exc:
        log_render(TAG, f"月频行业排序热图回退到旧实现: {exc!r}")
        plot_weight_heatmap(result.monthly_pca_weights, title, output_path)


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)
    log_step(TAG, "重算全市场月频 PCA 载荷（含符号定向），按行业排序")
    log_render(TAG, "绘制月频 PCA 因子权重热图（按行业排序，1:1 论文 Figure 5）")
    _full_monthly_weight_heatmap(result, cfg, title, output_path)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
