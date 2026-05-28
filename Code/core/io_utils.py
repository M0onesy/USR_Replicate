"""
core/io_utils.py
================

各图表脚本共享的输出工具层。把"输出目录、文件命名、绘图/落表底层函数、滚动结果
切片"等公共逻辑集中在这里，让 figcode / tablecode 下的每篇脚本保持轻薄、聚焦。

设计：
  - 图 / 表的"权威文件名"与原始 allcode_Need.py 保持一致，确保拆分前后产物可对照。
  - 直接复用 engine 里的底层绘图函数（_save_line_plot / _save_bar_plot /
    _save_heatmap / _save_placeholder_figure）和落表函数（_atomic_to_csv），
    不重写，保证视觉与数值都和原脚本一致。
  - get_rolling_frames(result) 复刻 engine 内部 _rolling_output_frames 的口径，
    供 Figure 6/7/9/10/11 与 Table IV 复用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.engine import (
    ReplicationResult,
    _rolling_output_frames,
    _atomic_to_csv,
    _save_line_plot,
    _save_bar_plot,
    _save_heatmap,
    _save_placeholder_figure,
    _atomic_save_figure,
)

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------

def figures_dir(result: ReplicationResult) -> Path:
    d = Path(result.output_root) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir(result: ReplicationResult) -> Path:
    d = Path(result.output_root) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def diagnostics_dir(result: ReplicationResult) -> Path:
    d = Path(result.output_root) / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 权威文件名（与 allcode_Need.py 保持一致）
# ---------------------------------------------------------------------------

# figure_number -> (figure_id, file_name, title)
FIGURE_SPECS: dict[int, tuple[str, str, str]] = {
    1: ("Figure_1", "Figure_1_number_of_hf_factors_unbalanced.png", "Figure 1. Number of High-Frequency Factors, Unbalanced Panel"),
    2: ("Figure_2", "Figure_2_number_of_hf_factors_balanced.png", "Figure 2. Number of High-Frequency Factors, Balanced Panel"),
    3: ("Figure_3", "Figure_3_proxy_factor_portfolio_weights.png", "Figure 3. Proxy Factor Portfolio Weights"),
    4: ("Figure_4", "Figure_4_continuous_pca_factor_portfolio_weights.png", "Figure 4. Continuous PCA Factor Portfolio Weights"),
    5: ("Figure_5", "Figure_5_monthly_pca_factor_portfolio_weights.png", "Figure 5. Monthly PCA Factor Portfolio Weights"),
    6: ("Figure_6", "Figure_6_time_variation_in_loadings.png", "Figure 6. Time Variation in Loadings"),
    7: ("Figure_7", "Figure_7_locally_estimated_continuous_factors.png", "Figure 7. Locally Estimated Continuous Factors"),
    8: ("Figure_8", "Figure_8_time_varying_portfolio_weights.png", "Figure 8. Time-Varying Portfolio Weights"),
    9: ("Figure_9", "Figure_9_time_varying_explained_variation.png", "Figure 9. Time-Varying Explained Variation"),
    10: ("Figure_10", "Figure_10_factor_structure_time_variation_decomposition.png", "Figure 10. Factor-Structure Time Variation Decomposition"),
    11: ("Figure_11", "Figure_11_continuous_factor_structure_decomposition.png", "Figure 11. Continuous Factor-Structure Decomposition"),
    12: ("Figure_12", "Figure_12_expected_intraday_and_overnight_returns.png", "Figure 12. Expected Intraday and Overnight Returns"),
    13: ("Figure_13", "Figure_13_cumulative_factor_returns.png", "Figure 13. Cumulative Factor Returns"),
    14: ("Figure_14", "Figure_14_asset_pricing_of_industry_portfolios.png", "Figure 14. Asset Pricing of Industry Portfolios"),
    15: ("Figure_15", "Figure_15_asset_pricing_of_size_and_value_sorted_portfolios.png", "Figure 15. Asset Pricing of Size- and Value-Sorted Portfolios"),
}

# 论文主表（罗马数字）权威文件名
TABLE_PAPER_FILES: dict[str, str] = {
    "I": "Table_I_summary_statistics_for_continuous_and_jump_returns.csv",
    "II": "Table_II_balanced_and_unbalanced_panel_results.csv",
    "III": "Table_III_generalized_correlations_with_industry_and_ffc_factors.csv",
    "IV": "Table_IV_time_variation_decomposition.csv",
    "V": "Table_V_intraday_overnight_daily_sharpe_ratios.csv",
}


def figure_path(result: ReplicationResult, number: int) -> Path:
    return figures_dir(result) / FIGURE_SPECS[number][1]


def figure_title(number: int) -> str:
    return FIGURE_SPECS[number][2]


def table_path(result: ReplicationResult, roman: str) -> Path:
    return tables_dir(result) / TABLE_PAPER_FILES[roman]


# ---------------------------------------------------------------------------
# 滚动结果切片（复用 engine 口径）
# ---------------------------------------------------------------------------

def get_rolling_frames(result: ReplicationResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 ReplicationResult 还原滚动 GC 与解释度两张表，口径与原脚本一致。"""
    return _rolling_output_frames(result.rolling_gc, result.rolling_explained_variation)


def gc_columns(rolling_gc_df: pd.DataFrame) -> list[str]:
    return [col for col in rolling_gc_df.columns if col.startswith("gc_")]


__all__ = [
    "figures_dir",
    "tables_dir",
    "diagnostics_dir",
    "FIGURE_SPECS",
    "TABLE_PAPER_FILES",
    "figure_path",
    "figure_title",
    "table_path",
    "get_rolling_frames",
    "gc_columns",
    # re-exported engine helpers
    "_atomic_to_csv",
    "_save_line_plot",
    "_save_bar_plot",
    "_save_heatmap",
    "_save_placeholder_figure",
    "_atomic_save_figure",
]
