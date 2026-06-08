"""
figcode/figure_01.py
====================

Figure 1 —— Number of High-Frequency Factors, Unbalanced Panel
（非平衡面板下的高频因子个数诊断）

论文含义：
  论文用“扰动特征值比 (perturbed eigenvalue ratio)”来估计因子个数 K。
  实际显著因子个数由最后一个满足 ER_k > 1 + gamma 的 k 决定，而不是只看最大的 ER_1。
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import _atomic_save_figure, _save_placeholder_figure, figure_path, figure_title
from core.logging_utils import log_render, log_step
from core.runner import run_standalone

TAG = "figure_01"
FIGURE_NUMBER = 1
PANEL_BLOCK = "Unbalanced panel"


def _build_significance_summary(df: pd.DataFrame, panel_block: str) -> tuple[pd.DataFrame, float]:
    sub = df.loc[df["panel_block"].eq(panel_block) & df["return_component"].eq("hf")].copy()
    er_cols = [col for col in sub.columns if col.startswith("er_")]
    if sub.empty or not er_cols:
        return pd.DataFrame(), 1.08

    gamma = 0.08
    if "gamma" in sub.columns and sub["gamma"].notna().any():
        try:
            gamma = float(sub["gamma"].dropna().iloc[0])
        except Exception:
            gamma = 0.08
    crit = 1.0 + gamma

    rows: list[dict[str, float | int | str]] = []
    for _, row in sub.sort_values("year").iterrows():
        er_values = [float(row[col]) for col in er_cols]
        above = [idx + 1 for idx, value in enumerate(er_values) if value > crit]
        rows.append(
            {
                "panel_block": str(panel_block),
                "year": int(row["year"]),
                "K_hat": int(row["K_hat"]),
                "gamma": float(gamma),
                "critical_value": float(crit),
                "max_k_above_critical": int(max(above) if above else 0),
                **{f"er_{idx + 1}": float(value) for idx, value in enumerate(er_values)},
            }
        )
    return pd.DataFrame(rows), crit


def _plot_er_panel(df: pd.DataFrame, panel_block: str, title: str, output_path: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    if df.empty:
        _save_placeholder_figure(output_path, title, "No factor-count diagnostics are available.")
        return pd.DataFrame()

    summary_df, crit = _build_significance_summary(df, panel_block)
    er_cols = [col for col in summary_df.columns if col.startswith("er_")]
    if summary_df.empty or not er_cols:
        _save_placeholder_figure(
            output_path,
            title,
            f"No HF perturbed eigenvalue-ratio data are available for {panel_block.lower()}.",
        )
        return pd.DataFrame()

    years = summary_df["year"].tolist()
    x = np.arange(1, len(er_cols) + 1)
    group_size = 4
    groups = [years[i : i + group_size] for i in range(0, len(years), group_size)] or [years]
    nrows = len(groups)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(12.8, max(3.2 * nrows, 3.6)),
        sharex="col",
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    axes = np.atleast_2d(axes)
    zoom_cols = [col for col in er_cols if col != "er_1"] or er_cols
    zoom_top = max(1.8, float(summary_df[zoom_cols].max().max()) + 0.08)

    for row_axes, grp in zip(axes, groups):
        ax_full, ax_zoom = row_axes
        group_df = summary_df.loc[summary_df["year"].isin(grp)].copy()
        for _, row in group_df.iterrows():
            y = [float(row[col]) for col in er_cols]
            year = int(row["year"])
            k_hat = int(row["K_hat"])
            max_k = int(row["max_k_above_critical"])
            label = f"{year} (K={k_hat})"
            line_full, = ax_full.plot(x, y, marker="o", linewidth=1.4, label=label)
            ax_zoom.plot(x, y, marker="o", linewidth=1.4, color=line_full.get_color(), label=label)
            if 0 < max_k <= len(y):
                y_marker = y[max_k - 1]
                ax_full.scatter([max_k], [y_marker], color=line_full.get_color(), s=34, zorder=4)
                ax_zoom.scatter([max_k], [y_marker], color=line_full.get_color(), s=34, zorder=4)
                ax_zoom.annotate(
                    f"K={max_k}",
                    xy=(max_k, y_marker),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=line_full.get_color(),
                    weight="bold",
                )
        for ax in (ax_full, ax_zoom):
            ax.axhline(crit, color="green", linewidth=1.3, label=f"Critical value {crit:.2f}")
            ax.set_xticks(x)
            ax.grid(True, alpha=0.25)
        ax_full.set_ylabel("Perturbed ER")
        ax_full.set_title(f"Years {grp[0]}-{grp[-1]}: full scale", fontsize=10)
        ax_zoom.set_title(f"Years {grp[0]}-{grp[-1]}: zoom near critical value", fontsize=10)
        ax_zoom.set_ylim(max(0.98, crit - 0.08), zoom_top)
        ax_full.legend(loc="best", fontsize=8, ncol=2)
        ax_zoom.legend(loc="best", fontsize=8, ncol=2)

    axes[0, 0].set_title(title, fontsize=13, pad=10)
    for ax in axes[-1]:
        ax.set_xlabel("k")
    fig.text(
        0.5,
        0.01,
        f"Significant factor count is the largest k with ER_k > 1 + gamma = {crit:.2f}.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    return summary_df


def _write_significance_summary(result: ReplicationResult, panel_block: str, summary_df: pd.DataFrame, *, tag: str) -> None:
    if summary_df.empty:
        return
    diagnostics_dir = Path(getattr(result, "runtime_root", result.output_root)) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / f"{tag}_significance_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gt_one = int((summary_df["K_hat"] > 1).sum())
    log_step(
        tag,
        (
            f"已写出显著性摘要 {summary_path.name}；"
            f"{panel_block} HF 中 K>1 的年份 {gt_one}/{len(summary_df)}。"
        ),
    )


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    log_step(TAG, f"读取 paper_factor_counts，过滤 {PANEL_BLOCK} / HF 行")
    df = result.paper_factor_counts.copy()
    n_rows = int(df.loc[df.get("panel_block", "").eq(PANEL_BLOCK)].shape[0]) if not df.empty else 0
    log_step(TAG, f"非平衡面板候选行数: {n_rows}")

    log_render(TAG, "绘制逐年扰动特征值比图，并突出 ER_k > 1+gamma 的显著因子个数")
    summary_df = _plot_er_panel(df, PANEL_BLOCK, title, output_path)
    _write_significance_summary(result, PANEL_BLOCK, summary_df, tag=TAG)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
