from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult, load_or_build_submission_figure_factor_counts
from core.io_utils import _atomic_save_figure, _save_placeholder_figure, figure_path, figure_title
from core.logging_utils import log_render, log_step
from core.runner import run_standalone

TAG = "figure_01"
FIGURE_NUMBER = 1
PANEL_BLOCK = "Yearwise balanced, changing universe"


def _build_significance_summary(df: pd.DataFrame, panel_block: str) -> tuple[pd.DataFrame, float]:
    sub = df.copy()
    if "panel_block" in sub.columns and sub["panel_block"].eq(panel_block).any():
        sub = sub.loc[sub["panel_block"].eq(panel_block)].copy()
    if "return_component" in sub.columns:
        sub = sub.loc[sub["return_component"].eq("hf")].copy()
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
                "n_symbols": int(row["n_symbols"]) if "n_symbols" in row and pd.notna(row["n_symbols"]) else np.nan,
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
        _save_placeholder_figure(output_path, title, "No HF perturbed eigenvalue-ratio data are available.")
        return pd.DataFrame()

    years = summary_df["year"].tolist()
    x = np.arange(1, len(er_cols) + 1)
    group_size = 4
    groups = [years[i : i + group_size] for i in range(0, len(years), group_size)] or [years]
    nrows = len(groups)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.8, max(2.8 * nrows + 0.6, 4.0)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, grp in zip(axes, groups):
        group_df = summary_df.loc[summary_df["year"].isin(grp)].copy()
        for _, row in group_df.iterrows():
            y = [float(row[col]) for col in er_cols]
            year = int(row["year"])
            k_hat = int(row["K_hat"])
            max_k = int(row["max_k_above_critical"])
            n_symbols = int(row["n_symbols"]) if pd.notna(row.get("n_symbols")) else None
            label = f"{year} (K={k_hat}" + (f", N={n_symbols})" if n_symbols is not None else ")")
            line, = ax.plot(x, y, marker="o", linewidth=1.4, label=label)
            if 0 < max_k <= len(y):
                y_marker = y[max_k - 1]
                ax.scatter([max_k], [y_marker], color=line.get_color(), s=34, zorder=4)
                ax.annotate(
                    f"K={max_k}",
                    xy=(max_k, y_marker),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=line.get_color(),
                    weight="bold",
                )
        ax.axhline(crit, color="green", linewidth=1.3, label=f"Critical value {crit:.2f}")
        ax.set_xticks(x)
        ax.set_ylabel("Perturbed ER")
        ax.set_title(f"Years {grp[0]}-{grp[-1]}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(title, fontsize=13, y=0.995)
    axes[-1].set_xlabel("k")
    fig.text(
        0.5,
        0.01,
        f"Significant factor count is the largest k with ER_k > 1 + gamma = {crit:.2f}.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
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
    log_step(tag, f"wrote {summary_path.name}; years with K>1: {gt_one}/{len(summary_df)} for {panel_block}")


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    title = figure_title(FIGURE_NUMBER)
    output_path = figure_path(result, FIGURE_NUMBER)

    log_step(TAG, "Loading submission-specific yearly diagnostics for Figure 1.")
    df = load_or_build_submission_figure_factor_counts(result, FIGURE_NUMBER)
    log_step(TAG, f"Figure 1 diagnostic rows: {len(df)}")

    log_render(TAG, "Rendering Figure 1 with cross-year unbalanced but within-year balanced yearly panels.")
    summary_df = _plot_er_panel(df, PANEL_BLOCK, title, output_path)
    _write_significance_summary(result, PANEL_BLOCK, summary_df, tag=TAG)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
