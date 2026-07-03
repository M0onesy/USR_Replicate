"""
figCode/_weights.py
===================

Figure 3 —— Proxy Factor Portfolio Weights（代理因子组合权重热图）

论文含义：
  论文把统计 PCA 因子用"少量股票的稀疏组合"去近似（proxy factors），考察这些代理
  组合的权重在股票层面的分布。热图行是因子、列是股票，颜色表示该股票在该因子代理
  组合里的权重。直观地展示因子的经济含义（哪些股票主导了某个因子）。

数据来源：
  result.proxy_weights（列：factor, rank, symbol, weight）。

数据处理 vs 图表输出：
  - 数据处理：把长表 pivot 成 因子 × 股票 的权重矩阵。
  - 图表输出：imshow 热图。
"""

from __future__ import annotations

# --- 允许 `python figCode/xxx.py` / `python tableCode/xxx.py` 直接运行时找到 prepareCore 包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

from prepareCore.io_utils import _save_heatmap, _save_placeholder_figure, _atomic_save_figure
from prepareCore.logging_utils import log_render

TAG = "figure_weights"


def plot_weight_heatmap(df, title: str, output_path: Path) -> None:
    """复刻 engine 内 weight_heatmap：长表 -> 因子×股票矩阵 -> 热图（按 symbol 排，回退用）。"""
    if df.empty:
        _save_placeholder_figure(output_path, title, "No portfolio-weight data are available.")
        return
    pivot = df.pivot_table(index="factor", columns="symbol", values="weight", aggfunc="first").fillna(0.0)
    _save_heatmap(
        pivot.to_numpy(),
        pivot.columns.tolist(),
        [f"Factor {idx}" for idx in pivot.index],
        title,
        output_path,
    )


def _load_industry_lookup(cfg) -> dict:
    """读取行业映射（新版 11 桶或旧版），返回 {ts_code: std_industry}。失败返回空 dict。"""
    import os as _os
    import pandas as pd
    fname = _os.environ.get("PELGER_INDUSTRY_INFO_FILENAME") or getattr(cfg, "industry_info_filename", "stock_full_info_std_industry_final.csv")
    path = Path(cfg.external_data_root) / "industry" / fname
    try:
        d = pd.read_csv(path)
        code_col = "ts_code" if "ts_code" in d.columns else d.columns[0]
        ind_col = "std_industry" if "std_industry" in d.columns else d.columns[-1]
        return dict(zip(d[code_col].astype(str).str.strip(), d[ind_col].astype(str).str.strip()))
    except Exception:
        return {}


_INDUSTRY_EN_LABELS = {
    "必需消费": "Staples",
    "大金融": "Financials",
    "电力设备与新能源": "Power/NewEn",
    "房地产与建筑": "RealEstate/Build",
    "高端制造": "AdvMfg",
    "公用事业与交运": "Utilities/Trans",
    "科技成长": "TechGrowth",
    "可选消费与服务": "Discretionary",
    "农林牧渔": "Agriculture",
    "医药生物": "Healthcare",
    "周期资源": "CyclicalRes",
}


def _industry_display_label(name: str, label_index: dict[str, int]) -> str:
    raw = str(name)
    if raw in _INDUSTRY_EN_LABELS:
        return _INDUSTRY_EN_LABELS[raw]
    try:
        raw.encode("ascii")
        return raw[:18]
    except UnicodeEncodeError:
        return f"IND{label_index.get(raw, 0):02d}"


def plot_industry_sorted_bars(W, tickers, industry_lookup: dict, k_count: int, title: str, output_path: Path) -> None:
    """1:1 复刻论文 Figure 3/4/5：每个因子一个子图（竖排），x = 全部股票（按行业排序），
    y = "Loadings"（组合权重），柱状图、**按行业着色**，右侧给行业色例。
    industry_lookup 为空时退化为单一颜色、按代码排序。"""
    import matplotlib.pyplot as plt
    import numpy as np

    W = np.asarray(W, dtype=float)
    k_count = max(1, min(int(k_count), W.shape[1]))
    tickers = [str(t) for t in tickers]
    inds = [industry_lookup.get(t, "Other") for t in tickers]
    if industry_lookup:
        order = sorted(range(len(tickers)), key=lambda i: (inds[i], tickers[i]))
    else:
        order = sorted(range(len(tickers)), key=lambda i: tickers[i])
    ind_sorted = [inds[i] for i in order]

    # 行业 -> 颜色（出现顺序映射到 tab20）
    uniq = list(dict.fromkeys(ind_sorted))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    color_of = {name: cmap(i % cmap.N) for i, name in enumerate(uniq)}
    bar_colors = [color_of[ind_sorted[j]] for j in range(len(order))]
    x = np.arange(len(order))

    fig, axes = plt.subplots(k_count, 1, figsize=(13.0, 2.6 * k_count + 0.8), sharex=True)
    if k_count == 1:
        axes = [axes]
    for k in range(k_count):
        ax = axes[k]
        heights = W[order, k]
        ax.bar(
            x,
            heights,
            width=0.86,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_ylabel("Loadings")
        ax.set_title(f"Factor {k + 1}", fontsize=10)
        # 行业分隔线
        if industry_lookup:
            start = 0
            for i in range(1, len(ind_sorted) + 1):
                if i == len(ind_sorted) or ind_sorted[i] != ind_sorted[start]:
                    if start > 0:
                        ax.axvline(start - 0.5, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.9)
                    start = i
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_xticks([])
    axes[-1].set_xlabel("Stocks sorted by industry")
    # 行业色例（图右侧）
    if industry_lookup:
        from matplotlib.patches import Patch
        label_index = {name: idx + 1 for idx, name in enumerate(uniq)}
        handles = [Patch(facecolor=color_of[name], label=_industry_display_label(name, label_index)) for name in uniq]
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.835, 0.5),
            fontsize=7,
            frameon=False,
            title="Industry",
            title_fontsize=8,
        )
    fig.tight_layout(rect=(0, 0, 0.82 if industry_lookup else 1.0, 0.985))
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)


def plot_industry_sorted_heatmap(W, tickers, industry_lookup: dict, k_count: int, title: str, output_path: Path) -> None:
    """1:1 复刻论文 Figure 3/4：因子 × 全部股票的权重热图，股票按行业排序，
    行业之间画分隔线、下方标注行业名。industry_lookup 为空时回退为按代码排序。"""
    import matplotlib.pyplot as plt
    import numpy as np

    W = np.asarray(W, dtype=float)
    k_count = max(1, min(int(k_count), W.shape[1]))
    tickers = [str(t) for t in tickers]
    # 列顺序：先按行业（无映射则归到 "Other"），同行业内按代码
    inds = [industry_lookup.get(t, "Other") for t in tickers]
    order = sorted(range(len(tickers)), key=lambda i: (inds[i], tickers[i]))
    if not industry_lookup:  # 回退：纯代码序
        order = sorted(range(len(tickers)), key=lambda i: tickers[i])
    mat = W[order, :k_count].T  # (k_count × N)
    ind_sorted = [inds[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(11, len(tickers) * 0.045), 4.6))
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    image = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(k_count))
    ax.set_yticklabels([f"Factor {i + 1}" for i in range(k_count)])
    ax.set_xticks([])  # 股票太多，不逐个标代码；改在行业块中心标行业名

    # 行业分隔线 + 行业名（块中心）
    if industry_lookup:
        boundaries = []
        labels = []
        start = 0
        for i in range(1, len(ind_sorted) + 1):
            if i == len(ind_sorted) or ind_sorted[i] != ind_sorted[start]:
                if start > 0:
                    ax.axvline(start - 0.5, color="0.25", linewidth=0.7)
                boundaries.append((start + i - 1) / 2.0)
                labels.append(ind_sorted[start])
                start = i
        ax.set_xticks(boundaries)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_xlabel("Stocks sorted by industry")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="portfolio weight")
    fig.tight_layout()
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)


def _full_weight_heatmap(result, cfg, *, proxy: bool, title: str, output_path: Path) -> None:
    """从 pipeline 重算【全部股票】的权重向量（已含 P4 符号定向），按行业排序画热图。
    任何异常都回退到旧的 top-N、按代码排序的热图，保证出图不中断。"""
    try:
        import prepareCore.engine as eng
        import numpy as np
        pipe = result.pipeline
        panel = result.panel
        disp = getattr(pipe, "pca_cont_display", None) or pipe.pca_cont
        Lam = np.asarray(disp.Lambda, dtype=float)
        N = Lam.shape[0]
        V = Lam / np.sqrt(max(N, 1))                 # 特征向量尺度（论文纵轴量级 ~0.05–0.2）
        if proxy:
            # 用 factor_portfolio_weights 经 build_proxy_factors 得稀疏掩码，施加到 V。
            Wp, _ = eng.build_proxy_factors(eng.factor_portfolio_weights(disp), pipe.R_cont)
            mask = (np.asarray(Wp, dtype=float) != 0.0)
            W = V * mask[:, : V.shape[1]]
        else:
            W = V
        k_count = min(W.shape[1], 4)
        lookup = _load_industry_lookup(cfg)
        plot_industry_sorted_bars(W, panel.tickers, lookup, k_count, title, output_path)
    except Exception as exc:  # 回退
        log_render(TAG, f"行业排序柱状图回退到旧实现: {exc!r}")
        df = result.proxy_weights if proxy else result.pca_weights
        plot_weight_heatmap(df, title, output_path)
