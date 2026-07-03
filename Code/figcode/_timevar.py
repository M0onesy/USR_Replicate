"""figCode/_timevar.py
=====================================================================
论文 Figure 6–11（时间变化类）的【自洽】滚动计算 + 论文形态绘图。

设计：不依赖既有重结果缓存里存了什么，而是从 result.pipeline.R_cont
（连续 5 分钟收益，形状 (M_total, N)）+ result.panel.day_ids（每个 5min
行对应的交易日序号）+ result.panel.R_daily 直接重算 21 个交易日的局部窗口，
得到论文需要的全部量。所有函数都被各 figure_0N.py 以 try/except 包裹，
失败则回退到旧的简单绘图，互不影响。

论文口径：
  * Fig 6：局部 vs 全样本【载荷】的广义相关(GC)随时间，6 个面板
    （连续4 / HF4 / 行业4 / FFC4 / FF3 / 市场1）。
  * Fig 7：局部 vs 全样本【权重】GC，前 7 个连续 PCA 因子，单面板。
  * Fig 8：两个指定月份窗口的 4 因子【权重】行业着色柱状（即 Fig4 形态 ×2 列）。
  * Fig 9：解释方差随时间。
  * Fig 10：连续 PCA 的 [系统性影响 Λ'Λ/N·σ² | 平均载荷 Λ'Λ/N | 波动 σ²]，
    左原始 / 右按时间均值归一化，每图 4 因子线（3 行 × 2 列）。
  * Fig 11：同 Fig 10，但针对 4 个 FFC 因子（载荷来自局部回归）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 基础：窗口与局部 PCA
# ----------------------------------------------------------------------
def _eng():
    import core.engine as eng
    return eng


def _day_windows(n_days: int, window: int = 21, step: int = 5) -> List[Tuple[int, int, int]]:
    """返回 [(d0, d1, center)] 覆盖 [0, n_days) 的交易日窗口（左闭右开）。"""
    out = []
    d0 = 0
    while d0 + window <= n_days:
        out.append((d0, d0 + window, d0 + window // 2))
        d0 += step
    if not out and n_days >= 3:
        out.append((0, n_days, n_days // 2))
    return out


def _rows_for_days(day_ids: np.ndarray, d0: int, d1: int) -> np.ndarray:
    return np.where((day_ids >= d0) & (day_ids < d1))[0]


def _safe_pca(R: np.ndarray, K: int):
    eng = _eng()
    R = np.nan_to_num(np.asarray(R, dtype=float), nan=0.0)
    K = max(1, min(int(K), R.shape[1] - 1, R.shape[0] - 1))
    if hasattr(eng, "_panel_pca"):
        return eng._panel_pca(R, K=K, use_corr=True)
    return eng.pca_factors(R, K=K, use_corr=True)


def local_pca_windows(result: Any, K: int, *, source: str = "cont", window: int = 21, step: int = 5) -> Dict[str, Any]:
    """对每个 21 日窗口做局部连续 PCA，返回 loadings/weights/factor_var/explained + 全样本基准。

    source: "cont" 用 pipe.R_cont（连续 5min）；"hf" 用 panel.R_5min_full（含跳跃的 HF）。
    """
    eng = _eng()
    pipe = result.pipeline
    panel = result.panel
    R = getattr(pipe, "R_cont", None) if source == "cont" else getattr(panel, "R_5min_full", None)
    if R is None:
        raise RuntimeError(f"source={source} 的 5 分钟收益不可用")
    R = np.asarray(R, dtype=float)
    day_ids = np.asarray(panel.day_ids, dtype=np.int64)
    n_days = int(len(panel.dates))
    dates = pd.to_datetime(list(panel.dates))

    # 全样本基准
    g = _safe_pca(R, K)
    eng.orient_pca_result(g)
    Kf = g.Lambda.shape[1]
    glob_load = np.asarray(g.Lambda, dtype=float)
    glob_w = eng.factor_portfolio_weights(g)

    centers, loads, weights, fvars, expl = [], [], [], [], []
    for d0, d1, cc in _day_windows(n_days, window, step):
        rows = _rows_for_days(day_ids, d0, d1)
        if rows.size < Kf + 2:
            continue
        sub = R[rows, :]
        try:
            lp = _safe_pca(sub, Kf)
            eng.orient_pca_result(lp)
        except Exception:
            continue
        Lam = np.asarray(lp.Lambda, dtype=float)
        W = eng.factor_portfolio_weights(lp)
        F = np.asarray(getattr(lp, "F", np.zeros((sub.shape[0], Kf))), dtype=float)
        if F.shape[1] < Kf:
            F = sub @ W  # 退路：用权重投影出因子
        centers.append(dates[min(cc, n_days - 1)])
        loads.append(Lam[:, :Kf])
        weights.append(W[:, :Kf])
        fvars.append(np.nanvar(F[:, :Kf], axis=0))
        ev = np.asarray(getattr(lp, "eigvals", []), dtype=float)
        expl.append(float(np.nansum(ev[:Kf]) / np.nansum(ev)) if ev.size else np.nan)

    return {
        "centers": pd.DatetimeIndex(centers),
        "loadings": loads, "weights": weights, "factor_var": np.array(fvars) if fvars else np.zeros((0, Kf)),
        "explained": np.array(expl), "global_loadings": glob_load, "global_weights": glob_w, "K": Kf,
    }


def _gc_series(local_mats: List[np.ndarray], global_mat: np.ndarray) -> np.ndarray:
    """逐窗口 GC(local, global)，返回 (n_windows × K)。"""
    eng = _eng()
    K = global_mat.shape[1]
    out = np.full((len(local_mats), K), np.nan)
    for i, M in enumerate(local_mats):
        try:
            gc = eng.generalized_correlations(np.asarray(M, dtype=float), global_mat)
            out[i, : len(gc)] = gc[:K]
        except Exception:
            pass
    return out


def _local_regression_loadings(result: Any, F_global_daily: np.ndarray, *, window: int = 21, step: int = 5):
    """对给定的【日频】全样本因子收益 F_global (D×K)，在每个 21 日窗口里用日频股票
    收益对其回归，得到局部载荷 (N×K)，并算各因子在窗口内的方差。返回基准载荷+窗口序列。"""
    panel = result.panel
    Rd = np.asarray(panel.R_daily, dtype=float)
    F = np.asarray(F_global_daily, dtype=float)
    n_days = min(Rd.shape[0], F.shape[0])
    Rd, F = Rd[:n_days], F[:n_days]
    dates = pd.to_datetime(list(panel.dates))[:n_days]
    K = F.shape[1]

    def _ols_loadings(y_win, x_win):
        # y_win: (T×N), x_win: (T×K) -> B: (N×K)
        x = np.column_stack([np.ones(x_win.shape[0]), x_win])  # 含截距
        xtx = x.T @ x
        try:
            beta = np.linalg.solve(xtx, x.T @ np.nan_to_num(y_win, nan=0.0))
        except Exception:
            beta = np.linalg.pinv(x) @ np.nan_to_num(y_win, nan=0.0)
        return beta[1:, :].T  # 去掉截距 -> (N×K)

    glob = _ols_loadings(Rd, F)
    centers, loads, fvars = [], [], []
    for d0, d1, cc in _day_windows(n_days, window, step):
        yw, xw = Rd[d0:d1], F[d0:d1]
        if yw.shape[0] < K + 2:
            continue
        loads.append(_ols_loadings(yw, xw))
        fvars.append(np.nanvar(xw, axis=0))
        centers.append(dates[min(cc, n_days - 1)])
    return {"centers": pd.DatetimeIndex(centers), "loadings": loads,
            "factor_var": np.array(fvars) if fvars else np.zeros((0, K)), "global_loadings": glob, "K": K}


# ----------------------------------------------------------------------
# 取各因子集的【全样本日频因子收益】（供 Fig 6 的行业/FFC/FF3/市场面板）
# ----------------------------------------------------------------------
def _daily_factor_sets(result: Any) -> Dict[str, np.ndarray]:
    eng = _eng()
    pipe = result.pipeline
    panel = result.panel
    payload = getattr(result, "paper_tail", {}) or {}
    out: Dict[str, np.ndarray] = {}
    cont = getattr(pipe, "F_cont_display_daily_total", None)
    if cont is not None:
        out["Four Continuous Factors"] = np.asarray(cont, dtype=float)[:, :4]
    # HF（连续+跳跃）4 因子：在 R_5min_full 上做 K=4 PCA 取权重，投影到日频。
    try:
        hf = getattr(pipe, "pca_hf", None)
        if hf is None or int(hf.Lambda.shape[1]) < 4:
            hf = eng.pca_factors(np.asarray(panel.R_5min_full, dtype=float), K=4, use_corr=True)
        else:
            hf = eng._truncate_pca_result(hf, 4)
        eng.orient_pca_result(hf)
        out["Four HF (Continuous+Jump) Factors"] = np.asarray(panel.R_daily, dtype=float) @ eng.factor_portfolio_weights(hf)
    except Exception:
        pass
    # 行业 / FFC：从 paper_tail 的长表透视成日频矩阵
    def _wide(df, value_factors):
        if df is None or not hasattr(df, "loc"):
            return None
        d = df.loc[df["segment_kind"].eq("daily")] if "segment_kind" in df.columns else df
        piv = d.pivot_table(index="date", columns="factor", values="ret", aggfunc="first")
        cols = [c for c in value_factors if c in piv.columns]
        return piv[cols].to_numpy(dtype=float) if cols else None
    ind = payload.get("industry_factors")
    if ind is not None:
        try:
            d = ind.loc[ind["segment_kind"].eq("daily")]
            piv = d.pivot_table(index="date", columns="factor", values="ret", aggfunc="first")
            out["Four Industry Factors"] = piv.to_numpy(dtype=float)[:, :4]
        except Exception:
            pass
    ffc = payload.get("ffc_segmented")
    w = _wide(ffc, ["MKT_excess", "SMB", "HML", "MOM"])
    if w is not None:
        out["Four Fama-French-Carhart Factors"] = w[:, :4]
        out["Three Fama-French Factors"] = w[:, :3]
        out["Market Factor"] = w[:, :1]
    return out


# ----------------------------------------------------------------------
# 绘图：Figure 6 / 7 / 8 / 9 / 10 / 11
# ----------------------------------------------------------------------
def _save(fig, output_path: Path):
    from core.io_utils import _atomic_save_figure
    _atomic_save_figure(fig, output_path, dpi=160)


def render_fig6(result, cfg, output_path: Path, title: str) -> None:
    """Fig 6（论文：保持权重不变、让载荷随时间变化）：对每个因子模型，用其【固定的
    全样本日频因子】回归出局部载荷，再算 局部 vs 全样本 载荷的 GC。6 个面板。"""
    import matplotlib.pyplot as plt
    dfs = _daily_factor_sets(result)
    order = ["Four Continuous Factors", "Four HF (Continuous+Jump) Factors", "Four Industry Factors",
             "Four Fama-French-Carhart Factors", "Three Fama-French Factors", "Market Factor"]
    panels = []
    for name in order:
        F = dfs.get(name)
        if F is None:
            continue
        try:
            r = _local_regression_loadings(result, F)
            panels.append((name, r["centers"], r["loadings"], r["global_loadings"]))
        except Exception:
            pass
    if not panels:
        raise RuntimeError("Fig6 无可用面板")
    nrow = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(14, 3.2 * nrow), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, (name, centers, loads, glob) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        gc = _gc_series(loads, glob)
        for k in range(gc.shape[1]):
            ax.plot(centers, gc[:, k], linewidth=1.2, label=f"{k+1}{['st','nd','rd','th'][min(k,3)]} GC")
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("Generalized Correlation"); ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2); ax.legend(loc="lower left", fontsize=6, ncol=2)
    for j in range(len(panels), nrow * 2):
        axes[j // 2, j % 2].axis("off")
    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, output_path); plt.close(fig)


def render_fig7(result, cfg, output_path: Path, title: str) -> None:
    """Fig 7：前 7 个连续 PCA 因子的【权重】GC，单面板。"""
    import matplotlib.pyplot as plt
    r = local_pca_windows(result, 7, source="cont")
    gc = _gc_series(r["weights"], r["global_weights"])
    fig, ax = plt.subplots(figsize=(11, 5.4))
    suff = ["st", "nd", "rd", "th", "th", "th", "th"]
    for k in range(gc.shape[1]):
        ax.plot(r["centers"], gc[:, k], linewidth=1.2, label=f"{k+1}{suff[min(k,6)]} GC")
    ax.set_xlabel("Time"); ax.set_ylabel("Generalized Correlation")
    ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.2); ax.legend(loc="center right", fontsize=8)
    fig.tight_layout(); _save(fig, output_path); plt.close(fig)


def render_fig8(result, cfg, output_path: Path, title: str, windows: Optional[Sequence[str]] = None) -> None:
    """Fig 8：两个窗口的 4 因子权重行业着色柱状（4 行因子 × 2 列窗口）。
    windows 给两个日期（窗口中心）；缺省自动取“低波动”和“高波动”两个窗口。"""
    import matplotlib.pyplot as plt
    from figCode._weights import _load_industry_lookup
    r = local_pca_windows(result, 4, source="cont", step=1)
    centers = r["centers"]
    if len(r["weights"]) < 2:
        raise RuntimeError("Fig8 窗口不足")
    # 选两个窗口：默认 = 因子方差和最小、最大的两个窗口（近似“平静月/危机月”）
    if windows and len(windows) >= 2:
        idxs = []
        for w in windows[:2]:
            try:
                idxs.append(int(np.argmin(np.abs(centers - pd.Timestamp(w)))))
            except Exception:
                idxs.append(0)
    else:
        tot_var = np.nansum(r["factor_var"], axis=1) if r["factor_var"].size else np.zeros(len(centers))
        idxs = [int(np.nanargmin(tot_var)), int(np.nanargmax(tot_var))]
    tickers = [str(t) for t in result.panel.tickers]
    lookup = _load_industry_lookup(cfg)
    inds = [lookup.get(t, "Other") for t in tickers]
    order = sorted(range(len(tickers)), key=lambda i: (inds[i], tickers[i])) if lookup else list(range(len(tickers)))
    ind_sorted = [inds[i] for i in order]
    uniq = list(dict.fromkeys(ind_sorted))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    color_of = {n: cmap(i % cmap.N) for i, n in enumerate(uniq)}
    bar_colors = [color_of[ind_sorted[j]] for j in range(len(order))]
    x = np.arange(len(order))

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=True)
    sqrtN = np.sqrt(max(len(tickers), 1))
    for col, wi in enumerate(idxs):
        Lam = np.asarray(r["loadings"][wi], dtype=float)
        W = Lam / sqrtN  # 特征向量尺度（与 Fig 4 一致）
        when = pd.Timestamp(centers[wi]).strftime("%Y-%m")
        for k in range(min(4, W.shape[1])):
            ax = axes[k, col]
            ax.bar(x, W[order, k], width=1.0, color=bar_colors, linewidth=0)
            ax.axhline(0.0, color="0.4", linewidth=0.7)
            ax.set_ylabel("Loadings"); ax.set_xticks([])
            ax.set_title(f"Factor {k+1}" + (f"  ({when})" if k == 0 else ""), fontsize=9)
    from matplotlib.patches import Patch
    if lookup:
        fig.legend(handles=[Patch(facecolor=color_of[n], label=n) for n in uniq],
                   loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.88 if lookup else 1.0, 0.985))
    _save(fig, output_path); plt.close(fig)


def render_fig9(result, cfg, output_path: Path, title: str) -> None:
    """Fig 9：解释方差随时间（连续 PCA 局部窗口）。"""
    import matplotlib.pyplot as plt
    r = local_pca_windows(result, 4, source="cont")
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(r["centers"], r["explained"], linewidth=1.4, color="#1f5fbf")
    ax.set_xlabel("Time"); ax.set_ylabel("Explained variation")
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); _save(fig, output_path); plt.close(fig)


def _decomp_panels(centers, loadings_list, factor_var, K):
    """由载荷序列与因子方差，算 [系统性影响, 平均载荷, 波动] 三量（n_windows × K）。"""
    n = len(loadings_list)
    avg_load = np.full((n, K), np.nan)
    for i, Lam in enumerate(loadings_list):
        Lam = np.asarray(Lam, dtype=float)
        N = Lam.shape[0]
        avg_load[i, :Lam.shape[1]] = np.nansum(Lam ** 2, axis=0) / max(N, 1)  # Λ'Λ/N
    vol = np.asarray(factor_var, dtype=float)
    m = min(avg_load.shape[0], vol.shape[0])
    avg_load, vol = avg_load[:m], vol[:m, :K]
    sys_imp = avg_load * vol
    return sys_imp, avg_load, vol


def _render_decomp(centers, sys_imp, avg_load, vol, output_path: Path, title: str):
    import matplotlib.pyplot as plt
    rows = [("Average loadings and volatility", sys_imp, "Variance times loadings"),
            ("Average loadings", avg_load, "Loadings"),
            ("Volatility", vol, "Volatility")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    K = sys_imp.shape[1]
    m = min(len(centers), sys_imp.shape[0])
    cc = centers[:m]
    for r_idx, (name, mat, ylab) in enumerate(rows):
        mat = mat[:m]
        tmean = np.nanmean(mat, axis=0, keepdims=True)
        norm = mat / np.where(np.abs(tmean) > 0, tmean, np.nan)
        for col, (M, suffix) in enumerate([(mat, ""), (norm, "Normalized ")]):
            ax = axes[r_idx, col]
            for k in range(K):
                ax.plot(cc, M[:, k], linewidth=1.1, label=f"{k+1}. Factor")
            ax.set_title(f"{suffix}{name.lower()}" if col else name, fontsize=9)
            ax.set_ylabel(("Normalized " if col else "") + ylab)
            ax.grid(True, alpha=0.2)
            if r_idx == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("Time"); axes[-1, 1].set_xlabel("Time")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, output_path); plt.close(fig)


def render_fig10(result, cfg, output_path: Path, title: str) -> None:
    """Fig 10（论文：固定权重、载荷随时间变化）：对【连续 PCA 因子】用固定全样本日频
    因子回归出局部载荷 + 因子方差，做 [系统性影响/平均载荷/波动]×[原始/归一化] 分解。"""
    dfs = _daily_factor_sets(result)
    F = dfs.get("Four Continuous Factors")
    if F is None:
        raise RuntimeError("Fig10 需要连续 PCA 日频因子")
    r = _local_regression_loadings(result, F)
    sys_imp, avg_load, vol = _decomp_panels(r["centers"], r["loadings"], r["factor_var"], r["K"])
    _render_decomp(r["centers"], sys_imp, avg_load, vol, output_path, title)


def render_fig11(result, cfg, output_path: Path, title: str) -> None:
    """Fig 11：FFC 因子结构分解（局部回归载荷 + FFC 因子方差）。"""
    dfs = _daily_factor_sets(result)
    F = dfs.get("Four Fama-French-Carhart Factors")
    if F is None:
        raise RuntimeError("Fig11 需要分段 FFC 日频因子")
    r = _local_regression_loadings(result, F)
    sys_imp, avg_load, vol = _decomp_panels(r["centers"], r["loadings"], r["factor_var"], r["K"])
    _render_decomp(r["centers"], sys_imp, avg_load, vol, output_path, title)
