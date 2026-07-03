"""core/paper_fidelity.py
================================================================
论文保真补丁模块（P6 / P7 / P9）—— 全市场口径的特征因子构造。

为什么单独成模块：
  * 这些是“尾部/视图”层的重写，复用既有重结果缓存即可生效（分钟级），
    不会让 replication_result_*.pkl 失效。
  * 与 paper_tail.py 解耦、便于审阅；paper_tail._build_payload 通过 env 开关
    调用本模块，并对异常回退到旧实现（保证 refresh 不中断）。

实现的论文口径：
  * P9（附录 A）：6 个 size/value 组合用【全部满足清洗条件的股票】构造，
    而非平衡子集；vw 用 June-end float_mv_adj、ew 等权；盘中/隔夜/日频三段。
  * P6（III.A）：自建 Carhart 12-1 月动量（过去 t-12..t-2 月累计收益，跳过
    t-1 月），月度再平衡，winner(top 30%) - loser(bottom 30%)，并给出
    盘中/隔夜/日频三段（高频版动量，隔夜 ≠ 0）。
  * P7（式 7-9 + III.A）：分段 FFC = 股票级直接构造的 MKT/SMB/HML/MOM，
    rf 按时长拆分，daily = intra + night（不做残差强制对齐）。

数据契约（与仓库一致）：
  * symbol_returns/<code>.npz 含 date_codes(int YYYYMMDD)、intraday_returns、
    overnight_returns、daily_returns。
  * assignments（_load_assignments 输出）含 code, sort_year, portfolio
    (SL/SM/SH/BL/BM/BH), float_mv_adj。
  * mcap_matrix: (T=len(global_dates), N=len(symbols)) 流通市值矩阵。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEGMENT_ORDER = ["intraday", "overnight", "daily"]
PORTFOLIO_ORDER = ["SL", "SM", "SH", "BL", "BM", "BH"]
SMALL_PORTFOLIOS = ["SL", "SM", "SH"]
BIG_PORTFOLIOS = ["BL", "BM", "BH"]
HIGH_PORTFOLIOS = ["SH", "BH"]
LOW_PORTFOLIOS = ["SL", "BL"]


# ----------------------------------------------------------------------
# 公共：遍历全市场 symbol_returns，按“每日所属组合”累计三段组合收益
# ----------------------------------------------------------------------
def _date_index_map(global_dates: pd.DatetimeIndex) -> Dict[int, int]:
    return {int(ts.strftime("%Y%m%d")): idx for idx, ts in enumerate(global_dates)}


def _active_sort_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """A 股 2×3 持有规则：7-12 月用当年 sort_year，1-6 月用上一年 sort_year。"""
    yr = dates.year.to_numpy()
    mo = dates.month.to_numpy()
    return np.where(mo >= 7, yr, yr - 1)


def _accumulate_sorted_portfolios(
    proc_root: Path,
    *,
    global_dates: pd.DatetimeIndex,
    mcap_matrix: np.ndarray,
    symbols: Sequence[str],
    # assignment_lookup: period_key -> {symbol: (portfolio_name, weight_or_None)}
    assignment_lookup: Mapping[int, Mapping[str, Tuple[str, Optional[float]]]],
    period_of_date: np.ndarray,  # 每个 global_date 对应的 period_key（如 active_sort_year / 月度标签）
    portfolios: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """对给定的“逐期组合分配”，累计每个组合的 vw/ew 三段日序列。

    返回 long df：列 date, portfolio, segment_kind, weighting, ret。
    """
    symbol_index = {s: i for i, s in enumerate(symbols)}
    date_index = _date_index_map(global_dates)
    T = len(global_dates)
    P = len(portfolios)
    pidx = {p: i for i, p in enumerate(portfolios)}

    sum_eq = {seg: np.zeros((T, P)) for seg in SEGMENT_ORDER}
    cnt_eq = {seg: np.zeros((T, P)) for seg in SEGMENT_ORDER}
    sum_vw = {seg: np.zeros((T, P)) for seg in SEGMENT_ORDER}
    wgt_vw = {seg: np.zeros((T, P)) for seg in SEGMENT_ORDER}

    for path in sorted((proc_root / "symbol_returns").glob("*.npz")):
        symbol = path.stem
        arrays = np.load(path)
        raw_codes = arrays["date_codes"].astype(np.int64, copy=False)
        row_idx = np.array([date_index.get(int(c), -1) for c in raw_codes], dtype=np.int64)
        ok = row_idx >= 0
        if not ok.any():
            continue
        row_idx = row_idx[ok]
        seg_vals = {
            "intraday": arrays["intraday_returns"][ok].astype(float),
            "overnight": arrays["overnight_returns"][ok].astype(float),
            "daily": arrays["daily_returns"][ok].astype(float),
        }
        periods = period_of_date[row_idx]
        mcap_col = symbol_index.get(symbol)

        for period in np.unique(periods):
            assign = assignment_lookup.get(int(period))
            if not assign:
                continue
            pw = assign.get(symbol)
            if pw is None:
                continue
            portfolio, weight = pw
            col = pidx.get(portfolio)
            if col is None:
                continue
            sel = periods == period
            rows = row_idx[sel]
            # 权重：优先用 assignment 给的 June-end float_mv_adj（固定）；否则用 mcap_matrix。
            if weight is not None and np.isfinite(weight) and weight > 0:
                w_rows = np.full(rows.shape, float(weight))
            elif mcap_col is not None:
                w_rows = mcap_matrix[rows, mcap_col]
            else:
                w_rows = np.full(rows.shape, np.nan)
            for seg in SEGMENT_ORDER:
                vals = seg_vals[seg][sel]
                finite = np.isfinite(vals)
                if finite.any():
                    np.add.at(sum_eq[seg][:, col], rows[finite], vals[finite])
                    np.add.at(cnt_eq[seg][:, col], rows[finite], 1.0)
                fw = finite & np.isfinite(w_rows) & (w_rows > 0)
                if fw.any():
                    np.add.at(sum_vw[seg][:, col], rows[fw], vals[fw] * w_rows[fw])
                    np.add.at(wgt_vw[seg][:, col], rows[fw], w_rows[fw])

    rows_out: List[Dict[str, Any]] = []
    for seg in SEGMENT_ORDER:
        eq = np.divide(sum_eq[seg], cnt_eq[seg], out=np.full((T, P), np.nan), where=cnt_eq[seg] > 0)
        vw = np.divide(sum_vw[seg], wgt_vw[seg], out=np.full((T, P), np.nan), where=wgt_vw[seg] > 0)
        for weighting, mat in (("equal_weighted", eq), ("value_weighted", vw)):
            for p, col in pidx.items():
                series = mat[:, col]
                rows_out.extend(
                    {
                        "date": global_dates[i],
                        "portfolio": p,
                        "segment_kind": seg,
                        "weighting": weighting,
                        "ret": float(series[i]) if np.isfinite(series[i]) else np.nan,
                    }
                    for i in range(T)
                )
    return {"long": pd.DataFrame(rows_out)}


# ----------------------------------------------------------------------
# P9：全市场 2×3 size/value
# ----------------------------------------------------------------------
def build_full_market_size_value(
    proc_root: Path,
    assignments: pd.DataFrame,
    *,
    global_dates: pd.DatetimeIndex,
    mcap_matrix: np.ndarray,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """P9：从全市场 symbol_returns + assignments 构造 6 个 size/value 组合。

    返回 long df（date, portfolio, segment_kind, weighting, ret），与旧
    _build_size_value_assets 的 size_value_assets 同 schema，供下游直接消费。
    """
    assign_lookup: Dict[int, Dict[str, Tuple[str, Optional[float]]]] = {}
    for sort_year, grp in assignments.groupby("sort_year"):
        d: Dict[str, Tuple[str, Optional[float]]] = {}
        for _, r in grp.iterrows():
            d[str(r["code"])] = (str(r["portfolio"]), float(r["float_mv_adj"]) if pd.notna(r["float_mv_adj"]) else None)
        assign_lookup[int(sort_year)] = d
    period_of_date = _active_sort_year(global_dates)
    out = _accumulate_sorted_portfolios(
        proc_root,
        global_dates=global_dates,
        mcap_matrix=mcap_matrix,
        symbols=symbols,
        assignment_lookup=assign_lookup,
        period_of_date=period_of_date,
        portfolios=PORTFOLIO_ORDER,
    )
    df = out["long"].sort_values(["date", "portfolio", "segment_kind", "weighting"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# P6：自建 Carhart 12-1 月动量（全市场，月度再平衡，三段）
# ----------------------------------------------------------------------
def build_full_market_momentum(
    proc_root: Path,
    *,
    global_dates: pd.DatetimeIndex,
    mcap_matrix: np.ndarray,
    symbols: Sequence[str],
    lookback_days: int = 252,
    skip_days: int = 21,
    quantile: float = 0.30,
    value_weighted: bool = True,
) -> pd.DataFrame:
    """P6/D4：Carhart 风格高频动量。

    步骤：
      1) 用 daily_returns 拼出全市场 (T×N) 日对数收益矩阵。
      2) 月度再平衡日，对每只有足够历史的股票计算过去 [t-lookback, t-skip]
         的累计对数收益作为动量信号（跳过最近 skip≈1 个月）。
      3) 截面按 quantile 选 winner(top)/loser(bottom)，持有至下个再平衡日。
      4) 用 _accumulate_sorted_portfolios 累计 winner/loser 三段组合收益，
         MOM_segment = winner_segment - loser_segment。
    返回 long df：date, factor("MOM"), segment_kind, ret。
    """
    symbol_index = {s: i for i, s in enumerate(symbols)}
    date_index = _date_index_map(global_dates)
    T = len(global_dates)
    N = len(symbols)

    # 1) 全市场日收益矩阵（缺失留 NaN）
    R = np.full((T, N), np.nan, dtype=float)
    for path in sorted((proc_root / "symbol_returns").glob("*.npz")):
        sym = path.stem
        col = symbol_index.get(sym)
        if col is None:
            continue
        arrays = np.load(path)
        raw_codes = arrays["date_codes"].astype(np.int64, copy=False)
        ridx = np.array([date_index.get(int(c), -1) for c in raw_codes], dtype=np.int64)
        ok = ridx >= 0
        R[ridx[ok], col] = arrays["daily_returns"][ok].astype(float)

    # 2) 月度再平衡日 = 每个自然月的最后一个交易日
    gd = pd.DatetimeIndex(global_dates)
    month_key = gd.year * 100 + gd.month
    rebal_pos = []
    for i in range(T):
        if i == T - 1 or month_key[i] != month_key[i + 1]:
            rebal_pos.append(i)
    # winner/loser 月度分配：period_key = 该月序号；逐日映射到所属持有月
    period_of_date = np.zeros(T, dtype=np.int64)
    assign_lookup: Dict[int, Dict[str, Tuple[str, Optional[float]]]] = {}
    cum = np.nancumsum(np.nan_to_num(R, nan=0.0), axis=0)  # 累计对数收益（缺失按0增量）
    obs = np.cumsum(np.isfinite(R).astype(int), axis=0)    # 累计有效观测数（判断历史是否足够）

    period_id = 0
    for k, rb in enumerate(rebal_pos):
        # 形成期累计收益： cum[rb-skip] - cum[rb-skip-lookback]
        hi = rb - skip_days
        lo = rb - skip_days - lookback_days
        if hi <= 0 or lo < 0:
            # 历史不足，该期不分配（持有期内组合为空）
            hold_start = rb + 1
            hold_end = rebal_pos[k + 1] if k + 1 < len(rebal_pos) else T - 1
            period_of_date[hold_start:hold_end + 1] = period_id
            period_id += 1
            continue
        signal = cum[hi] - cum[lo]
        enough = (obs[hi] - obs[max(lo - 1, 0)]) >= int(0.6 * lookback_days)
        valid = np.isfinite(signal) & enough
        if valid.sum() >= 10:
            s = signal.copy()
            s[~valid] = np.nan
            lo_thr = np.nanquantile(s, quantile)
            hi_thr = np.nanquantile(s, 1.0 - quantile)
            d: Dict[str, Tuple[str, Optional[float]]] = {}
            for col in np.where(valid)[0]:
                name = symbols[col]
                if signal[col] >= hi_thr:
                    d[name] = ("winner", None)
                elif signal[col] <= lo_thr:
                    d[name] = ("loser", None)
            assign_lookup[period_id] = d
        # 持有期 = (本再平衡日+1) 到 下个再平衡日
        hold_start = rb + 1
        hold_end = rebal_pos[k + 1] if k + 1 < len(rebal_pos) else T - 1
        period_of_date[hold_start:hold_end + 1] = period_id
        period_id += 1

    out = _accumulate_sorted_portfolios(
        proc_root,
        global_dates=global_dates,
        mcap_matrix=mcap_matrix if value_weighted else np.full_like(mcap_matrix, np.nan),
        symbols=symbols,
        assignment_lookup=assign_lookup,
        period_of_date=period_of_date,
        portfolios=["winner", "loser"],
    )
    wl = out["long"]
    weighting = "value_weighted" if value_weighted else "equal_weighted"
    wl = wl.loc[wl["weighting"].eq(weighting)]
    wide = wl.pivot_table(index=["date", "segment_kind"], columns="portfolio", values="ret").reset_index()
    wide["ret"] = wide.get("winner") - wide.get("loser")
    wide["factor"] = "MOM"
    return wide[["date", "factor", "segment_kind", "ret"]].sort_values(["date", "segment_kind"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# P7：股票级分段 FFC（无残差强制对齐）
# ----------------------------------------------------------------------
def build_ffc_segmented_clean(
    market_returns_df: pd.DataFrame,
    size_value_long: pd.DataFrame,
    mom_segmented_long: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    rf_split: Mapping[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """P7：股票级直接构造分段 MKT/SMB/HML/MOM。

    MKT_seg = vw 全市场该段 - rf_seg；
    SMB_seg = mean(small) - mean(big)（vw）；
    HML_seg = 0.5(SH+BH) - 0.5(SL+BL)（vw）；
    MOM_seg = winner - loser（来自 build_full_market_momentum）；
    daily = intraday + overnight（不做残差强制对齐）。

    返回 (raw_frame, final_frame)，schema：date, factor, segment_kind, ret。
    raw_frame 与 final_frame 在本实现里一致（没有“强制 daily=官方”这一步），
    保留两个返回值是为兼容旧调用签名。
    """
    market_vw = (
        market_returns_df.loc[market_returns_df["weighting"].eq("value_weighted"), ["date", "segment_kind", "ret"]]
        .pivot(index="date", columns="segment_kind", values="ret")
        .reindex(dates)
        .reindex(columns=SEGMENT_ORDER)
    )
    vw_sv = size_value_long.loc[size_value_long["weighting"].eq("value_weighted")].copy()
    mom_wide = (
        mom_segmented_long.pivot(index="date", columns="segment_kind", values="ret").reindex(dates).reindex(columns=SEGMENT_ORDER)
    )

    seg_mats: Dict[str, np.ndarray] = {}
    for seg in ("intraday", "overnight"):
        piv = (
            vw_sv.loc[vw_sv["segment_kind"].eq(seg)]
            .pivot(index="date", columns="portfolio", values="ret")
            .reindex(dates)
            .reindex(columns=PORTFOLIO_ORDER)
        )
        smb = piv[SMALL_PORTFOLIOS].mean(axis=1) - piv[BIG_PORTFOLIOS].mean(axis=1)
        hml = 0.5 * (piv["SH"] + piv["BH"]) - 0.5 * (piv["SL"] + piv["BL"])
        rf_seg = np.asarray(rf_split[seg], dtype=float)
        mkt = market_vw[seg].to_numpy(dtype=float) - rf_seg  # P8：盘中/隔夜各减自身 rf
        mom = mom_wide[seg].to_numpy(dtype=float)
        seg_mats[seg] = np.column_stack([mkt, smb.to_numpy(float), hml.to_numpy(float), mom])

    daily_mat = seg_mats["intraday"] + seg_mats["overnight"]  # daily = intra + night（无强制）
    final = {
        "intraday": seg_mats["intraday"],
        "overnight": seg_mats["overnight"],
        "daily": daily_mat,
    }
    rows: List[Dict[str, Any]] = []
    factor_names = ["MKT_excess", "SMB", "HML", "MOM"]
    for seg, mat in final.items():
        for j, fac in enumerate(factor_names):
            col = mat[:, j]
            rows.extend(
                {
                    "date": dates[i],
                    "factor": fac,
                    "segment_kind": seg,
                    "ret": float(col[i]) if np.isfinite(col[i]) else np.nan,
                }
                for i in range(len(dates))
            )
    frame = pd.DataFrame(rows).sort_values(["date", "segment_kind", "factor"]).reset_index(drop=True)
    return frame.copy(), frame.copy()


def split_daily_rf(rf_daily: np.ndarray, intraday_hours: float = 4.0, calendar_hours: float = 24.0) -> Dict[str, np.ndarray]:
    """P8：日内常数假设 -> 把日 rf 按时长拆成盘中/隔夜。

    A 股连续竞价约 4 小时（9:30-11:30, 13:00-15:00）；其余约 20 小时归隔夜。
    论文指出 rf 远小于股票收益，拆分影响很小，但应口径一致。
    """
    rf = np.asarray(rf_daily, dtype=float)
    intra_frac = float(intraday_hours) / float(calendar_hours)
    return {
        "intraday": rf * intra_frac,
        "overnight": rf * (1.0 - intra_frac),
        "daily": rf,
    }
