"""
================================================================================
Pelger (2020) "Understanding Systematic Risk: A High-Frequency Approach"
中国 A 股 5 分钟数据复现脚本
--------------------------------------------------------------------------------
本文件把论文中的通用数学框架改造成了可以直接对接本地
`EXTRA_STOCK_A/<symbol>/data.bz2` 数据的一站式脚本。

当前默认口径:
1. 读取本地中国 A 股 5 分钟 K 线 `.bz2` 数据
2. 扫描样本覆盖度, 识别严格平衡样本和 99% 覆盖样本
3. 构造 `HFPanel`
4. 使用 TOD 阈值法拆分连续/跳跃收益
5. 使用 PCA + 扰动特征值比率提取系统性风险因子
6. 计算日内/隔夜/日度 Sharpe
7. 输出滚动因子空间稳定性结果
8. 对年度 99% 覆盖样本做 pairwise-covariance 稳健性比较

只依赖本地已有 K 线数据即可完成论文主干复现。行业因子、FFC、测试资产
组合等扩展数据接口已经预留，但不会自动下载任何外部数据。

依赖:
    numpy, pandas, scipy
可选绘图:
    matplotlib
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh


REQUIRED_BZ2_COLUMNS = [
    "code",
    "kline_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

CN_5MIN_BAR_TIMES = [
    "09:30:00", "09:35:00", "09:40:00", "09:45:00", "09:50:00", "09:55:00",
    "10:00:00", "10:05:00", "10:10:00", "10:15:00", "10:20:00", "10:25:00",
    "10:30:00", "10:35:00", "10:40:00", "10:45:00", "10:50:00", "10:55:00",
    "11:00:00", "11:05:00", "11:10:00", "11:15:00", "11:20:00", "11:25:00",
    "13:00:00", "13:05:00", "13:10:00", "13:15:00", "13:20:00", "13:25:00",
    "13:30:00", "13:35:00", "13:40:00", "13:45:00", "13:50:00", "13:55:00",
    "14:00:00", "14:05:00", "14:10:00", "14:15:00", "14:20:00", "14:25:00",
    "14:30:00", "14:35:00", "14:40:00", "14:45:00", "14:50:00", "14:55:00",
]
CN_5MIN_BAR_CODES = np.array(
    [int(time_text[:2]) * 100 + int(time_text[3:5]) for time_text in CN_5MIN_BAR_TIMES],
    dtype=np.int32,
)

STRICT_BALANCED_SAMPLE = "strict_balanced"
NEAR_BALANCED_99_SAMPLE = "near_balanced_99"

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "EXTRA_STOCK_A"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "replication_output"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / ".hf_cache" / "pelger_cn"
SCAN_CACHE_VERSION = "v2"
RETURNS_CACHE_VERSION = "v1"
PANEL_CACHE_VERSION = "v2"
SUSPICIOUS_OVERNIGHT_THRESHOLDS = (0.12, 0.20)


def _safe_inv(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """稳定求逆; 对 1x1 或更高维方阵都适用."""
    arr = np.asarray(mat, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    d = arr.shape[0]
    return np.linalg.inv(arr + eps * np.eye(d))


def _sym_eig_desc(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对称矩阵特征分解, 返回降序的 (eigvals, eigvecs)."""
    w, V = eigh((M + M.T) / 2.0)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def _ensure_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _normalize_years(years: Optional[Sequence[int]]) -> Optional[List[int]]:
    if years is None:
        return None
    unique_years = sorted({int(year) for year in years})
    return unique_years or None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_root(cache_root: Optional[str | Path]) -> Path:
    return _ensure_path(cache_root or DEFAULT_CACHE_ROOT)


def _cache_subdir(cache_root: Optional[str | Path], *parts: str) -> Path:
    root = _cache_root(cache_root)
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _symbol_file_signature(file_path: Path) -> Dict[str, Any]:
    stat = file_path.stat()
    return {
        "path": str(file_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _signature_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _returns_cache_key(file_path: Path, return_mode: str) -> str:
    payload = {
        "version": RETURNS_CACHE_VERSION,
        "return_mode": return_mode,
        **_symbol_file_signature(file_path),
    }
    return _signature_hash(payload)


def _scan_cache_paths(cache_root: Optional[str | Path]) -> Tuple[Path, Path, Path, Path]:
    root = _cache_subdir(cache_root, "scan")
    return (
        root / "manifest.json",
        root / "summary.json",
        root / "universe.pkl",
        root / "universe.csv",
    )


def _returns_cache_paths(
    cache_root: Optional[str | Path],
    file_path: Path,
    return_mode: str,
) -> Tuple[Path, Path]:
    root = _cache_subdir(cache_root, "returns", return_mode)
    symbol = _safe_symbol_name(file_path.parent.name)
    key = _returns_cache_key(file_path, return_mode)
    base = root / f"{symbol}_{key}"
    return base.with_suffix(".json"), base.with_suffix(".npz")


def _panel_cache_dir(cache_root: Optional[str | Path]) -> Path:
    return _cache_subdir(cache_root, "panels")


def _selected_sample_signature_hash(selected: pd.DataFrame) -> str:
    records: List[Dict[str, Any]] = []
    if selected.empty:
        return _signature_hash({"symbols": records})

    for row in selected.itertuples(index=False):
        signature = getattr(row, "signature", None)
        if not isinstance(signature, dict):
            signature = _symbol_file_signature(Path(row.file_path))
        records.append({
            "symbol": str(row.symbol),
            "signature": signature,
        })
    return _signature_hash({"symbols": records})


def _panel_cache_key(
    data_root: Path,
    summary: Dict[str, Any],
    sample_mode: str,
    return_mode: str,
    years: Optional[Sequence[int]],
    max_stocks: Optional[int],
    sample_signature_hash: str,
) -> str:
    payload = {
        "version": PANEL_CACHE_VERSION,
        "data_root": str(data_root),
        "sample_mode": sample_mode,
        "return_mode": return_mode,
        "years": _normalize_years(years),
        "max_stocks": None if max_stocks is None else int(max_stocks),
        "sample_signature_hash": sample_signature_hash,
        "global_start": summary["global_start"],
        "global_end": summary["global_end"],
        "total_symbols": summary["total_symbols"],
        "strict_balanced_symbols": summary["strict_balanced_symbols"],
        "near_balanced_99_symbols": summary["near_balanced_99_symbols"],
    }
    return _signature_hash(payload)


def _panel_cache_paths(
    cache_root: Optional[str | Path],
    cache_key: str,
) -> Tuple[Path, Path]:
    root = _panel_cache_dir(cache_root)
    return root / f"{cache_key}.json", root / f"{cache_key}.npz"


def _date_codes_from_dates(dates: Sequence[str]) -> np.ndarray:
    return np.array([int(str(date).replace("-", "")) for date in dates], dtype=np.int32)


def _date_code_to_text(code: int) -> str:
    code_text = f"{int(code):08d}"
    return f"{code_text[:4]}-{code_text[4:6]}-{code_text[6:]}"


def _load_cn_symbol_frame(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """读取单只股票 bz2 数据, 返回清洗后的 DataFrame 和清洗统计."""
    raw = pd.read_pickle(file_path, compression="bz2")
    missing = sorted(set(REQUIRED_BZ2_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"{file_path} 缺少字段: {missing}")

    df = raw.loc[:, REQUIRED_BZ2_COLUMNS].copy()
    raw_rows = int(len(df))
    df["kline_time"] = pd.to_datetime(df["kline_time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_timestamp_rows = int(df["kline_time"].isna().sum())
    bad_numeric_rows = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())

    df = df.dropna(subset=["kline_time", "open", "high", "low", "close"]).copy()
    df.sort_values("kline_time", inplace=True)
    duplicate_rows = int(df.duplicated(subset=["kline_time"], keep="last").sum())
    df = df.drop_duplicates(subset=["kline_time"], keep="last").reset_index(drop=True)

    return df, {
        "raw_rows": raw_rows,
        "bad_timestamp_rows": bad_timestamp_rows,
        "bad_numeric_rows": bad_numeric_rows,
        "duplicate_rows": duplicate_rows,
    }


def _symbol_scan_record(file_path: Path) -> Dict[str, Any]:
    """读取单只股票并生成扫描缓存记录."""
    symbol = file_path.parent.name
    df, clean_stats = _load_cn_symbol_frame(file_path)
    day_info = _classify_symbol_days(df)
    valid_dates = day_info["valid_dates"]
    invalid_dates = day_info["invalid_dates"]
    returns_meta, _ = _build_symbol_returns_cache_payload(
        file_path,
        return_mode="open_close",
        df=df,
    )

    row: Dict[str, Any] = {
        "symbol": symbol,
        "file_path": str(file_path.resolve()),
        "raw_rows": clean_stats["raw_rows"],
        "n_rows": int(len(df)),
        "bad_timestamp_rows": clean_stats["bad_timestamp_rows"],
        "bad_numeric_rows": clean_stats["bad_numeric_rows"],
        "duplicate_rows": clean_stats["duplicate_rows"],
        "n_observed_days": int(sum(day_info["observed_days_by_year"].values())),
        "n_valid_days": int(len(valid_dates)),
        "n_invalid_days": int(len(invalid_dates)),
        "bad_price_days": int(day_info["bad_price_days"]),
        "first_timestamp": df["kline_time"].min().isoformat() if not df.empty else None,
        "last_timestamp": df["kline_time"].max().isoformat() if not df.empty else None,
        "first_valid_date": valid_dates[0] if valid_dates else None,
        "last_valid_date": valid_dates[-1] if valid_dates else None,
        "expected_bars_per_day": len(CN_5MIN_BAR_TIMES),
        "max_abs_overnight": float(returns_meta["max_abs_overnight"]),
        "suspicious_overnight_count_012": int(returns_meta["suspicious_overnight_count_012"]),
        "suspicious_overnight_count_020": int(returns_meta["suspicious_overnight_count_020"]),
        "valid_dates": valid_dates,
        "invalid_dates": invalid_dates,
        "valid_days_by_year": {str(k): int(v) for k, v in day_info["valid_days_by_year"].items()},
        "observed_days_by_year": {str(k): int(v) for k, v in day_info["observed_days_by_year"].items()},
        "bars_by_year": {str(k): int(v) for k, v in day_info["bars_by_year"].items()},
        "signature": _symbol_file_signature(file_path),
    }
    return row


def _classify_symbol_days(
    df: pd.DataFrame,
    expected_times: Sequence[str] = CN_5MIN_BAR_TIMES,
) -> Dict[str, Any]:
    """
    判断每个交易日是否是完整的 A 股 5 分钟网格.
    完整日要求:
    - 恰好 48 根 bar
    - 时间顺序与 `CN_5MIN_BAR_TIMES` 完全一致
    - OHLC 全部严格大于 0
    """
    if df.empty:
        return {
            "valid_dates": [],
            "invalid_dates": [],
            "valid_days_by_year": {},
            "observed_days_by_year": {},
            "bars_by_year": {},
            "bad_price_days": 0,
        }

    valid_dates: List[str] = []
    invalid_dates: List[str] = []
    valid_days_by_year: Dict[int, int] = {}
    observed_days_by_year: Dict[int, int] = {}
    bars_by_year: Dict[int, int] = {}
    bad_price_days = 0

    ts = df["kline_time"]
    dates = ts.to_numpy(dtype="datetime64[D]")
    years = ts.dt.year.to_numpy(dtype=np.int32)
    time_codes = (ts.dt.hour.to_numpy(dtype=np.int16) * 100 + ts.dt.minute.to_numpy(dtype=np.int16)).astype(np.int32)
    price_ok_rows = df[["open", "high", "low", "close"]].gt(0.0).to_numpy().all(axis=1)

    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(df)]
    expected_codes = CN_5MIN_BAR_CODES if list(expected_times) == CN_5MIN_BAR_TIMES else np.asarray(
        [int(time_text[:2]) * 100 + int(time_text[3:5]) for time_text in expected_times],
        dtype=np.int32,
    )

    for start, end in zip(starts, ends):
        year = int(years[start])
        observed_days_by_year[year] = observed_days_by_year.get(year, 0) + 1
        bars_by_year[year] = bars_by_year.get(year, 0) + int(end - start)

        prices_ok = bool(price_ok_rows[start:end].all())
        if not prices_ok:
            bad_price_days += 1

        is_valid_day = (
            (end - start) == len(expected_times)
            and prices_ok
            and np.array_equal(time_codes[start:end], expected_codes)
        )

        date_text = np.datetime_as_string(dates[start], unit="D")
        if is_valid_day:
            valid_dates.append(date_text)
            valid_days_by_year[year] = valid_days_by_year.get(year, 0) + 1
        else:
            invalid_dates.append(date_text)

    return {
        "valid_dates": valid_dates,
        "invalid_dates": invalid_dates,
        "valid_days_by_year": valid_days_by_year,
        "observed_days_by_year": observed_days_by_year,
        "bars_by_year": bars_by_year,
        "bad_price_days": bad_price_days,
    }


def _find_bz2_files(data_root: Path) -> List[Path]:
    files = sorted(data_root.rglob("data.bz2"))
    if not files:
        raise FileNotFoundError(f"在 {data_root} 下没有找到 data.bz2 文件")
    return files


def scan_cn_bz2_universe(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    扫描全部 `.bz2` 数据并生成样本元数据表.

    返回的 DataFrame 按 `symbol` 排序, 并在 `attrs["summary"]` 中附带全样本摘要:
    - 全局交易日历
    - 每年交易日数
    - 严格平衡样本数量
    - 99% 覆盖样本数量
    """
    data_root = _ensure_path(data_root)
    files = _find_bz2_files(data_root)
    file_map = {file_path.parent.name: file_path for file_path in files}

    manifest_path, summary_path, universe_pkl_path, universe_csv_path = _scan_cache_paths(cache_root)
    manifest: Dict[str, Any] = {"version": SCAN_CACHE_VERSION, "symbols": {}}
    cached_universe: Optional[pd.DataFrame] = None
    cached_summary: Optional[Dict[str, Any]] = None

    if (
        use_cache
        and not refresh_cache
        and manifest_path.exists()
        and summary_path.exists()
        and universe_pkl_path.exists()
    ):
        try:
            manifest = _load_json(manifest_path)
            if manifest.get("version") == SCAN_CACHE_VERSION:
                cached_summary = _load_json(summary_path)
                cached_universe = pd.read_pickle(universe_pkl_path)
        except Exception:
            manifest = {"version": SCAN_CACHE_VERSION, "symbols": {}}
            cached_summary = None
            cached_universe = None

    current_signatures = {symbol: _symbol_file_signature(file_path) for symbol, file_path in file_map.items()}
    cached_signatures = manifest.get("symbols", {})

    changed_symbols = sorted(
        symbol
        for symbol, signature in current_signatures.items()
        if cached_signatures.get(symbol) != signature
    )
    stale_symbols = sorted(set(cached_signatures) - set(current_signatures))

    if (
        use_cache
        and not refresh_cache
        and cached_universe is not None
        and cached_summary is not None
        and not changed_symbols
        and not stale_symbols
    ):
        cached_universe = cached_universe.copy()
        cached_universe.attrs["summary"] = cached_summary
        return cached_universe

    base_rows_df = cached_universe.copy() if cached_universe is not None else pd.DataFrame()
    if not base_rows_df.empty:
        base_rows_df = base_rows_df.loc[~base_rows_df["symbol"].isin(set(changed_symbols) | set(stale_symbols))].copy()

    refreshed_rows = []
    for symbol in changed_symbols:
        refreshed_rows.append(_symbol_scan_record(file_map[symbol]))

    refreshed_df = pd.DataFrame(refreshed_rows)
    if base_rows_df.empty:
        universe = refreshed_df
    elif refreshed_df.empty:
        universe = base_rows_df
    else:
        universe = pd.concat([base_rows_df, refreshed_df], ignore_index=True)
    universe = universe.sort_values("symbol").reset_index(drop=True)

    manifest = {
        "version": SCAN_CACHE_VERSION,
        "data_root": str(data_root),
        "symbols": current_signatures,
    }
    rows = universe.to_dict("records")
    global_valid_dates: set[str] = set()
    year_set: set[int] = set()
    for row in rows:
        global_valid_dates.update(str(date) for date in row["valid_dates"])
        year_set.update(int(year) for year in row["observed_days_by_year"])
        year_set.update(int(year) for year in row["valid_days_by_year"])
    years_all = sorted(year_set)
    for year in years_all:
        observed_values = []
        valid_values = []
        bars_values = []
        for row in rows:
            observed_values.append(int(row["observed_days_by_year"].get(str(year), 0)))
            valid_values.append(int(row["valid_days_by_year"].get(str(year), 0)))
            bars_values.append(int(row["bars_by_year"].get(str(year), 0)))
        universe[f"observed_days_{year}"] = observed_values
        universe[f"valid_days_{year}"] = valid_values
        universe[f"bars_{year}"] = bars_values
        for prefix in ("observed_days", "valid_days", "bars"):
            col = f"{prefix}_{year}"
            universe[col] = universe[col].fillna(0).astype(int)

    global_dates = [str(date) for date in sorted(global_valid_dates)]
    if not global_dates:
        raise RuntimeError("未能从数据中识别出任何完整交易日")

    global_start = global_dates[0]
    global_end = global_dates[-1]
    calendar_days = len(global_dates)
    calendar_days_by_year = {
        int(year): int(count)
        for year, count in pd.Series(pd.to_datetime(global_dates).year).value_counts(sort=False).sort_index().items()
    }

    universe["missing_days"] = calendar_days - universe["n_valid_days"]
    universe["coverage_ratio"] = universe["n_valid_days"] / float(calendar_days)
    universe["is_strict_balanced"] = (
        (universe["n_valid_days"] == calendar_days)
        & (universe["n_invalid_days"] == 0)
        & (universe["first_valid_date"] == global_start)
        & (universe["last_valid_date"] == global_end)
    )
    universe["is_near_balanced_99"] = (
        (universe["coverage_ratio"] >= 0.99)
        & (universe["n_invalid_days"] == 0)
    )

    global_dates_by_year = {
        int(year): sorted(date for date in global_dates if pd.Timestamp(date).year == int(year))
        for year in calendar_days_by_year
    }

    for year, year_days in calendar_days_by_year.items():
        valid_col = f"valid_days_{year}"
        observed_col = f"observed_days_{year}"
        universe[f"coverage_{year}"] = universe[valid_col] / float(year_days)
        universe[f"is_strict_{year}"] = (
            (universe[valid_col] == year_days)
            & (universe[observed_col] == year_days)
        )
        universe[f"is_near_99_{year}"] = (
            (universe[f"coverage_{year}"] >= 0.99)
            & (universe[observed_col] == universe[valid_col])
        )

    summary = {
        "data_root": str(data_root),
        "total_symbols": int(len(universe)),
        "global_start": global_start,
        "global_end": global_end,
        "global_calendar_days": calendar_days,
        "bars_per_day": len(CN_5MIN_BAR_TIMES),
        "bar_times": list(CN_5MIN_BAR_TIMES),
        "calendar_days_by_year": calendar_days_by_year,
        "global_dates": global_dates,
        "global_dates_by_year": global_dates_by_year,
        "strict_balanced_symbols": int(universe["is_strict_balanced"].sum()),
        "near_balanced_99_symbols": int(universe["is_near_balanced_99"].sum()),
    }
    universe.attrs["summary"] = summary

    if use_cache:
        _write_json(manifest_path, manifest)
        _write_json(summary_path, summary)
        universe_pkl_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_pickle(universe_pkl_path)
        csv_ready = universe.drop(columns=[col for col in ["valid_dates", "invalid_dates", "valid_days_by_year", "observed_days_by_year", "bars_by_year", "signature"] if col in universe.columns])
        universe_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_ready.to_csv(universe_csv_path, index=False, encoding="utf-8-sig")

    return universe


def _get_universe_summary(
    universe: Optional[pd.DataFrame],
    data_root: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if universe is None or "summary" not in universe.attrs:
        universe = scan_cn_bz2_universe(data_root=data_root)
    return universe, universe.attrs["summary"]


def summarize_cn_universe(universe: pd.DataFrame) -> Dict[str, Any]:
    universe, summary = _get_universe_summary(universe, data_root=DEFAULT_DATA_ROOT)
    stats = dict(summary)
    stats["coverage_ratio_quantiles"] = {
        "p0": float(universe["coverage_ratio"].min()),
        "p25": float(universe["coverage_ratio"].quantile(0.25)),
        "p50": float(universe["coverage_ratio"].quantile(0.50)),
        "p75": float(universe["coverage_ratio"].quantile(0.75)),
        "p95": float(universe["coverage_ratio"].quantile(0.95)),
        "p100": float(universe["coverage_ratio"].max()),
    }
    stats["invalid_day_symbols"] = int((universe["n_invalid_days"] > 0).sum())
    if "suspicious_overnight_count_012" in universe.columns:
        stats["corp_action_risk_summary"] = {
            "symbols_with_suspicious_overnight_012": int((universe["suspicious_overnight_count_012"] > 0).sum()),
            "symbols_with_suspicious_overnight_020": int((universe["suspicious_overnight_count_020"] > 0).sum()),
            "max_abs_overnight_overall": float(universe["max_abs_overnight"].max()),
        }
    return stats


def _years_total_days(summary: Dict[str, Any], years: Optional[Sequence[int]]) -> int:
    years = _normalize_years(years)
    if years is None:
        return int(summary["global_calendar_days"])
    calendar_days_by_year = {int(k): int(v) for k, v in summary["calendar_days_by_year"].items()}
    return int(sum(calendar_days_by_year[year] for year in years))


def _selected_calendar_dates(summary: Dict[str, Any], years: Optional[Sequence[int]]) -> List[str]:
    years = _normalize_years(years)
    if years is None:
        return list(summary["global_dates"])
    global_dates_by_year = {int(k): list(v) for k, v in summary["global_dates_by_year"].items()}
    selected: List[str] = []
    for year in years:
        selected.extend(global_dates_by_year[year])
    return selected


def _select_sample_rows(
    universe: pd.DataFrame,
    sample_mode: str,
    years: Optional[Sequence[int]] = None,
    coverage_threshold: float = 0.99,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    universe, summary = _get_universe_summary(universe, data_root=DEFAULT_DATA_ROOT)
    years = _normalize_years(years)

    if sample_mode not in {STRICT_BALANCED_SAMPLE, NEAR_BALANCED_99_SAMPLE}:
        raise ValueError(f"未知 sample_mode: {sample_mode}")

    if years is None:
        if sample_mode == STRICT_BALANCED_SAMPLE:
            mask = universe["is_strict_balanced"]
        else:
            mask = universe["is_near_balanced_99"]
    else:
        total_days = _years_total_days(summary, years)
        valid_sum = sum(universe[f"valid_days_{year}"] for year in years)
        observed_sum = sum(universe[f"observed_days_{year}"] for year in years)
        coverage = valid_sum / float(total_days)

        if sample_mode == STRICT_BALANCED_SAMPLE:
            mask = (valid_sum == total_days) & (observed_sum == total_days)
        else:
            mask = (coverage >= coverage_threshold) & (observed_sum == valid_sum)

    selected = universe.loc[mask].sort_values("symbol").copy()
    if max_stocks is not None:
        selected = selected.head(int(max_stocks)).copy()

    selected.attrs["summary"] = summary
    selected.attrs["sample_mode"] = sample_mode
    selected.attrs["years"] = years
    return selected


def _build_daily_return_map(
    df: pd.DataFrame,
    return_mode: str = "open_close",
) -> Dict[str, Dict[str, Any]]:
    """
    对单只股票构造完整交易日的日内/隔夜收益映射.

    返回值:
        {
            "YYYY-MM-DD": {
                "intraday": np.ndarray shape (48,),
                "overnight": float,
                "day_open": float,
                "day_close": float,
                "daily": float,
            },
            ...
        }
    """
    if return_mode not in {"open_close", "close_close"}:
        raise ValueError(f"未知 return_mode: {return_mode}")

    temp = df.copy()
    temp["date"] = temp["kline_time"].dt.normalize()
    temp["time"] = temp["kline_time"].dt.strftime("%H:%M:%S")

    daily_map: Dict[str, Dict[str, Any]] = {}
    previous_close: Optional[float] = None

    for date, day in temp.groupby("date", sort=True):
        times = day["time"].tolist()
        if len(day) != len(CN_5MIN_BAR_TIMES) or times != CN_5MIN_BAR_TIMES:
            continue
        if not bool(day[["open", "high", "low", "close"]].gt(0.0).all().all()):
            continue

        open_px = day["open"].to_numpy(dtype=float)
        close_px = day["close"].to_numpy(dtype=float)

        if return_mode == "open_close":
            intraday = np.log(close_px / open_px)
        else:
            intraday = np.empty(len(close_px), dtype=float)
            intraday[0] = np.log(close_px[0] / open_px[0])
            intraday[1:] = np.log(close_px[1:] / close_px[:-1])

        day_open = float(open_px[0])
        day_close = float(close_px[-1])
        overnight = 0.0 if previous_close is None else float(np.log(day_open / previous_close))
        daily_map[date.strftime("%Y-%m-%d")] = {
            "intraday": intraday,
            "overnight": overnight,
            "day_open": day_open,
            "day_close": day_close,
            "daily": float(intraday.sum() + overnight),
        }
        previous_close = day_close

    return daily_map


def _build_symbol_returns_cache_payload(
    file_path: Path,
    return_mode: str,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if df is None:
        df, _ = _load_cn_symbol_frame(file_path)

    if df.empty:
        intraday = np.zeros((0, len(CN_5MIN_BAR_TIMES)), dtype=np.float64)
        overnight = np.zeros(0, dtype=np.float64)
        day_open = np.zeros(0, dtype=np.float64)
        day_close = np.zeros(0, dtype=np.float64)
        daily = np.zeros(0, dtype=np.float64)
        date_codes = np.zeros(0, dtype=np.int32)
        sorted_dates: List[str] = []
    else:
        ts = df["kline_time"]
        dates = ts.to_numpy(dtype="datetime64[D]")
        time_codes = (ts.dt.hour.to_numpy(dtype=np.int16) * 100 + ts.dt.minute.to_numpy(dtype=np.int16)).astype(np.int32)
        open_px_all = df["open"].to_numpy(dtype=np.float64)
        close_px_all = df["close"].to_numpy(dtype=np.float64)
        price_ok_rows = df[["open", "high", "low", "close"]].gt(0.0).to_numpy().all(axis=1)

        boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(df)]

        valid_intraday = []
        valid_open = []
        valid_close = []
        sorted_dates = []
        previous_close: Optional[float] = None
        overnight_list: List[float] = []
        daily_list: List[float] = []

        for start, end in zip(starts, ends):
            is_valid_day = (
                (end - start) == len(CN_5MIN_BAR_TIMES)
                and bool(price_ok_rows[start:end].all())
                and np.array_equal(time_codes[start:end], CN_5MIN_BAR_CODES)
            )
            if not is_valid_day:
                continue

            day_open_arr = open_px_all[start:end]
            day_close_arr = close_px_all[start:end]
            if return_mode == "open_close":
                intraday_day = np.log(day_close_arr / day_open_arr)
            else:
                intraday_day = np.empty(len(day_close_arr), dtype=np.float64)
                intraday_day[0] = np.log(day_close_arr[0] / day_open_arr[0])
                intraday_day[1:] = np.log(day_close_arr[1:] / day_close_arr[:-1])

            day_open_val = float(day_open_arr[0])
            day_close_val = float(day_close_arr[-1])
            overnight_val = 0.0 if previous_close is None else float(np.log(day_open_val / previous_close))

            valid_intraday.append(intraday_day)
            valid_open.append(day_open_val)
            valid_close.append(day_close_val)
            overnight_list.append(overnight_val)
            daily_list.append(float(intraday_day.sum() + overnight_val))
            sorted_dates.append(np.datetime_as_string(dates[start], unit="D"))
            previous_close = day_close_val

        if sorted_dates:
            intraday = np.vstack(valid_intraday).astype(np.float64, copy=False)
            overnight = np.asarray(overnight_list, dtype=np.float64)
            day_open = np.asarray(valid_open, dtype=np.float64)
            day_close = np.asarray(valid_close, dtype=np.float64)
            daily = np.asarray(daily_list, dtype=np.float64)
            date_codes = np.array([int(date.replace("-", "")) for date in sorted_dates], dtype=np.int32)
        else:
            intraday = np.zeros((0, len(CN_5MIN_BAR_TIMES)), dtype=np.float64)
            overnight = np.zeros(0, dtype=np.float64)
            day_open = np.zeros(0, dtype=np.float64)
            day_close = np.zeros(0, dtype=np.float64)
            daily = np.zeros(0, dtype=np.float64)
            date_codes = np.zeros(0, dtype=np.int32)

    abs_overnight = np.abs(overnight)
    suspicious_012 = abs_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[0]
    suspicious_020 = abs_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[1]

    meta = {
        "version": RETURNS_CACHE_VERSION,
        "symbol": file_path.parent.name,
        "return_mode": return_mode,
        "signature": _symbol_file_signature(file_path),
        "n_days": int(len(sorted_dates)),
        "first_date": sorted_dates[0] if sorted_dates else None,
        "last_date": sorted_dates[-1] if sorted_dates else None,
        "suspicious_overnight_count_012": int(suspicious_012.sum()),
        "suspicious_overnight_count_020": int(suspicious_020.sum()),
        "max_abs_overnight": float(abs_overnight.max()) if len(abs_overnight) else 0.0,
    }
    arrays = {
        "date_codes": date_codes,
        "intraday_returns": intraday,
        "overnight_returns": overnight,
        "day_open": day_open,
        "day_close": day_close,
        "daily_returns": daily,
        "valid_day_mask": np.ones(len(date_codes), dtype=np.uint8),
        "suspicious_overnight_mask_012": suspicious_012.astype(np.uint8, copy=False),
        "suspicious_overnight_mask_020": suspicious_020.astype(np.uint8, copy=False),
    }
    return meta, arrays


def _load_symbol_returns_cached(
    file_path: Path,
    return_mode: str,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_root: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    meta_path, npz_path = _returns_cache_paths(cache_root, file_path, return_mode)
    signature = _symbol_file_signature(file_path)

    if use_cache and not refresh_cache and meta_path.exists() and npz_path.exists():
        try:
            meta = _load_json(meta_path)
            if meta.get("signature") == signature and meta.get("version") == RETURNS_CACHE_VERSION:
                arrays_raw = np.load(npz_path, allow_pickle=False)
                arrays = {name: arrays_raw[name] for name in arrays_raw.files}
                return meta, arrays
        except Exception:
            pass

    meta, arrays = _build_symbol_returns_cache_payload(file_path, return_mode=return_mode)
    if use_cache:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        _write_json(meta_path, meta)
    return meta, arrays


@dataclass
class HFPanel:
    """
    高频面板数据容器.

    字段:
        R_intra:
            shape (M_total, N), 所有日内 5 分钟收益拼接后的矩阵.
        R_night:
            shape (D, N), 日度隔夜收益.
        day_ids:
            shape (M_total,), 每个日内观测所属交易日索引.
        tickers:
            股票代码列表, 长度 N.
        dates:
            交易日日期, 长度 D.
    """

    R_intra: np.ndarray
    R_night: np.ndarray
    day_ids: np.ndarray
    tickers: List[str]
    dates: List[pd.Timestamp]
    R_daily: Optional[np.ndarray] = None
    rf_intra: Optional[np.ndarray] = None
    rf_night: Optional[np.ndarray] = None
    bar_times: Optional[List[str]] = None
    sample_report: Optional[Dict[str, Any]] = None
    sample_mode: str = "custom"
    return_mode: str = "open_close"

    def __post_init__(self) -> None:
        self.R_intra = np.asarray(self.R_intra, dtype=float)
        self.R_night = np.asarray(self.R_night, dtype=float)
        self.day_ids = np.asarray(self.day_ids, dtype=int)
        self.dates = [pd.Timestamp(date) for date in self.dates]
        self.tickers = [str(ticker) for ticker in self.tickers]
        if self.bar_times is not None:
            self.bar_times = list(self.bar_times)

        assert self.R_intra.ndim == 2, "R_intra 必须是 2D"
        assert self.R_night.ndim == 2, "R_night 必须是 2D"
        assert self.R_night.shape[0] == len(self.dates), "R_night 行数必须等于交易日数"
        assert self.R_intra.shape[1] == self.R_night.shape[1] == len(self.tickers), "股票维度不一致"
        assert self.day_ids.shape[0] == self.R_intra.shape[0], "day_ids 长度错误"

        if self.bar_times is not None and len(self.bar_times) > 0:
            inferred = self.R_intra.shape[0] / max(len(self.dates), 1)
            assert int(round(inferred)) == len(self.bar_times), "bar_times 与 R_intra 行数不一致"

        if self.R_daily is None:
            D, N = self.D, self.N
            R_intra_day_sum = np.zeros((D, N))
            np.add.at(R_intra_day_sum, self.day_ids, np.nan_to_num(self.R_intra, nan=0.0))
            self.R_daily = R_intra_day_sum + self.R_night
        else:
            self.R_daily = np.asarray(self.R_daily, dtype=float)

        if self.rf_intra is None:
            self.rf_intra = np.zeros(self.D)
        else:
            self.rf_intra = np.asarray(self.rf_intra, dtype=float)

        if self.rf_night is None:
            self.rf_night = np.zeros(self.D)
        else:
            self.rf_night = np.asarray(self.rf_night, dtype=float)

    @property
    def N(self) -> int:
        return int(self.R_intra.shape[1])

    @property
    def D(self) -> int:
        return int(len(self.dates))

    @property
    def M_per_day(self) -> int:
        if self.bar_times:
            return len(self.bar_times)
        return int(self.R_intra.shape[0] / max(self.D, 1))


def build_cn_hf_panel(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    sample_mode: str = STRICT_BALANCED_SAMPLE,
    return_mode: str = "open_close",
    years: Optional[Sequence[int]] = None,
    universe: Optional[pd.DataFrame] = None,
    max_stocks: Optional[int] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    refresh_symbol_cache: bool = False,
    cache_root: Optional[str | Path] = None,
) -> HFPanel:
    """
    从本地 A 股 `.bz2` 数据构造 `HFPanel`.

    默认主流程使用 `strict_balanced`:
        - 全样本默认会得到 2013-01-04 到 2016-12-30 的严格平衡样本
        - 每天 48 根日内收益

    `near_balanced_99` 可直接构造一个交集日历面板, 但这不是论文主结果默认口径.
    """
    data_root = _ensure_path(data_root)
    if universe is None:
        universe = scan_cn_bz2_universe(
            data_root=data_root,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_root=cache_root,
        )
    universe, summary = _get_universe_summary(universe, data_root=data_root)
    # Determine cache dependencies before attempting to load a cached panel.
    selected = _select_sample_rows(
        universe=universe,
        sample_mode=sample_mode,
        years=years,
        max_stocks=max_stocks,
    )
    if selected.empty:
        raise ValueError("娌℃湁绛涢€夊嚭浠讳綍鑲＄エ, 鏃犳硶鏋勯€?HFPanel")
    sample_signature_hash = _selected_sample_signature_hash(selected)
    panel_key = _panel_cache_key(
        data_root=data_root,
        summary=summary,
        sample_mode=sample_mode,
        return_mode=return_mode,
        years=years,
        max_stocks=max_stocks,
        sample_signature_hash=sample_signature_hash,
    )
    panel_meta_path, panel_npz_path = _panel_cache_paths(cache_root, panel_key)
    if use_cache and not (refresh_cache or refresh_symbol_cache) and panel_meta_path.exists() and panel_npz_path.exists():
        try:
            panel_meta = _load_json(panel_meta_path)
            if panel_meta.get("version") == PANEL_CACHE_VERSION:
                arrays_raw = np.load(panel_npz_path, allow_pickle=False)
                sample_report = panel_meta["sample_report"]
                return HFPanel(
                    R_intra=arrays_raw["R_intra"],
                    R_night=arrays_raw["R_night"],
                    day_ids=arrays_raw["day_ids"],
                    tickers=list(panel_meta["tickers"]),
                    dates=[pd.Timestamp(date) for date in panel_meta["dates"]],
                    R_daily=arrays_raw["R_daily"],
                    bar_times=list(panel_meta["bar_times"]),
                    sample_report=sample_report,
                    sample_mode=panel_meta["sample_mode"],
                    return_mode=panel_meta["return_mode"],
                )
        except Exception:
            pass

    selected = _select_sample_rows(
        universe=universe,
        sample_mode=sample_mode,
        years=years,
        max_stocks=max_stocks,
    )
    if selected.empty:
        raise ValueError("没有筛选出任何股票, 无法构造 HFPanel")

    target_dates = _selected_calendar_dates(summary, years)
    if not target_dates:
        raise ValueError("所选年份下没有交易日")

    target_date_codes = _date_codes_from_dates(target_dates)
    if sample_mode == STRICT_BALANCED_SAMPLE:
        common_date_codes = target_date_codes
        common_dates = target_dates
    else:
        target_date_code_set = {int(code) for code in target_date_codes.tolist()}
        common_date_codes_set: Optional[set[int]] = None
        for row in selected.itertuples(index=False):
            _, arrays = _load_symbol_returns_cached(
                Path(row.file_path),
                return_mode=return_mode,
                use_cache=use_cache,
                refresh_cache=refresh_cache or refresh_symbol_cache,
                cache_root=cache_root,
            )
            valid_date_codes = {
                int(code)
                for code in arrays["date_codes"].tolist()
                if int(code) in target_date_code_set
            }
            common_date_codes_set = (
                valid_date_codes
                if common_date_codes_set is None
                else (common_date_codes_set & valid_date_codes)
            )
        common_date_codes = np.array(
            [int(code) for code in target_date_codes if int(code) in (common_date_codes_set or set())],
            dtype=np.int32,
        )
        common_dates = [_date_code_to_text(code) for code in common_date_codes.tolist()]
        if not common_dates:
            raise ValueError("近似平衡样本在交集日历下没有可用交易日")

    D = len(common_dates)
    M = len(CN_5MIN_BAR_TIMES)
    N = len(selected)

    R_intra = np.zeros((D * M, N), dtype=float)
    R_night = np.zeros((D, N), dtype=float)
    R_daily = np.zeros((D, N), dtype=float)
    tickers = selected["symbol"].tolist()
    corp_action_rows: List[Dict[str, Any]] = []

    for col_idx, row in enumerate(selected.itertuples(index=False)):
        meta, arrays = _load_symbol_returns_cached(
            Path(row.file_path),
            return_mode=return_mode,
            use_cache=use_cache,
            refresh_cache=refresh_cache or refresh_symbol_cache,
            cache_root=cache_root,
        )
        date_codes = arrays["date_codes"]
        locate = {int(code): idx for idx, code in enumerate(date_codes.tolist())}

        for day_idx, date_code in enumerate(common_date_codes):
            arr_idx = locate.get(int(date_code))
            if arr_idx is None:
                raise ValueError(
                    f"{row.symbol} 缺少 {str(date_code)} 的完整日内数据; "
                    f"请改用严格平衡样本或更窄年份窗口"
                )
            start = day_idx * M
            end = start + M
            R_intra[start:end, col_idx] = arrays["intraday_returns"][arr_idx]
            R_night[day_idx, col_idx] = arrays["overnight_returns"][arr_idx]
            R_daily[day_idx, col_idx] = arrays["daily_returns"][arr_idx]

        corp_action_rows.append(
            {
                "symbol": row.symbol,
                "max_abs_overnight": float(meta.get("max_abs_overnight", 0.0)),
                "suspicious_overnight_count_012": int(meta.get("suspicious_overnight_count_012", 0)),
                "suspicious_overnight_count_020": int(meta.get("suspicious_overnight_count_020", 0)),
            }
        )

    day_ids = np.repeat(np.arange(D), M)
    dates = [pd.Timestamp(date_text) for date_text in common_dates]
    corp_action_df = pd.DataFrame(corp_action_rows)
    corp_action_summary = {
        "symbols_with_suspicious_overnight_012": int((corp_action_df["suspicious_overnight_count_012"] > 0).sum()),
        "symbols_with_suspicious_overnight_020": int((corp_action_df["suspicious_overnight_count_020"] > 0).sum()),
        "max_abs_overnight_overall": float(corp_action_df["max_abs_overnight"].max()) if not corp_action_df.empty else 0.0,
    }

    sample_report = {
        "data_root": str(data_root),
        "sample_mode": sample_mode,
        "return_mode": return_mode,
        "years": _normalize_years(years),
        "n_symbols_selected": int(N),
        "n_days_selected": int(D),
        "bars_per_day": int(M),
        "target_calendar_days": int(len(target_dates)),
        "selected_calendar_start": common_dates[0],
        "selected_calendar_end": common_dates[-1],
        "target_symbols": tickers,
        "corp_action_risk_summary": corp_action_summary,
    }
    panel = HFPanel(
        R_intra=R_intra,
        R_night=R_night,
        day_ids=day_ids,
        tickers=tickers,
        dates=dates,
        R_daily=R_daily,
        bar_times=list(CN_5MIN_BAR_TIMES),
        sample_report=sample_report,
        sample_mode=sample_mode,
        return_mode=return_mode,
    )
    if use_cache:
        panel_meta = {
            "version": PANEL_CACHE_VERSION,
            "sample_mode": sample_mode,
            "return_mode": return_mode,
            "dates": [date.strftime("%Y-%m-%d") for date in dates],
            "tickers": tickers,
            "bar_times": list(CN_5MIN_BAR_TIMES),
            "sample_report": sample_report,
            "sample_signature_hash": sample_signature_hash,
        }
        np.savez_compressed(
            panel_npz_path,
            R_intra=panel.R_intra,
            R_night=panel.R_night,
            R_daily=panel.R_daily,
            day_ids=panel.day_ids,
        )
        _write_json(panel_meta_path, panel_meta)
    return panel


def subset_panel_by_years(panel: HFPanel, years: Sequence[int]) -> HFPanel:
    """从现有 `HFPanel` 中切出指定年份子样本."""
    years = _normalize_years(years)
    if years is None:
        return panel

    day_mask = np.array([date.year in years for date in panel.dates], dtype=bool)
    if not day_mask.any():
        raise ValueError(f"HFPanel 中没有年份 {years}")

    selected_day_idx = np.where(day_mask)[0]
    row_mask = np.isin(panel.day_ids, selected_day_idx)
    remap = -np.ones(panel.D, dtype=int)
    remap[selected_day_idx] = np.arange(len(selected_day_idx))
    new_day_ids = remap[panel.day_ids[row_mask]]

    sample_report = dict(panel.sample_report or {})
    sample_report["years"] = list(years)
    sample_report["n_days_selected"] = int(len(selected_day_idx))
    sample_report["selected_calendar_start"] = panel.dates[selected_day_idx[0]].strftime("%Y-%m-%d")
    sample_report["selected_calendar_end"] = panel.dates[selected_day_idx[-1]].strftime("%Y-%m-%d")

    return HFPanel(
        R_intra=panel.R_intra[row_mask],
        R_night=panel.R_night[selected_day_idx],
        day_ids=new_day_ids,
        tickers=list(panel.tickers),
        dates=[panel.dates[idx] for idx in selected_day_idx],
        R_daily=panel.R_daily[selected_day_idx],
        rf_intra=panel.rf_intra[selected_day_idx],
        rf_night=panel.rf_night[selected_day_idx],
        bar_times=list(panel.bar_times or []),
        sample_report=sample_report,
        sample_mode=panel.sample_mode,
        return_mode=panel.return_mode,
    )


def _bipower_variation(r_day: np.ndarray) -> np.ndarray:
    """
    BNS bipower variation, 对 NaN 鲁棒.
    shape:
        r_day: (M, N)
        return: (N,)
    """
    abs_r = np.abs(r_day)
    prod = abs_r[1:] * abs_r[:-1]
    return (np.pi / 2.0) * np.nansum(prod, axis=0)


def _time_of_day_pattern(R_intra: np.ndarray, M_per_day: int) -> np.ndarray:
    """
    估计日内 time-of-day 模式.
    对有 NaN 的矩阵使用 nanmean, 以支持非平衡样本的稳健性分析.
    """
    D = R_intra.shape[0] // M_per_day
    R_cube = R_intra.reshape(D, M_per_day, -1)
    r2_by_tod = np.nanmean(R_cube ** 2, axis=0)
    tod = np.nanmean(r2_by_tod, axis=1)
    tod = np.where(np.isfinite(tod) & (tod > 0), tod, np.nan)
    scale = np.nanmean(tod)
    if not np.isfinite(scale) or scale <= 0:
        return np.ones(M_per_day)
    return tod / scale


def detect_jumps(
    panel: HFPanel,
    a: float = 3.0,
    omega_bar: float = 0.49,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TOD 阈值法识别跳跃.

    返回:
        R_cont, R_jump
    若输入包含 NaN, 输出会在缺失位置保留 NaN.
    """
    R = panel.R_intra
    D, M, N = panel.D, panel.M_per_day, panel.N
    delta_M = 1.0 / M

    BV = np.empty((D, N))
    R_cube = R.reshape(D, M, N)
    for day_idx in range(D):
        BV[day_idx] = _bipower_variation(R_cube[day_idx])
    BV = np.where(np.isfinite(BV) & (BV > 0.0), BV, np.nan)

    tod = _time_of_day_pattern(R, M)
    sqrt_BV = np.sqrt(BV)
    sqrt_TOD = np.sqrt(np.where(np.isfinite(tod) & (tod > 0.0), tod, 1.0))
    sigma_cube = sqrt_BV[:, None, :] * sqrt_TOD[None, :, None]
    sigma_hat = sigma_cube.reshape(D * M, N)
    thr = a * (delta_M ** omega_bar) * sigma_hat

    is_finite = np.isfinite(R)
    is_jump = is_finite & (np.abs(R) > thr)
    R_jump = np.where(~is_finite, np.nan, np.where(is_jump, R, 0.0))
    R_cont = np.where(~is_finite, np.nan, np.where(is_jump, 0.0, R))
    return R_cont, R_jump


def jump_summary_stats(R_cont: np.ndarray, R_jump: np.ndarray) -> Dict[str, float]:
    """对应论文 Table I 的两个核心统计量."""
    observed = int(np.isfinite(R_cont).sum())
    n_jump = int(np.count_nonzero(np.nan_to_num(R_jump, nan=0.0)))
    qv_total = float(np.nansum(R_cont ** 2) + np.nansum(R_jump ** 2))
    qv_jump = float(np.nansum(R_jump ** 2))
    return {
        "frac_jump_increments": n_jump / observed if observed > 0 else 0.0,
        "frac_qv_explained_by_jumps": qv_jump / qv_total if qv_total > 0 else 0.0,
    }


@dataclass
class PCAResult:
    """PCA 结果容器."""

    Lambda: np.ndarray
    F: np.ndarray
    eigvals: np.ndarray
    scales: np.ndarray
    use_corr: bool
    covariance: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None


def pca_factors(
    R: np.ndarray,
    K: int,
    use_corr: bool = True,
) -> PCAResult:
    """
    对完整矩形高频收益矩阵做 PCA.
    若矩阵含 NaN, 请改用 `pca_factors_pairwise`.
    """
    R = np.asarray(R, dtype=float)
    if np.isnan(R).any():
        raise ValueError("pca_factors 不接受 NaN; 请使用 pca_factors_pairwise")

    M, N = R.shape
    if use_corr:
        qv = np.sum(R ** 2, axis=0)
        scales = np.sqrt(np.where(qv > 0.0, qv, 1.0))
        Rs = R / scales
    else:
        Rs = R
        scales = np.ones(N)

    S = (Rs.T @ Rs) / N
    eigvals, V = _sym_eig_desc(S)
    K_eff = min(K, V.shape[1])
    Lambda = np.sqrt(N) * V[:, :K_eff]
    F = Rs @ Lambda / N
    return PCAResult(
        Lambda=Lambda,
        F=F,
        eigvals=eigvals,
        scales=scales,
        use_corr=use_corr,
        covariance=S,
    )


def _nearest_psd(M: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = _sym_eig_desc(M)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def pairwise_available_covariance(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用成对可用观测构造协方差/相关矩阵.

    返回:
        cov_like: pairwise mean cross-product matrix
        counts:   每对股票的共同观测数
    """
    mask = np.isfinite(R)
    X = np.where(mask, R, 0.0)
    counts = mask.astype(float).T @ mask.astype(float)
    cross = X.T @ X
    cov_like = np.divide(cross, counts, out=np.zeros_like(cross), where=counts > 0)
    cov_like = (cov_like + cov_like.T) / 2.0
    return cov_like, counts


def pca_factors_pairwise(
    R: np.ndarray,
    K: int,
    use_corr: bool = True,
    psd_fix: bool = True,
) -> PCAResult:
    """
    对含缺失值的非平衡样本做 pairwise-covariance PCA.

    该实现主要用于年度 99% 覆盖样本的稳健性分析。
    """
    R = np.asarray(R, dtype=float)
    M, N = R.shape
    if use_corr:
        qv = np.nansum(R ** 2, axis=0)
        scales = np.sqrt(np.where(qv > 0.0, qv, 1.0))
        Rs = R / scales
    else:
        Rs = R
        scales = np.ones(N)

    pairwise_cov, counts = pairwise_available_covariance(Rs)
    if psd_fix:
        pairwise_cov = _nearest_psd(pairwise_cov)

    eigvals, V = _sym_eig_desc(pairwise_cov)
    K_eff = min(K, V.shape[1])
    Lambda = np.sqrt(N) * V[:, :K_eff]
    F = np.nan_to_num(Rs, nan=0.0) @ Lambda / N

    return PCAResult(
        Lambda=Lambda,
        F=F,
        eigvals=eigvals,
        scales=scales,
        use_corr=use_corr,
        covariance=pairwise_cov,
        counts=counts,
    )


def factor_portfolio_weights(res: PCAResult) -> np.ndarray:
    """
    把 PCA loadings 转成股票组合权重:
        w = Lambda / (sqrt(N) * scales)
    """
    N = res.Lambda.shape[0]
    return res.Lambda / (np.sqrt(N) * res.scales[:, None])


def perturbed_eigenvalue_ratio(
    eigvals: np.ndarray,
    g_fn: str = "median_sqrtN",
    N: Optional[int] = None,
    gamma: float = 0.08,
    K_max: Optional[int] = None,
) -> Tuple[int, np.ndarray]:
    """Pelger (2019) 扰动特征值比率."""
    lam = np.asarray(eigvals, dtype=float).copy()
    if g_fn == "median_sqrtN":
        assert N is not None
        g = np.sqrt(N) * np.median(lam)
    elif g_fn == "logN":
        assert N is not None
        g = np.log(N) * np.median(lam)
    elif g_fn == "none":
        g = 0.0
    else:
        raise ValueError(f"未知 g_fn: {g_fn}")

    lam_tilde = lam + g
    ER = lam_tilde[:-1] / np.maximum(lam_tilde[1:], 1e-300)
    if K_max is None:
        K_max = len(ER)
    ER_search = ER[:K_max]
    mask = ER_search > (1.0 + gamma)
    K_hat = int(np.where(mask)[0].max() + 1) if mask.any() else 0
    return K_hat, ER


def generalized_correlations(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Bai-Ng (2006) generalized correlation."""
    FtF = F.T @ F
    GtG = G.T @ G
    FtG = F.T @ G
    GtF = FtG.T
    M = _safe_inv(GtG) @ GtF @ _safe_inv(FtF) @ FtG
    eigvals, _ = _sym_eig_desc((M + M.T) / 2.0)
    eigvals = np.clip(eigvals, 0.0, 1.0)
    k = min(F.shape[1], G.shape[1])
    return np.sqrt(eigvals[:k])


def build_proxy_factors(
    weights: np.ndarray,
    R_original: np.ndarray,
    top_fracs: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pelger-Xiong 风格的 proxy factors."""
    N, K = weights.shape
    if top_fracs is None:
        defaults = [1.0, 0.15, 0.11, 0.11]
        top_fracs = defaults[:K] + [0.11] * max(0, K - len(defaults))

    W_proxy = np.zeros_like(weights)
    for k in range(K):
        if k == 0:
            W_proxy[:, 0] = np.sign(weights[:, 0].mean()) / N
            continue

        n_keep = max(1, int(round(top_fracs[k] * N)))
        idx_keep = np.argsort(-np.abs(weights[:, k]))[:n_keep]
        W_proxy[idx_keep, k] = weights[idx_keep, k]
        orig_norm = np.linalg.norm(weights[:, k])
        cur_norm = np.linalg.norm(W_proxy[:, k])
        if cur_norm > 0:
            W_proxy[:, k] *= orig_norm / cur_norm

    F_proxy = np.nan_to_num(R_original, nan=0.0) @ W_proxy
    return W_proxy, F_proxy


def rolling_local_pca(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    step_days: int = 1,
    use_corr: bool = True,
) -> List[Dict[str, Any]]:
    """按滚动交易日窗口做局部 PCA."""
    D = int(day_ids.max()) + 1
    results: List[Dict[str, Any]] = []
    for start in range(0, D - window_days + 1, step_days):
        end = start + window_days
        mask = (day_ids >= start) & (day_ids < end)
        R_win = R[mask]
        if R_win.shape[0] < 2 * K:
            continue
        res = pca_factors(R_win, K=K, use_corr=use_corr)
        results.append({
            "start_day": start,
            "end_day": end,
            "Lambda": res.Lambda,
            "F": res.F,
            "eigvals": res.eigvals,
        })
    return results


def rolling_gc_vs_global(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    global_Lambda: np.ndarray,
    step_days: int = 1,
) -> np.ndarray:
    """局部载荷与全局载荷的 generalized correlation."""
    loc = rolling_local_pca(R, day_ids, window_days, K=K, step_days=step_days)
    if not loc:
        return np.zeros((0, K))
    GC = np.zeros((len(loc), K))
    for i, result in enumerate(loc):
        gc = generalized_correlations(global_Lambda, result["Lambda"])
        GC[i, : len(gc)] = gc
    return GC


def rolling_explained_variation(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    step_days: int = 1,
) -> np.ndarray:
    """滚动窗口下前 K 个因子解释的变异占比."""
    loc = rolling_local_pca(R, day_ids, window_days, K=K, step_days=step_days)
    ratios = []
    for result in loc:
        eigvals = result["eigvals"]
        total = float(eigvals.sum())
        ratios.append(float(eigvals[:K].sum() / total) if total > 0 else 0.0)
    return np.asarray(ratios, dtype=float)


def tangency_portfolio(
    returns: np.ndarray,
    rf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """最大 Sharpe 切点组合."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns[:, None]
    if rf is None:
        rf = np.zeros(returns.shape[0])
    rf = np.asarray(rf, dtype=float)

    excess = returns - rf[:, None]
    mu = excess.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(excess, rowvar=False, ddof=1))
    Sigma_inv = _safe_inv(Sigma)
    w = Sigma_inv @ mu
    sharpe = float(np.sqrt(max(float(mu @ Sigma_inv @ mu), 0.0)))
    return np.asarray(w).reshape(-1), sharpe


def intraday_overnight_sharpes(
    F_intra: np.ndarray,
    F_night: np.ndarray,
    rf_intra: Optional[np.ndarray] = None,
    rf_night: Optional[np.ndarray] = None,
    annualize: int = 252,
) -> Dict[str, float]:
    """复现论文 Table V 的核心日内/隔夜/日度 Sharpe."""
    F_intra = np.asarray(F_intra, dtype=float)
    F_night = np.asarray(F_night, dtype=float)
    if rf_intra is None:
        rf_intra = np.zeros(F_intra.shape[0])
    if rf_night is None:
        rf_night = np.zeros(F_night.shape[0])

    _, sr_intra = tangency_portfolio(F_intra, rf_intra)
    _, sr_night = tangency_portfolio(F_night, rf_night)
    F_daily = F_intra + F_night
    _, sr_daily = tangency_portfolio(F_daily, np.asarray(rf_intra) + np.asarray(rf_night))

    scale = np.sqrt(annualize)
    return {
        "SR_intraday": float(sr_intra * scale),
        "SR_overnight": float(sr_night * scale),
        "SR_daily": float(sr_daily * scale),
    }


def time_series_pricing(
    test_asset_returns: np.ndarray,
    factor_returns: np.ndarray,
) -> Dict[str, np.ndarray]:
    """批量时间序列回归."""
    Y = np.asarray(test_asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if F.ndim == 1:
        F = F[:, None]

    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    alpha = B[0, :]
    beta = B[1:, :].T

    fitted = X @ B
    resid = Y - fitted
    ss_res = np.sum(resid ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    R2 = 1.0 - ss_res / np.where(ss_tot > 0.0, ss_tot, 1.0)

    avg_ret = Y.mean(axis=0)
    pred = beta @ F.mean(axis=0)
    return {
        "alpha": alpha,
        "beta": beta,
        "R2": R2,
        "pred": pred,
        "avg_ret": avg_ret,
    }


def aggregate_intraday_to_daily(F_hf: np.ndarray, day_ids: np.ndarray) -> np.ndarray:
    """把高频因子收益聚合成日频收益."""
    D = int(day_ids.max()) + 1
    K = F_hf.shape[1]
    out = np.zeros((D, K))
    np.add.at(out, day_ids, np.nan_to_num(F_hf, nan=0.0))
    return out


@dataclass
class PelgerPipeline:
    """论文主流程封装."""

    panel: HFPanel
    jump_a: float = 3.0
    K_max: int = 10
    gamma: float = 0.08
    use_corr: bool = True

    R_cont: Optional[np.ndarray] = None
    R_jump: Optional[np.ndarray] = None
    jump_stats: Dict[str, float] = field(default_factory=dict)

    K_hf_hat: int = 0
    K_cont_hat: int = 0
    K_jump_hat: int = 0

    pca_hf: Optional[PCAResult] = None
    pca_cont: Optional[PCAResult] = None
    pca_jump: Optional[PCAResult] = None

    F_cont_daily_intra: Optional[np.ndarray] = None
    F_cont_daily_night: Optional[np.ndarray] = None
    sharpes: Dict[str, float] = field(default_factory=dict)

    def step1_decompose(self) -> None:
        self.R_cont, self.R_jump = detect_jumps(self.panel, a=self.jump_a)
        self.jump_stats = jump_summary_stats(self.R_cont, self.R_jump)

    def step2_determine_K(self) -> None:
        N = self.panel.N
        r_hf = pca_factors(self.panel.R_intra, K=1, use_corr=self.use_corr)
        r_cont = pca_factors(self.R_cont, K=1, use_corr=self.use_corr)
        r_jump = pca_factors(self.R_jump, K=1, use_corr=self.use_corr)
        self.K_hf_hat, _ = perturbed_eigenvalue_ratio(r_hf.eigvals, N=N, gamma=self.gamma, K_max=self.K_max)
        self.K_cont_hat, _ = perturbed_eigenvalue_ratio(r_cont.eigvals, N=N, gamma=self.gamma, K_max=self.K_max)
        self.K_jump_hat, _ = perturbed_eigenvalue_ratio(r_jump.eigvals, N=N, gamma=self.gamma, K_max=self.K_max)

        self.K_hf_hat = max(1, self.K_hf_hat)
        self.K_cont_hat = max(1, self.K_cont_hat)
        self.K_jump_hat = max(1, self.K_jump_hat)

    def step3_extract_factors(self) -> None:
        self.pca_hf = pca_factors(self.panel.R_intra, K=self.K_hf_hat, use_corr=self.use_corr)
        self.pca_cont = pca_factors(self.R_cont, K=self.K_cont_hat, use_corr=self.use_corr)
        self.pca_jump = pca_factors(self.R_jump, K=self.K_jump_hat, use_corr=self.use_corr)

    def step4_asset_pricing(self) -> None:
        W = factor_portfolio_weights(self.pca_cont)
        F_intra_hf = self.panel.R_intra @ W
        F_intra_daily = aggregate_intraday_to_daily(F_intra_hf, self.panel.day_ids)
        F_night_daily = self.panel.R_night @ W

        self.F_cont_daily_intra = F_intra_daily
        self.F_cont_daily_night = F_night_daily
        self.sharpes = intraday_overnight_sharpes(
            F_intra=F_intra_daily,
            F_night=F_night_daily,
            rf_intra=self.panel.rf_intra,
            rf_night=self.panel.rf_night,
        )

    def run_full(self) -> "PelgerPipeline":
        self.step1_decompose()
        self.step2_determine_K()
        self.step3_extract_factors()
        self.step4_asset_pricing()
        return self


def print_pipeline_summary(pipe: PelgerPipeline) -> None:
    """打印论文主结果摘要."""
    report = pipe.panel.sample_report or {}
    print("=" * 78)
    print("Pelger (2020) China A-share High-Frequency Replication")
    print("=" * 78)
    print(f" Sample mode   : {pipe.panel.sample_mode}")
    print(f" Return mode   : {pipe.panel.return_mode}")
    print(f" N stocks      : {pipe.panel.N}")
    print(f" D days        : {pipe.panel.D}")
    print(f" M per day     : {pipe.panel.M_per_day}")
    if report.get("years"):
        print(f" Years         : {report['years']}")
    print("-" * 78)
    print(" Jump decomposition (Table I)")
    print(f"   frac jumps      : {pipe.jump_stats['frac_jump_increments']:.4f}")
    print(f"   frac QV by jumps: {pipe.jump_stats['frac_qv_explained_by_jumps']:.4f}")
    print("-" * 78)
    print(" Number of factors")
    print(f"   K_HF   : {pipe.K_hf_hat}")
    print(f"   K_cont : {pipe.K_cont_hat}")
    print(f"   K_jump : {pipe.K_jump_hat}")
    print("-" * 78)
    print(" Annualized tangency-portfolio Sharpe")
    for name, value in pipe.sharpes.items():
        print(f"   {name:15s}: {value:.3f}")
    print("=" * 78)


def _build_nan_intraday_matrix_for_year(
    selected: pd.DataFrame,
    year_dates: Sequence[str],
    return_mode: str,
) -> Tuple[np.ndarray, List[str]]:
    """为年度非平衡样本构造带 NaN 的日内收益矩阵."""
    M = len(CN_5MIN_BAR_TIMES)
    D = len(year_dates)
    N = len(selected)
    R = np.full((D * M, N), np.nan, dtype=float)
    tickers = selected["symbol"].tolist()

    date_index = {date: idx for idx, date in enumerate(year_dates)}
    for col_idx, row in enumerate(selected.itertuples(index=False)):
        df, _ = _load_cn_symbol_frame(Path(row.file_path))
        daily_map = _build_daily_return_map(df, return_mode=return_mode)
        for date_text, day_info in daily_map.items():
            if date_text not in date_index:
                continue
            day_idx = date_index[date_text]
            start = day_idx * M
            end = start + M
            R[start:end, col_idx] = day_info["intraday"]

    return R, tickers


def _build_strict_year_panel(
    data_root: Path,
    universe: pd.DataFrame,
    year: int,
    return_mode: str,
) -> HFPanel:
    return build_cn_hf_panel(
        data_root=data_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        return_mode=return_mode,
        years=[year],
        universe=universe,
    )


def run_near_balanced_robustness(
    data_root: str | Path,
    universe: pd.DataFrame,
    years: Optional[Sequence[int]],
    K_compare: int,
    return_mode: str = "open_close",
    jump_a: float = 3.0,
    max_near_symbols: Optional[int] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    refresh_symbol_cache: bool = False,
    cache_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    对年度 99% 覆盖样本做稳健性比较:
    - 平衡样本: 常规 PCA
    - 99% 样本: pairwise-covariance PCA + PSD 修正
    - 比较两者连续因子空间的 generalized correlation
    """
    data_root = _ensure_path(data_root)
    universe, summary = _get_universe_summary(universe, data_root=data_root)
    years = _normalize_years(years) or sorted(int(year) for year in summary["calendar_days_by_year"])
    strict_full_panel = build_cn_hf_panel(
        data_root=data_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        return_mode=return_mode,
        years=None,
        universe=universe,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        refresh_symbol_cache=refresh_symbol_cache,
        cache_root=cache_root,
    )

    rows: List[Dict[str, Any]] = []
    for year in years:
        near_selected = _select_sample_rows(
            universe=universe,
            sample_mode=NEAR_BALANCED_99_SAMPLE,
            years=[year],
            max_stocks=max_near_symbols,
        )
        if near_selected.empty:
            continue

        strict_panel = subset_panel_by_years(strict_full_panel, [year])
        R_cont_strict, _ = detect_jumps(strict_panel, a=jump_a)
        pca_strict = pca_factors(R_cont_strict, K=K_compare, use_corr=True)

        year_dates = _selected_calendar_dates(summary, [year])
        year_date_codes = np.array([int(date.replace("-", "")) for date in year_dates], dtype=np.int32)
        M = len(CN_5MIN_BAR_TIMES)
        R_near = np.full((len(year_dates) * M, len(near_selected)), np.nan, dtype=float)
        for col_idx, row in enumerate(near_selected.itertuples(index=False)):
            _, arrays = _load_symbol_returns_cached(
                Path(row.file_path),
                return_mode=return_mode,
                use_cache=use_cache,
                refresh_cache=refresh_cache or refresh_symbol_cache,
                cache_root=cache_root,
            )
            locate = {int(code): idx for idx, code in enumerate(arrays["date_codes"].tolist())}
            for day_idx, date_code in enumerate(year_date_codes):
                arr_idx = locate.get(int(date_code))
                if arr_idx is None:
                    continue
                start = day_idx * M
                end = start + M
                R_near[start:end, col_idx] = arrays["intraday_returns"][arr_idx]
        day_ids = np.repeat(np.arange(len(year_dates)), len(CN_5MIN_BAR_TIMES))
        near_panel = HFPanel(
            R_intra=R_near,
            R_night=np.zeros((len(year_dates), len(near_selected))),
            day_ids=day_ids,
            tickers=near_selected["symbol"].tolist(),
            dates=[pd.Timestamp(date) for date in year_dates],
            bar_times=list(CN_5MIN_BAR_TIMES),
            sample_mode=NEAR_BALANCED_99_SAMPLE,
            return_mode=return_mode,
        )
        R_cont_near, _ = detect_jumps(near_panel, a=jump_a)
        pca_near = pca_factors_pairwise(R_cont_near, K=K_compare, use_corr=True, psd_fix=True)

        gc = generalized_correlations(pca_strict.F, pca_near.F)
        row = {
            "year": int(year),
            "n_strict_symbols": int(strict_panel.N),
            "n_near_symbols": int(len(near_selected)),
            "n_days": int(len(year_dates)),
        }
        for idx, value in enumerate(gc, start=1):
            row[f"gc_{idx}"] = float(value)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["year", "n_strict_symbols", "n_near_symbols", "n_days"])
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def load_external_factor_csv(
    path: str | Path,
    freq: str,
    schema: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    预留的外部因子加载接口.

    参数:
        path:
            CSV / Parquet 路径.
        freq:
            仅作记录与诊断使用, 例如 "daily" / "hf".
        schema:
            可选映射, 例如 {"datetime": "timestamp", "factor_1": "MKT"}.
    """
    path = _ensure_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

    if schema:
        missing = sorted(set(schema.values()) - set(df.columns))
        if missing:
            raise ValueError(f"{path} 缺少 schema 指定的列: {missing}")
        df = df.rename(columns={v: k for k, v in schema.items()})

    df.attrs["freq"] = str(freq)
    df.attrs["path"] = str(path)
    return df


def load_test_asset_csv(
    path: str | Path,
    schema: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """预留的测试资产收益加载接口."""
    return load_external_factor_csv(path=path, freq="test_asset", schema=schema)


@dataclass
class ReplicationResult:
    universe: pd.DataFrame
    panel: HFPanel
    pipeline: PelgerPipeline
    rolling_gc: np.ndarray
    rolling_explained_variation: np.ndarray
    robustness: pd.DataFrame
    output_root: Path
    corp_action_risk: pd.DataFrame = field(default_factory=pd.DataFrame)
    exported_files: Dict[str, str] = field(default_factory=dict)


def _rolling_output_frames(
    rolling_gc: np.ndarray,
    rolling_explained: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gc_cols = [f"gc_{idx}" for idx in range(1, rolling_gc.shape[1] + 1)] if rolling_gc.size else []
    rolling_gc_df = pd.DataFrame(rolling_gc, columns=gc_cols)
    rolling_gc_df.insert(0, "window_index", np.arange(len(rolling_gc_df)))

    rolling_explained_df = pd.DataFrame(
        {
            "window_index": np.arange(len(rolling_explained)),
            "explained_variation": rolling_explained,
        }
    )
    return rolling_gc_df, rolling_explained_df


def _maybe_save_plot(
    series_df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
    title: str,
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in y_cols:
        ax.plot(series_df[x_col], series_df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def export_replication_outputs(
    result: ReplicationResult,
    save_plots: bool = True,
) -> Dict[str, str]:
    """导出核心表格、摘要和可选图形."""
    output_root = result.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    universe_summary = summarize_cn_universe(result.universe)
    main_summary = {
        "sample_report": result.panel.sample_report or {},
        "jump_stats": result.pipeline.jump_stats,
        "factor_counts": {
            "K_hf_hat": result.pipeline.K_hf_hat,
            "K_cont_hat": result.pipeline.K_cont_hat,
            "K_jump_hat": result.pipeline.K_jump_hat,
        },
        "sharpes": result.pipeline.sharpes,
    }

    universe_csv = output_root / "universe_scan.csv"
    result.universe.to_csv(universe_csv, index=False, encoding="utf-8-sig")
    _write_json(output_root / "universe_summary.json", universe_summary)
    _write_json(output_root / "main_summary.json", main_summary)

    pd.DataFrame([result.pipeline.jump_stats]).to_csv(
        output_root / "jump_stats.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [{
            "K_hf_hat": result.pipeline.K_hf_hat,
            "K_cont_hat": result.pipeline.K_cont_hat,
            "K_jump_hat": result.pipeline.K_jump_hat,
        }]
    ).to_csv(output_root / "factor_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result.pipeline.sharpes]).to_csv(
        output_root / "sharpes.csv", index=False, encoding="utf-8-sig"
    )

    sample_symbols_path = output_root / "main_sample_symbols.csv"
    pd.DataFrame({"symbol": result.panel.tickers}).to_csv(
        sample_symbols_path,
        index=False,
        encoding="utf-8-sig",
    )

    rolling_gc_df, rolling_explained_df = _rolling_output_frames(
        result.rolling_gc,
        result.rolling_explained_variation,
    )
    rolling_gc_path = output_root / "rolling_gc.csv"
    rolling_ev_path = output_root / "rolling_explained_variation.csv"
    rolling_gc_df.to_csv(rolling_gc_path, index=False, encoding="utf-8-sig")
    rolling_explained_df.to_csv(rolling_ev_path, index=False, encoding="utf-8-sig")

    robustness_path = output_root / "robustness_yearly_gc.csv"
    result.robustness.to_csv(robustness_path, index=False, encoding="utf-8-sig")

    corp_action_path = output_root / "corp_action_risk.csv"
    result.corp_action_risk.to_csv(corp_action_path, index=False, encoding="utf-8-sig")

    exported_files = {
        "universe_scan": str(universe_csv),
        "universe_summary": str(output_root / "universe_summary.json"),
        "main_summary": str(output_root / "main_summary.json"),
        "jump_stats": str(output_root / "jump_stats.csv"),
        "factor_counts": str(output_root / "factor_counts.csv"),
        "sharpes": str(output_root / "sharpes.csv"),
        "main_sample_symbols": str(sample_symbols_path),
        "rolling_gc": str(rolling_gc_path),
        "rolling_explained_variation": str(rolling_ev_path),
        "robustness_yearly_gc": str(robustness_path),
        "corp_action_risk": str(corp_action_path),
    }

    if save_plots and not rolling_gc_df.empty:
        gc_plot = output_root / "rolling_gc.png"
        if _maybe_save_plot(
            series_df=rolling_gc_df,
            x_col="window_index",
            y_cols=[col for col in rolling_gc_df.columns if col.startswith("gc_")],
            title="Rolling Generalized Correlations vs Global Continuous Factors",
            output_path=gc_plot,
        ):
            exported_files["rolling_gc_plot"] = str(gc_plot)

    if save_plots and not rolling_explained_df.empty:
        ev_plot = output_root / "rolling_explained_variation.png"
        if _maybe_save_plot(
            series_df=rolling_explained_df,
            x_col="window_index",
            y_cols=["explained_variation"],
            title="Rolling Explained Variation",
            output_path=ev_plot,
        ):
            exported_files["rolling_explained_plot"] = str(ev_plot)

    result.exported_files = exported_files
    return exported_files


def run_cn_replication(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    years: Optional[Sequence[int]] = None,
    run_robustness: bool = True,
    return_mode: str = "open_close",
    sample_mode: str = STRICT_BALANCED_SAMPLE,
    max_stocks: Optional[int] = None,
    jump_a: float = 3.0,
    k_max: int = 10,
    gamma: float = 0.08,
    save_plots: bool = True,
    universe: Optional[pd.DataFrame] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    refresh_symbol_cache: bool = False,
    cache_root: Optional[str | Path] = None,
) -> ReplicationResult:
    """
    一站式运行中国 A 股复现流程.

    默认行为:
    - 扫描全市场 `.bz2`
    - 用严格平衡样本构造主面板
    - 跑完整论文主干
    - 导出核心表和图
    - 可选执行年度 99% 样本稳健性
    """
    data_root = _ensure_path(data_root)
    output_root = _ensure_path(output_root)

    universe = scan_cn_bz2_universe(
        data_root=data_root,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_root=cache_root,
    ) if universe is None else universe
    panel = build_cn_hf_panel(
        data_root=data_root,
        sample_mode=sample_mode,
        return_mode=return_mode,
        years=years,
        universe=universe,
        max_stocks=max_stocks,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        refresh_symbol_cache=refresh_symbol_cache,
        cache_root=cache_root,
    )
    corp_action_rows = []
    for ticker in panel.tickers:
        row = universe.loc[universe["symbol"] == ticker]
        if row.empty:
            continue
        row0 = row.iloc[0]
        corp_action_rows.append(
            {
                "symbol": ticker,
                "max_abs_overnight": float(row0.get("max_abs_overnight", np.nan)) if "max_abs_overnight" in row0.index else np.nan,
                "suspicious_overnight_count_012": int(row0.get("suspicious_overnight_count_012", 0)) if "suspicious_overnight_count_012" in row0.index else 0,
                "suspicious_overnight_count_020": int(row0.get("suspicious_overnight_count_020", 0)) if "suspicious_overnight_count_020" in row0.index else 0,
            }
        )
    corp_action_risk = pd.DataFrame(corp_action_rows)

    pipeline = PelgerPipeline(
        panel=panel,
        jump_a=jump_a,
        K_max=k_max,
        gamma=gamma,
    ).run_full()

    rolling_window = 21 if panel.D >= 21 else max(5, panel.D // 3)
    rolling_gc = rolling_gc_vs_global(
        R=pipeline.R_cont,
        day_ids=panel.day_ids,
        window_days=max(rolling_window, 2),
        K=pipeline.K_cont_hat,
        global_Lambda=pipeline.pca_cont.Lambda,
        step_days=1,
    )
    rolling_ev = rolling_explained_variation(
        R=pipeline.R_cont,
        day_ids=panel.day_ids,
        window_days=max(rolling_window, 2),
        K=pipeline.K_cont_hat,
        step_days=1,
    )

    robustness = pd.DataFrame()
    if run_robustness:
        robustness = run_near_balanced_robustness(
            data_root=data_root,
            universe=universe,
            years=years,
            K_compare=pipeline.K_cont_hat,
            return_mode=return_mode,
            jump_a=jump_a,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            refresh_symbol_cache=refresh_symbol_cache,
            cache_root=cache_root,
        )

    result = ReplicationResult(
        universe=universe,
        panel=panel,
        pipeline=pipeline,
        rolling_gc=rolling_gc,
        rolling_explained_variation=rolling_ev,
        robustness=robustness,
        output_root=output_root,
        corp_action_risk=corp_action_risk,
    )
    export_replication_outputs(result, save_plots=save_plots)
    return result


def _print_scan_summary(universe: pd.DataFrame) -> None:
    summary = summarize_cn_universe(universe)
    print("=" * 78)
    print("China A-share 5-minute universe scan")
    print("=" * 78)
    print(f" Data root              : {summary['data_root']}")
    print(f" Total symbols          : {summary['total_symbols']}")
    print(f" Global date range      : {summary['global_start']} -> {summary['global_end']}")
    print(f" Global calendar days   : {summary['global_calendar_days']}")
    print(f" Bars per day           : {summary['bars_per_day']}")
    print(f" Strict balanced sample : {summary['strict_balanced_symbols']}")
    print(f" Near-balanced 99%      : {summary['near_balanced_99_symbols']}")
    print(f" Invalid-day symbols    : {summary['invalid_day_symbols']}")
    corp = summary.get("corp_action_risk_summary")
    if corp:
        print(f" Suspicious overnight>12% symbols : {corp['symbols_with_suspicious_overnight_012']}")
        print(f" Suspicious overnight>20% symbols : {corp['symbols_with_suspicious_overnight_020']}")
        print(f" Max abs overnight                : {corp['max_abs_overnight_overall']:.4f}")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Pelger (2020) replication on local China A-share 5-minute bz2 data."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="EXTRA_STOCK_A 根目录")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="结果输出目录")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="缓存目录")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--refresh-cache", action="store_true", help="刷新全部缓存")
    parser.add_argument("--refresh-symbol-cache", action="store_true", help="刷新逐股票收益缓存")
    parser.add_argument("--scan-only", action="store_true", help="只扫描样本, 不运行主流程")
    parser.add_argument(
        "--sample-mode",
        default=STRICT_BALANCED_SAMPLE,
        choices=[STRICT_BALANCED_SAMPLE, NEAR_BALANCED_99_SAMPLE],
        help="主流程样本口径",
    )
    parser.add_argument(
        "--return-mode",
        default="open_close",
        choices=["open_close", "close_close"],
        help="日内收益构造口径",
    )
    parser.add_argument("--years", nargs="+", type=int, help="只运行指定年份, 例如 --years 2015 2016")
    parser.add_argument("--max-stocks", type=int, help="仅取前若干只股票做 smoke test")
    parser.add_argument("--no-robustness", action="store_true", help="跳过 99% 覆盖样本稳健性分析")
    parser.add_argument("--no-plots", action="store_true", help="不输出 PNG 图形")
    parser.add_argument("--jump-a", type=float, default=3.0, help="跳跃阈值倍数")
    parser.add_argument("--k-max", type=int, default=10, help="扰动特征值比率搜索上限")
    parser.add_argument("--gamma", type=float, default=0.08, help="扰动特征值比率阈值")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = _ensure_path(args.data_root)
    output_root = _ensure_path(args.output_root)
    cache_root = _ensure_path(args.cache_root)
    use_cache = not args.no_cache
    refresh_cache = bool(args.refresh_cache)

    universe = scan_cn_bz2_universe(
        data_root=data_root,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_root=cache_root,
    )
    _print_scan_summary(universe)

    output_root.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_root / "universe_scan.csv", index=False, encoding="utf-8-sig")
    _write_json(output_root / "universe_summary.json", summarize_cn_universe(universe))

    if args.scan_only:
        return 0

    result = run_cn_replication(
        data_root=data_root,
        output_root=output_root,
        years=args.years,
        run_robustness=not args.no_robustness,
        return_mode=args.return_mode,
        sample_mode=args.sample_mode,
        max_stocks=args.max_stocks,
        jump_a=args.jump_a,
        k_max=args.k_max,
        gamma=args.gamma,
        save_plots=not args.no_plots,
        universe=universe,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        refresh_symbol_cache=bool(args.refresh_symbol_cache),
        cache_root=cache_root,
    )
    print_pipeline_summary(result.pipeline)
    print("Exported files:")
    for name, path in sorted(result.exported_files.items()):
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
