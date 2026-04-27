"""Preprocess China A-share 5-minute K-line data for the Pelger replication.

This script is the only project stage that reads raw `data.bz2` files. It cleans
raw bars, applies `backward_factor.csv`, constructs intraday/overnight/daily
returns, and writes fast processed panels for `allcode_Need.py`.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "Data" / "kline_Data" / "EXTRA_STOCK_A"
DEFAULT_FACTOR_PATH = REPO_ROOT / "Data" / "fact_Data" / "backward_factor.csv"
DEFAULT_PROC_ROOT = REPO_ROOT / "Data" / "proc_Data" / "pelger_cn_adjusted"
DEFAULT_PREPROCESS_CACHE_ROOT = REPO_ROOT / ".hf_cache" / "pelger_cn_preprocess"
PREPROCESS_VERSION = "v2"

REQUIRED_RAW_COLUMNS = ["code", "kline_time", "open", "high", "low", "close", "volume", "amount"]
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
SUSPICIOUS_OVERNIGHT_THRESHOLDS = (0.12, 0.20)


def _default_worker_count() -> int:
    """给大数据预处理使用的保守默认并行度，避免磁盘和内存被打满。"""
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def _normalize_worker_count(workers: Optional[int]) -> int:
    """把 CLI/API 传入的 worker 数规范为正整数。"""
    if workers is None:
        return _default_worker_count()
    return max(1, int(workers))


def _ensure_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _normalize_years(years: Optional[Sequence[int]]) -> Optional[List[int]]:
    if years is None:
        return None
    normalized = sorted({int(year) for year in years})
    return normalized or None


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
    """以 UTF-8 JSON 保存元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    """读取 UTF-8 JSON 元数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _file_signature(path: Path) -> Dict[str, Any]:
    """记录文件签名，用于判断预处理结果是否过期。"""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _signature_hash(payload: Dict[str, Any]) -> str:
    """生成稳定哈希，便于在 manifest 中记录输入状态。"""
    encoded = json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _date_codes_from_dates(dates: Sequence[str]) -> np.ndarray:
    """将 YYYY-MM-DD 日期转为 int32 日期码。"""
    return np.array([int(str(date).replace("-", "")) for date in dates], dtype=np.int32)


def _date_code_to_text(code: int) -> str:
    """将 20160104 形式的日期码转回 YYYY-MM-DD。"""
    text = f"{int(code):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _find_raw_kline_files(raw_root: Path) -> List[Path]:
    """Return all raw symbol files under EXTRA_STOCK_A."""
    files = sorted(raw_root.rglob("data.bz2"))
    if not files:
        raise FileNotFoundError(f"No data.bz2 files found under {raw_root}")
    return files


def _load_cn_symbol_frame(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Read one pandas-bz2 K-line file and normalize columns used downstream."""
    raw = pd.read_pickle(file_path, compression="bz2")
    missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {missing}")
    df = raw.loc[:, REQUIRED_RAW_COLUMNS].copy()
    raw_rows = int(len(df))
    df["kline_time"] = pd.to_datetime(df["kline_time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["kline_time", "open", "high", "low", "close"])
    dropped_bad_rows = raw_rows - int(len(df))
    duplicate_rows = int(df.duplicated(subset=["kline_time"], keep="last").sum())
    df = df.drop_duplicates(subset=["kline_time"], keep="last").sort_values("kline_time").reset_index(drop=True)
    return df, {"raw_rows": raw_rows, "dropped_bad_rows": dropped_bad_rows, "duplicate_rows": duplicate_rows}


def _classify_symbol_days(df: pd.DataFrame) -> Dict[str, Any]:
    """Classify complete 48-bar trading days for one symbol."""
    valid_dates: List[str] = []
    invalid_dates: List[str] = []
    valid_days_by_year: Dict[int, int] = {}
    observed_days_by_year: Dict[int, int] = {}
    bars_by_year: Dict[int, int] = {}
    bad_price_days = 0
    if df.empty:
        return {"valid_dates": valid_dates, "invalid_dates": invalid_dates, "valid_days_by_year": valid_days_by_year, "observed_days_by_year": observed_days_by_year, "bars_by_year": bars_by_year, "bad_price_days": bad_price_days}
    timestamps = df["kline_time"]
    dates = timestamps.to_numpy(dtype="datetime64[D]")
    time_codes = (timestamps.dt.hour.to_numpy(dtype=np.int16) * 100 + timestamps.dt.minute.to_numpy(dtype=np.int16)).astype(np.int32)
    price_ok_rows = df[["open", "high", "low", "close"]].gt(0.0).to_numpy().all(axis=1)
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(df)]
    for start, end in zip(starts, ends):
        date_text = str(np.datetime_as_string(dates[start], unit="D"))
        year = int(pd.Timestamp(date_text).year)
        observed_days_by_year[year] = observed_days_by_year.get(year, 0) + 1
        bars_by_year[year] = bars_by_year.get(year, 0) + int(end - start)
        prices_ok = bool(price_ok_rows[start:end].all())
        if not prices_ok:
            bad_price_days += 1
        is_valid_day = (end - start) == len(CN_5MIN_BAR_TIMES) and prices_ok and np.array_equal(time_codes[start:end], CN_5MIN_BAR_CODES)
        if is_valid_day:
            valid_dates.append(date_text)
            valid_days_by_year[year] = valid_days_by_year.get(year, 0) + 1
        else:
            invalid_dates.append(date_text)
    return {"valid_dates": valid_dates, "invalid_dates": invalid_dates, "valid_days_by_year": valid_days_by_year, "observed_days_by_year": observed_days_by_year, "bars_by_year": bars_by_year, "bad_price_days": bad_price_days}


def _raw_symbol_record(file_path: Path) -> Dict[str, Any]:
    df, clean_stats = _load_cn_symbol_frame(file_path)
    day_info = _classify_symbol_days(df)
    valid_dates = day_info["valid_dates"]
    invalid_dates = day_info["invalid_dates"]
    stat = file_path.stat()
    return {"symbol": file_path.parent.name, "file_path": str(file_path.resolve()), "file_size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "first_valid_date": valid_dates[0] if valid_dates else None, "last_valid_date": valid_dates[-1] if valid_dates else None, "n_valid_days": int(len(valid_dates)), "n_invalid_days": int(len(invalid_dates)), "valid_dates": valid_dates, "invalid_dates": invalid_dates, "valid_days_by_year": day_info["valid_days_by_year"], "observed_days_by_year": day_info["observed_days_by_year"], "bars_by_year": day_info["bars_by_year"], "bad_price_days": int(day_info["bad_price_days"]), **clean_stats}


def scan_raw_kline_universe(raw_root: str | Path = DEFAULT_RAW_ROOT, max_files: Optional[int] = None) -> pd.DataFrame:
    """Scan raw K-line files. This is intentionally owned by preprocessing only."""
    raw_root = _ensure_path(raw_root)
    files = _find_raw_kline_files(raw_root)
    if max_files is not None:
        files = files[: int(max_files)]
    rows = [_raw_symbol_record(file_path) for file_path in files]
    universe = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    return _summarize_symbol_universe(universe, raw_root)


def _symbol_npz_path(proc_root: Path, symbol: str) -> Path:
    """逐股票收益数组保存位置。"""
    return proc_root / "symbol_returns" / f"{symbol}.npz"


def _symbol_meta_path(proc_root: Path, symbol: str) -> Path:
    """逐股票收益元数据保存位置。"""
    return proc_root / "symbol_returns" / f"{symbol}.json"


def _panel_paths(proc_root: Path, sample_mode: str, panel_name: str) -> Tuple[Path, Path]:
    """面板数组和面板元数据保存位置。"""
    panel_dir = proc_root / "panels" / sample_mode
    return panel_dir / panel_name, panel_dir / f"{panel_name}.json"


def load_backward_factor_matrix(
    factor_path: str | Path,
    symbols: Sequence[str],
    dates: Sequence[str],
) -> pd.DataFrame:
    """按需读取后复权因子宽表，并按股票和日期对齐。

    `backward_factor.csv` 是宽表：第一列为交易日，后续列为股票代码。
    为了避免一次读取全部 5500 多列，本函数只加载当前预处理需要的股票列。
    """
    factor_path = _ensure_path(factor_path)
    if not factor_path.exists():
        raise FileNotFoundError(f"复权因子文件不存在: {factor_path}")

    header = pd.read_csv(factor_path, nrows=0).columns.tolist()
    date_column = header[0]
    requested_symbols = [str(symbol) for symbol in symbols]
    available_symbols = [symbol for symbol in requested_symbols if symbol in header]
    usecols = [date_column] + available_symbols

    factor = pd.read_csv(factor_path, usecols=usecols, index_col=0)
    factor.index = pd.to_datetime(factor.index, errors="coerce").strftime("%Y-%m-%d")
    factor = factor.loc[factor.index.notna()]
    factor = factor[~factor.index.duplicated(keep="last")]
    factor = factor.reindex(index=list(dates), columns=requested_symbols)
    return factor.apply(pd.to_numeric, errors="coerce")


def _load_factor_dates(factor_path: Path, years: Optional[Sequence[int]]) -> List[str]:
    """只读取复权因子日期列，用于快速确定目标交易日历。"""
    header = pd.read_csv(factor_path, nrows=0).columns.tolist()
    date_column = header[0]
    date_frame = pd.read_csv(factor_path, usecols=[date_column])
    dates = pd.to_datetime(date_frame[date_column], errors="coerce").dropna()
    if years is not None:
        year_set = set(int(year) for year in years)
        dates = dates[dates.dt.year.isin(year_set)]
    return dates.dt.strftime("%Y-%m-%d").tolist()


def _infer_smoke_target_dates(raw_root: Path, factor_path: Path, factor_dates: Sequence[str]) -> List[str]:
    """Infer a smoke-test trading calendar from the first usable raw symbol."""
    factor_date_set = set(str(date) for date in factor_dates)
    factor_header = set(pd.read_csv(factor_path, nrows=0).columns.tolist())
    for file_path in sorted(raw_root.rglob("data.bz2")):
        symbol = file_path.parent.name
        if symbol not in factor_header:
            continue
        df, _ = _load_cn_symbol_frame(file_path)
        day_info = _classify_symbol_days(df)
        valid_dates = [date for date in day_info["valid_dates"] if date in factor_date_set]
        if valid_dates:
            return valid_dates
    raise RuntimeError("Unable to infer a smoke-test calendar from raw K-line files")


def _select_smoke_symbols_fast(
    raw_root: Path,
    factor_path: Path,
    target_dates: Sequence[str],
    max_stocks: int,
) -> pd.DataFrame:
    """小样本模式下快速挑选严格平衡股票，避免先扫描全市场。"""
    target_date_set = set(target_dates)
    if not target_date_set:
        raise ValueError("小样本预处理没有可用目标日期")

    factor_header = set(pd.read_csv(factor_path, nrows=0).columns.tolist())
    rows: List[Dict[str, Any]] = []
    for file_path in sorted(raw_root.rglob("data.bz2")):
        symbol = file_path.parent.name
        if symbol not in factor_header:
            continue

        df, _ = _load_cn_symbol_frame(file_path)
        day_info = _classify_symbol_days(df)
        valid_dates = [date for date in day_info["valid_dates"] if date in target_date_set]
        if len(valid_dates) != len(target_date_set):
            continue

        rows.append({"symbol": symbol, "file_path": str(file_path.resolve())})
        if len(rows) >= int(max_stocks):
            break

    if len(rows) < int(max_stocks):
        raise RuntimeError(f"只找到 {len(rows)} 只严格平衡股票，少于请求的 {max_stocks} 只")
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _build_adjusted_symbol_returns(
    file_path: Path,
    factor_by_date: pd.Series,
    return_mode: str = "open_close",
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """读取单只股票 K 线，生成后复权后的日内、隔夜和日度收益。"""
    if return_mode not in {"open_close", "close_close"}:
        raise ValueError(f"未知 return_mode: {return_mode}")

    df, clean_stats = _load_cn_symbol_frame(file_path)
    factor_lookup = factor_by_date.to_dict()

    valid_intraday: List[np.ndarray] = []
    overnight_values: List[float] = []
    raw_overnight_values: List[float] = []
    daily_values: List[float] = []
    valid_dates: List[str] = []
    factor_values: List[float] = []
    day_open_values: List[float] = []
    day_close_values: List[float] = []
    raw_day_open_values: List[float] = []
    raw_day_close_values: List[float] = []
    target_date_set = set(str(date) for date in factor_by_date.index)
    observed_days_by_year: Dict[int, int] = {}
    valid_days_by_year: Dict[int, int] = {}

    missing_factor_days = 0
    invalid_grid_days = 0
    bad_price_days = 0
    previous_adjusted_close: Optional[float] = None
    previous_raw_close: Optional[float] = None

    if not df.empty:
        timestamps = df["kline_time"]
        dates = timestamps.to_numpy(dtype="datetime64[D]")
        time_codes = (
            timestamps.dt.hour.to_numpy(dtype=np.int16) * 100
            + timestamps.dt.minute.to_numpy(dtype=np.int16)
        ).astype(np.int32)
        open_raw = df["open"].to_numpy(dtype=np.float64)
        high_raw = df["high"].to_numpy(dtype=np.float64)
        low_raw = df["low"].to_numpy(dtype=np.float64)
        close_raw = df["close"].to_numpy(dtype=np.float64)
        price_ok_rows = df[["open", "high", "low", "close"]].gt(0.0).to_numpy().all(axis=1)

        boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(df)]

        for start, end in zip(starts, ends):
            date_text = str(np.datetime_as_string(dates[start], unit="D"))
            if date_text not in target_date_set:
                continue

            year = int(pd.Timestamp(date_text).year)
            observed_days_by_year[year] = observed_days_by_year.get(year, 0) + 1
            factor_value = factor_lookup.get(date_text)
            factor_ok = bool(pd.notna(factor_value) and np.isfinite(float(factor_value)) and float(factor_value) > 0.0)
            grid_ok = (end - start) == len(CN_5MIN_BAR_TIMES) and np.array_equal(time_codes[start:end], CN_5MIN_BAR_CODES)
            prices_ok = bool(price_ok_rows[start:end].all())

            if not factor_ok:
                missing_factor_days += 1
            if not grid_ok:
                invalid_grid_days += 1
            if not prices_ok:
                bad_price_days += 1
            if not (factor_ok and grid_ok and prices_ok):
                continue

            factor_float = float(factor_value)
            adjusted_open = open_raw[start:end] * factor_float
            adjusted_close = close_raw[start:end] * factor_float

            if return_mode == "open_close":
                intraday = np.log(adjusted_close / adjusted_open)
            else:
                intraday = np.empty(len(adjusted_close), dtype=np.float64)
                intraday[0] = np.log(adjusted_close[0] / adjusted_open[0])
                intraday[1:] = np.log(adjusted_close[1:] / adjusted_close[:-1])

            day_open = float(adjusted_open[0])
            day_close = float(adjusted_close[-1])
            raw_day_open = float(open_raw[start])
            raw_day_close = float(close_raw[end - 1])
            overnight = 0.0 if previous_adjusted_close is None else float(np.log(day_open / previous_adjusted_close))
            raw_overnight = 0.0 if previous_raw_close is None else float(np.log(raw_day_open / previous_raw_close))

            valid_dates.append(date_text)
            valid_days_by_year[year] = valid_days_by_year.get(year, 0) + 1
            factor_values.append(factor_float)
            valid_intraday.append(intraday)
            overnight_values.append(overnight)
            raw_overnight_values.append(raw_overnight)
            daily_values.append(float(intraday.sum() + overnight))
            day_open_values.append(day_open)
            day_close_values.append(day_close)
            raw_day_open_values.append(raw_day_open)
            raw_day_close_values.append(raw_day_close)
            previous_adjusted_close = day_close
            previous_raw_close = raw_day_close

    intraday_array = np.vstack(valid_intraday).astype(np.float64, copy=False) if valid_intraday else np.zeros((0, len(CN_5MIN_BAR_TIMES)), dtype=np.float64)
    overnight_array = np.asarray(overnight_values, dtype=np.float64)
    raw_overnight_array = np.asarray(raw_overnight_values, dtype=np.float64)
    daily_array = np.asarray(daily_values, dtype=np.float64)
    date_codes = _date_codes_from_dates(valid_dates) if valid_dates else np.zeros(0, dtype=np.int32)
    abs_overnight = np.abs(overnight_array)
    abs_raw_overnight = np.abs(raw_overnight_array)

    meta = {
        "version": PREPROCESS_VERSION,
        "symbol": file_path.parent.name,
        "source_file": str(file_path.resolve()),
        "return_mode": return_mode,
        "adjustment": "backward",
        "raw_rows": int(clean_stats["raw_rows"]),
        "clean_rows": int(len(df)),
        "duplicate_rows": int(clean_stats["duplicate_rows"]),
        "n_observed_days": int(sum(observed_days_by_year.values())),
        "n_valid_days": int(len(valid_dates)),
        "n_invalid_days": int(sum(observed_days_by_year.values()) - len(valid_dates)),
        "invalid_grid_days": int(invalid_grid_days),
        "bad_price_days": int(bad_price_days),
        "missing_factor_days": int(missing_factor_days),
        "first_valid_date": valid_dates[0] if valid_dates else None,
        "last_valid_date": valid_dates[-1] if valid_dates else None,
        "valid_dates": valid_dates,
        "valid_days_by_year": {int(k): int(v) for k, v in valid_days_by_year.items()},
        "observed_days_by_year": {int(k): int(v) for k, v in observed_days_by_year.items()},
        "max_abs_raw_overnight": float(abs_raw_overnight.max()) if len(abs_raw_overnight) else 0.0,
        "max_abs_overnight": float(abs_overnight.max()) if len(abs_overnight) else 0.0,
        "suspicious_raw_overnight_count_012": int((abs_raw_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[0]).sum()),
        "suspicious_raw_overnight_count_020": int((abs_raw_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[1]).sum()),
        "suspicious_overnight_count_012": int((abs_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[0]).sum()),
        "suspicious_overnight_count_020": int((abs_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[1]).sum()),
    }
    arrays = {
        "date_codes": date_codes,
        "intraday_returns": intraday_array,
        "overnight_returns": overnight_array,
        "raw_overnight_returns": raw_overnight_array,
        "daily_returns": daily_array,
        "day_open": np.asarray(day_open_values, dtype=np.float64),
        "day_close": np.asarray(day_close_values, dtype=np.float64),
        "raw_day_open": np.asarray(raw_day_open_values, dtype=np.float64),
        "raw_day_close": np.asarray(raw_day_close_values, dtype=np.float64),
        "backward_factor": np.asarray(factor_values, dtype=np.float64),
        "suspicious_overnight_mask_012": (abs_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[0]).astype(np.uint8),
        "suspicious_overnight_mask_020": (abs_overnight > SUSPICIOUS_OVERNIGHT_THRESHOLDS[1]).astype(np.uint8),
    }
    return meta, arrays


def _summarize_processed_universe(universe: pd.DataFrame, proc_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """基于复权后有效日期重新计算样本覆盖率和年度标签。"""
    global_valid_dates = sorted({date for dates in universe["valid_dates"] for date in dates})
    if not global_valid_dates:
        raise RuntimeError("预处理后没有任何完整交易日")

    years = pd.Series(pd.to_datetime(global_valid_dates).year)
    calendar_days_by_year = years.value_counts(sort=False).sort_index().astype(int).to_dict()
    global_dates_by_year = {
        int(year): [date for date in global_valid_dates if pd.Timestamp(date).year == int(year)]
        for year in calendar_days_by_year
    }

    universe = universe.copy()
    for year, year_dates in global_dates_by_year.items():
        universe[f"valid_days_{year}"] = universe["valid_days_by_year"].apply(lambda item: int(item.get(year, item.get(str(year), 0))))
        universe[f"observed_days_{year}"] = universe["observed_days_by_year"].apply(lambda item: int(item.get(year, item.get(str(year), 0))))
        universe[f"coverage_{year}"] = universe[f"valid_days_{year}"] / float(len(year_dates))
        universe[f"is_strict_{year}"] = universe[f"valid_days_{year}"].eq(len(year_dates)) & universe[f"observed_days_{year}"].eq(len(year_dates))
        universe[f"is_near_99_{year}"] = universe[f"coverage_{year}"].ge(0.99) & universe[f"observed_days_{year}"].eq(universe[f"valid_days_{year}"])

    calendar_days = len(global_valid_dates)
    universe["missing_days"] = calendar_days - universe["n_valid_days"]
    universe["coverage_ratio"] = universe["n_valid_days"] / float(calendar_days)
    universe["is_strict_balanced"] = (
        universe["n_valid_days"].eq(calendar_days)
        & universe["n_invalid_days"].eq(0)
        & universe["first_valid_date"].eq(global_valid_dates[0])
        & universe["last_valid_date"].eq(global_valid_dates[-1])
    )
    universe["is_near_balanced_99"] = universe["coverage_ratio"].ge(0.99) & universe["n_invalid_days"].eq(0)

    summary = {
        "data_root": str(proc_root.resolve()),
        "source": "processed_adjusted",
        "adjustment": "backward",
        "total_symbols": int(len(universe)),
        "global_start": global_valid_dates[0],
        "global_end": global_valid_dates[-1],
        "global_calendar_days": int(calendar_days),
        "bars_per_day": len(CN_5MIN_BAR_TIMES),
        "bar_times": list(CN_5MIN_BAR_TIMES),
        "calendar_days_by_year": {int(k): int(v) for k, v in calendar_days_by_year.items()},
        "global_dates": global_valid_dates,
        "global_dates_by_year": global_dates_by_year,
        "strict_balanced_symbols": int(universe["is_strict_balanced"].sum()),
        "near_balanced_99_symbols": int(universe["is_near_balanced_99"].sum()),
    }
    universe.attrs["summary"] = summary
    return universe, summary


def _summarize_symbol_universe(universe: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Attach coverage summary to a raw scan table."""
    summarized, summary = _summarize_processed_universe(universe, data_root)
    summary["source"] = "raw_kline"
    summary["adjustment"] = "none"
    summarized.attrs["summary"] = summary
    return summarized


def _selected_calendar_dates(summary: Dict[str, Any], years: Optional[Sequence[int]]) -> List[str]:
    years = _normalize_years(years)
    if years is None:
        return list(summary["global_dates"])
    by_year = {int(k): list(v) for k, v in summary["global_dates_by_year"].items()}
    selected: List[str] = []
    for year in years:
        selected.extend(by_year[int(year)])
    return selected


def _select_sample_rows(
    universe: pd.DataFrame,
    sample_mode: str,
    years: Optional[Sequence[int]] = None,
    coverage_threshold: float = 0.99,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """Select strict-balanced or 99%-coverage symbols from a summarized universe."""
    if "summary" not in universe.attrs:
        raise ValueError("universe.attrs['summary'] is required before sample selection")
    summary = universe.attrs["summary"]
    years = _normalize_years(years)
    if sample_mode not in {STRICT_BALANCED_SAMPLE, NEAR_BALANCED_99_SAMPLE}:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    if years is None:
        mask = universe["is_strict_balanced"] if sample_mode == STRICT_BALANCED_SAMPLE else universe["is_near_balanced_99"]
    else:
        days_by_year = {int(k): int(v) for k, v in summary["calendar_days_by_year"].items()}
        total_days = int(sum(days_by_year[int(year)] for year in years))
        valid_sum = sum(universe[f"valid_days_{int(year)}"] for year in years)
        observed_sum = sum(universe[f"observed_days_{int(year)}"] for year in years)
        if sample_mode == STRICT_BALANCED_SAMPLE:
            mask = (valid_sum == total_days) & (observed_sum == total_days)
        else:
            mask = (valid_sum / float(total_days) >= coverage_threshold) & (observed_sum == valid_sum)

    selected = universe.loc[mask].sort_values("symbol").copy()
    if max_stocks is not None:
        selected = selected.head(int(max_stocks)).copy()
    selected.attrs["summary"] = summary
    return selected


def _save_symbol_outputs(
    proc_root: Path,
    symbol: str,
    meta: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    compress: bool = False,
) -> None:
    """保存单只股票的复权收益数组和元数据。"""
    npz_path = _symbol_npz_path(proc_root, symbol)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(npz_path, **arrays)
    else:
        np.savez(npz_path, **arrays)
    _write_json(_symbol_meta_path(proc_root, symbol), meta)


def _symbol_output_row(proc_root: Path, symbol: str, raw_file_path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """把单只股票元数据整理成 universe 表的一行。"""
    return {
        "symbol": symbol,
        "file_path": str(_symbol_npz_path(proc_root, symbol).resolve()),
        "raw_file_path": str(raw_file_path.resolve()),
        "n_observed_days": meta["n_observed_days"],
        "n_valid_days": meta["n_valid_days"],
        "n_invalid_days": meta["n_invalid_days"],
        "first_valid_date": meta["first_valid_date"],
        "last_valid_date": meta["last_valid_date"],
        "missing_factor_days": meta["missing_factor_days"],
        "max_abs_raw_overnight": meta["max_abs_raw_overnight"],
        "max_abs_overnight": meta["max_abs_overnight"],
        "suspicious_raw_overnight_count_012": meta["suspicious_raw_overnight_count_012"],
        "suspicious_raw_overnight_count_020": meta["suspicious_raw_overnight_count_020"],
        "suspicious_overnight_count_012": meta["suspicious_overnight_count_012"],
        "suspicious_overnight_count_020": meta["suspicious_overnight_count_020"],
        "valid_dates": meta["valid_dates"],
        "valid_days_by_year": meta["valid_days_by_year"],
        "observed_days_by_year": meta["observed_days_by_year"],
    }


def _preprocess_symbol_task(task: Tuple[str, str, str, List[str], List[float], str, bool]) -> Dict[str, Any]:
    """Windows 多进程 worker：处理并保存一只股票，只返回小型元数据。"""
    proc_root_text, symbol, file_path_text, factor_dates, factor_values, return_mode, compress = task
    proc_root = Path(proc_root_text)
    file_path = Path(file_path_text)
    try:
        factor_by_date = pd.Series(factor_values, index=factor_dates, dtype="float64")
        meta, arrays = _build_adjusted_symbol_returns(file_path, factor_by_date, return_mode=return_mode)
        _save_symbol_outputs(proc_root, symbol, meta, arrays, compress=compress)
        row = _symbol_output_row(proc_root, symbol, file_path, meta)
        return {"ok": True, "symbol": symbol, "row": row, "error": None}
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "row": None, "error": repr(exc)}


def _load_symbol_arrays(proc_root: Path, symbol: str) -> Dict[str, np.ndarray]:
    """读取单只股票的预处理收益数组。"""
    arrays_raw = np.load(_symbol_npz_path(proc_root, symbol), allow_pickle=False)
    return {name: arrays_raw[name] for name in arrays_raw.files}


def _align_symbol_to_panel(
    proc_root: Path,
    symbol: str,
    date_codes: np.ndarray,
    allow_missing: bool,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """按目标日期一次性对齐单只股票收益，供串行和线程面板组装共用。"""
    arrays = _load_symbol_arrays(proc_root, symbol)
    source_codes = arrays["date_codes"].astype(np.int32, copy=False)
    if len(source_codes) == 0:
        if not allow_missing:
            raise ValueError(f"严格平衡面板缺少 {symbol} 的全部目标交易日")
        day_count = len(date_codes)
        bar_count = len(CN_5MIN_BAR_TIMES)
        return (
            symbol,
            np.full(day_count * bar_count, np.nan, dtype=np.float64),
            np.full(day_count, np.nan, dtype=np.float64),
            np.full(day_count, np.nan, dtype=np.float64),
        )
    source_pos = np.searchsorted(source_codes, date_codes)
    matched = (source_pos < len(source_codes)) & (source_codes[source_pos.clip(max=max(len(source_codes) - 1, 0))] == date_codes)
    if not bool(matched.all()) and not allow_missing:
        missing_code = int(date_codes[np.where(~matched)[0][0]])
        raise ValueError(f"严格平衡面板缺少 {symbol} 在 {_date_code_to_text(missing_code)} 的完整数据")

    day_count = len(date_codes)
    bar_count = len(CN_5MIN_BAR_TIMES)
    fill_value = np.nan if allow_missing else 0.0
    intra_col = np.full((day_count, bar_count), fill_value, dtype=np.float64)
    night_col = np.full(day_count, fill_value, dtype=np.float64)
    daily_col = np.full(day_count, fill_value, dtype=np.float64)
    if matched.any():
        target_idx = np.where(matched)[0]
        source_idx = source_pos[matched]
        intra_col[target_idx, :] = arrays["intraday_returns"][source_idx]
        night_col[target_idx] = arrays["overnight_returns"][source_idx]
        daily_col[target_idx] = arrays["daily_returns"][source_idx]
    return symbol, intra_col.reshape(day_count * bar_count), night_col, daily_col


def _write_panel(
    proc_root: Path,
    sample_mode: str,
    panel_name: str,
    selected: pd.DataFrame,
    dates: Sequence[str],
    allow_missing: bool,
    return_mode: str = "open_close",
    panel_workers: int = 1,
) -> Dict[str, Any]:
    """把逐股票收益数组拼成面板文件。"""
    tickers = selected["symbol"].tolist()
    date_codes = _date_codes_from_dates(dates)
    day_count = len(dates)
    bar_count = len(CN_5MIN_BAR_TIMES)
    stock_count = len(tickers)

    fill_value = np.nan if allow_missing else 0.0
    R_intra = np.full((day_count * bar_count, stock_count), fill_value, dtype=np.float64)
    R_night = np.full((day_count, stock_count), fill_value, dtype=np.float64)
    R_daily = np.full((day_count, stock_count), fill_value, dtype=np.float64)

    panel_workers = max(1, int(panel_workers))
    if panel_workers == 1 or stock_count <= 1:
        aligned_columns = [
            _align_symbol_to_panel(proc_root, symbol, date_codes, allow_missing)
            for symbol in tickers
        ]
    else:
        aligned_columns = []
        with ThreadPoolExecutor(max_workers=panel_workers) as executor:
            futures = {
                executor.submit(_align_symbol_to_panel, proc_root, symbol, date_codes, allow_missing): symbol
                for symbol in tickers
            }
            for future in as_completed(futures):
                aligned_columns.append(future.result())

    column_lookup = {symbol: (intra, night, daily) for symbol, intra, night, daily in aligned_columns}
    for col_idx, symbol in enumerate(tickers):
        intra_col, night_col, daily_col = column_lookup[symbol]
        R_intra[:, col_idx] = intra_col
        R_night[:, col_idx] = night_col
        R_daily[:, col_idx] = daily_col

    day_ids = np.repeat(np.arange(day_count), bar_count).astype(np.int32)
    sample_report = {
        "sample_mode": sample_mode,
        "return_mode": return_mode,
        "panel_name": panel_name,
        "adjustment": "backward",
        "n_symbols_selected": int(stock_count),
        "n_days_selected": int(day_count),
        "bars_per_day": int(bar_count),
        "selected_calendar_start": dates[0] if dates else None,
        "selected_calendar_end": dates[-1] if dates else None,
        "target_symbols": tickers,
        "contains_nan": bool(np.isnan(R_intra).any()),
        "panel_workers": int(panel_workers),
    }

    panel_array_dir, meta_path = _panel_paths(proc_root, sample_mode, panel_name)
    panel_array_dir.mkdir(parents=True, exist_ok=True)
    array_files = {
        "R_intra": str((panel_array_dir / "R_intra.npy").relative_to(proc_root)),
        "R_night": str((panel_array_dir / "R_night.npy").relative_to(proc_root)),
        "R_daily": str((panel_array_dir / "R_daily.npy").relative_to(proc_root)),
        "day_ids": str((panel_array_dir / "day_ids.npy").relative_to(proc_root)),
    }
    np.save(panel_array_dir / "R_intra.npy", R_intra, allow_pickle=False)
    np.save(panel_array_dir / "R_night.npy", R_night, allow_pickle=False)
    np.save(panel_array_dir / "R_daily.npy", R_daily, allow_pickle=False)
    np.save(panel_array_dir / "day_ids.npy", day_ids, allow_pickle=False)
    panel_meta = {
        "version": PREPROCESS_VERSION,
        "sample_mode": sample_mode,
        "return_mode": return_mode,
        "panel_name": panel_name,
        "dates": list(dates),
        "tickers": tickers,
        "bar_times": list(CN_5MIN_BAR_TIMES),
        "sample_report": sample_report,
        "storage_format": "npy_dir",
        "array_files": array_files,
    }
    _write_json(meta_path, panel_meta)
    return panel_meta


def _save_metadata_tables(proc_root: Path, universe: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """保存样本表、摘要和复权后公司行为诊断。"""
    metadata_dir = proc_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = ["valid_dates", "valid_days_by_year", "observed_days_by_year"]
    universe.drop(columns=[col for col in drop_cols if col in universe.columns]).to_csv(
        metadata_dir / "universe.csv",
        index=False,
        encoding="utf-8-sig",
    )
    universe.to_pickle(metadata_dir / "universe.pkl")
    _write_json(metadata_dir / "universe_summary.json", summary)

    risk_cols = [
        "symbol",
        "max_abs_raw_overnight",
        "max_abs_overnight",
        "suspicious_raw_overnight_count_012",
        "suspicious_raw_overnight_count_020",
        "suspicious_overnight_count_012",
        "suspicious_overnight_count_020",
        "missing_factor_days",
    ]
    universe[[col for col in risk_cols if col in universe.columns]].to_csv(
        metadata_dir / "corp_action_risk_after_adjustment.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _select_preprocess_symbols(universe: pd.DataFrame, years: Optional[Sequence[int]], max_stocks: Optional[int]) -> pd.DataFrame:
    """选择需要预处理的股票；全量运行不筛选，smoke run 优先保证严格平衡样本可用。"""
    if max_stocks is None:
        return universe.sort_values("symbol").copy()

    strict = _select_sample_rows(universe, STRICT_BALANCED_SAMPLE, years=years, max_stocks=max_stocks)
    near_parts = [_select_sample_rows(universe, NEAR_BALANCED_99_SAMPLE, years=[year], max_stocks=max_stocks) for year in (_normalize_years(years) or sorted(universe.attrs["summary"]["calendar_days_by_year"]))]
    selected_symbols = set(strict["symbol"])
    for part in near_parts:
        selected_symbols.update(part["symbol"])
    return universe.loc[universe["symbol"].isin(selected_symbols)].sort_values("symbol").copy()


def preprocess_cn_data(
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    proc_root: str | Path = DEFAULT_PROC_ROOT,
    years: Optional[Sequence[int]] = None,
    max_stocks: Optional[int] = None,
    refresh: bool = False,
    return_mode: str = "open_close",
    workers: Optional[int] = None,
    panel_workers: Optional[int] = None,
    compress_symbol_returns: bool = False,
) -> Dict[str, Any]:
    """执行完整预处理流程，并返回 manifest。"""
    raw_root = _ensure_path(raw_root)
    factor_path = _ensure_path(factor_path)
    proc_root = _ensure_path(proc_root)
    years = _normalize_years(years)
    workers = _normalize_worker_count(workers)
    panel_workers = _normalize_worker_count(panel_workers)
    manifest_path = proc_root / "manifest.json"
    raw_files = _find_raw_kline_files(raw_root)
    raw_symbol_count = len(raw_files)
    raw_file_state = {
        "count": raw_symbol_count,
        "total_size": int(sum(file_path.stat().st_size for file_path in raw_files)),
        "max_mtime_ns": int(max(file_path.stat().st_mtime_ns for file_path in raw_files)),
    }

    input_signature = {
        "version": PREPROCESS_VERSION,
        "raw_root": str(raw_root),
        "raw_files": raw_file_state,
        "factor": _file_signature(factor_path),
        "years": years,
        "max_stocks": max_stocks,
        "return_mode": return_mode,
        "adjustment": "backward",
        "compress_symbol_returns": bool(compress_symbol_returns),
    }
    input_hash = _signature_hash(input_signature)
    if manifest_path.exists() and not refresh:
        manifest = _load_json(manifest_path)
        if manifest.get("input_hash") == input_hash:
            print(f"[SKIP] 预处理结果已存在且输入未变化: {proc_root}")
            return manifest

    proc_root.mkdir(parents=True, exist_ok=True)
    factor_dates = _load_factor_dates(factor_path, years)
    if max_stocks is not None:
        target_dates = factor_dates if years is not None else _infer_smoke_target_dates(raw_root, factor_path, factor_dates)
        selected_universe = _select_smoke_symbols_fast(
            raw_root=raw_root,
            factor_path=factor_path,
            target_dates=target_dates,
            max_stocks=max_stocks,
        )
    else:
        factor_header = set(pd.read_csv(factor_path, nrows=0).columns.tolist())
        rows_for_all_symbols = [
            {"symbol": file_path.parent.name, "file_path": str(file_path.resolve())}
            for file_path in raw_files
            if file_path.parent.name in factor_header
        ]
        selected_universe = pd.DataFrame(rows_for_all_symbols).sort_values("symbol").reset_index(drop=True)
        target_dates = factor_dates
    if selected_universe.empty:
        raise ValueError("No symbols are available for preprocessing")

    symbols = selected_universe["symbol"].tolist()
    factor_matrix = load_backward_factor_matrix(factor_path, symbols=symbols, dates=target_dates)

    tasks = []
    factor_dates = [str(date) for date in factor_matrix.index.tolist()]
    for row in selected_universe.itertuples(index=False):
        symbol = str(row.symbol)
        factor_values = [float(value) if pd.notna(value) else np.nan for value in factor_matrix[symbol].to_numpy(dtype=np.float64)]
        tasks.append((str(proc_root), symbol, str(Path(row.file_path).resolve()), factor_dates, factor_values, return_mode, bool(compress_symbol_returns)))

    rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    print(f"[INFO] 预处理 {len(tasks)} 只股票，workers={workers}，panel_workers={panel_workers}")
    if workers == 1 or len(tasks) <= 1:
        for symbol_idx, task in enumerate(tasks, start=1):
            print(f"[{symbol_idx}/{len(tasks)}] 预处理 {task[1]}")
            result = _preprocess_symbol_task(task)
            if result["ok"]:
                rows.append(result["row"])
            else:
                failed_rows.append({"symbol": result["symbol"], "error": result["error"]})
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_preprocess_symbol_task, task): task[1] for task in tasks}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                symbol = str(result["symbol"])
                if result["ok"]:
                    rows.append(result["row"])
                    print(f"[{completed}/{len(tasks)}] 完成 {symbol}")
                else:
                    failed_rows.append({"symbol": symbol, "error": result["error"]})
                    print(f"[{completed}/{len(tasks)}] 失败 {symbol}: {result['error']}")

    if failed_rows:
        failed_path = proc_root / "metadata" / "failed_symbols.csv"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed_rows).sort_values("symbol").to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] {len(failed_rows)} 只股票预处理失败，详情见 {failed_path}")
    if not rows:
        raise RuntimeError("所有股票预处理均失败，无法生成面板")

    processed_universe = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    processed_universe, processed_summary = _summarize_processed_universe(processed_universe, proc_root)
    processed_summary["raw_symbol_count"] = int(raw_symbol_count)
    processed_summary["processed_symbol_count"] = int(len(processed_universe))
    _save_metadata_tables(proc_root, processed_universe, processed_summary)

    panel_outputs: List[Dict[str, Any]] = []
    strict_selected = _select_sample_rows(processed_universe, STRICT_BALANCED_SAMPLE, years=None)
    if not strict_selected.empty:
        panel_outputs.append(_write_panel(proc_root, STRICT_BALANCED_SAMPLE, "full", strict_selected, processed_summary["global_dates"], allow_missing=False, return_mode=return_mode, panel_workers=panel_workers))
        for year, year_dates in processed_summary["global_dates_by_year"].items():
            strict_year = _select_sample_rows(processed_universe, STRICT_BALANCED_SAMPLE, years=[int(year)])
            if not strict_year.empty:
                panel_outputs.append(_write_panel(proc_root, STRICT_BALANCED_SAMPLE, f"year_{year}", strict_year, year_dates, allow_missing=False, return_mode=return_mode, panel_workers=panel_workers))

    near_full = _select_sample_rows(processed_universe, NEAR_BALANCED_99_SAMPLE, years=None)
    if not near_full.empty:
        panel_outputs.append(_write_panel(proc_root, NEAR_BALANCED_99_SAMPLE, "full", near_full, processed_summary["global_dates"], allow_missing=True, return_mode=return_mode, panel_workers=panel_workers))
    for year, year_dates in processed_summary["global_dates_by_year"].items():
        near_year = _select_sample_rows(processed_universe, NEAR_BALANCED_99_SAMPLE, years=[int(year)])
        if not near_year.empty:
            panel_outputs.append(_write_panel(proc_root, NEAR_BALANCED_99_SAMPLE, f"year_{year}", near_year, year_dates, allow_missing=True, return_mode=return_mode, panel_workers=panel_workers))

    manifest = {
        "version": PREPROCESS_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_hash": input_hash,
        "input_signature": input_signature,
        "raw_root": str(raw_root),
        "factor_path": str(factor_path),
        "proc_root": str(proc_root),
        "adjustment": "backward",
        "return_mode": return_mode,
        "years": years,
        "max_stocks": max_stocks,
        "raw_symbol_count": int(raw_symbol_count),
        "processed_symbol_count": int(len(processed_universe)),
        "failed_symbol_count": int(len(failed_rows)),
        "workers": int(workers),
        "panel_workers": int(panel_workers),
        "symbol_return_storage": "npz_compressed" if compress_symbol_returns else "npz_uncompressed",
        "strict_balanced_symbols": int(processed_summary["strict_balanced_symbols"]),
        "near_balanced_99_symbols": int(processed_summary["near_balanced_99_symbols"]),
        "summary": processed_summary,
        "panel_outputs": panel_outputs,
        "metadata_files": {
            "universe": "metadata/universe.csv",
            "universe_pickle": "metadata/universe.pkl",
            "universe_summary": "metadata/universe_summary.json",
            "corp_action_risk_after_adjustment": "metadata/corp_action_risk_after_adjustment.csv",
        },
    }
    _write_json(manifest_path, manifest)
    print(f"[OK] 预处理完成: {proc_root}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="预处理中国 A 股 5 分钟 K 线并生成后复权面板。")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="原始 EXTRA_STOCK_A/data.bz2 根目录")
    parser.add_argument("--factor-path", default=str(DEFAULT_FACTOR_PATH), help="后复权因子 CSV，默认 Data/fact_Data/backward_factor.csv")
    parser.add_argument("--proc-root", default=str(DEFAULT_PROC_ROOT), help="预处理输出目录")
    parser.add_argument("--adjustment", default="backward", choices=["backward"], help="当前默认且唯一实现：后复权")
    parser.add_argument("--return-mode", default="open_close", choices=["open_close", "close_close"], help="日内收益构造口径")
    parser.add_argument("--years", nargs="+", type=int, help="只预处理指定年份")
    parser.add_argument("--max-stocks", type=int, help="smoke run 用：限制每类样本股票数量")
    parser.add_argument("--refresh", action="store_true", help="忽略已有 manifest，强制重建")
    parser.add_argument("--workers", type=int, help="逐股票预处理并行进程数；默认 min(cpu_count - 1, 8)")
    parser.add_argument("--panel-workers", type=int, help="面板拼接并行线程数；默认 min(cpu_count - 1, 8)")
    parser.add_argument("--compress-symbol-returns", action="store_true", help="压缩逐股票 .npz 中间文件；更省空间但 IO 更慢")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preprocess_cn_data(
        raw_root=args.raw_root,
        factor_path=args.factor_path,
        proc_root=args.proc_root,
        years=args.years,
        max_stocks=args.max_stocks,
        refresh=args.refresh,
        return_mode=args.return_mode,
        workers=args.workers,
        panel_workers=args.panel_workers,
        compress_symbol_returns=args.compress_symbol_returns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
