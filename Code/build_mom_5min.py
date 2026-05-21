"""Build a 5-minute Carhart-style MOM factor from China A-share K-lines."""

from __future__ import annotations
import argparse
import os
import json
import math
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTOR_PATH = REPO_ROOT / "Data" / "fact_Data" / "backward_factor.csv"
DEFAULT_PROC_ROOT = REPO_ROOT / "Data" / "proc_Data" / "mom_5min"
DEFAULT_RAW_ROOT = REPO_ROOT / "Data" / "kline_Data" / "EXTRA_STOCK_A"

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
CN_5MIN_BAR_CODES = np.array([int(text[:2]) * 100 + int(text[3:5]) for text in CN_5MIN_BAR_TIMES], dtype=np.int32)
SESSION_OPEN_CODES = {930, 1300}
TARGET_SYMBOLS_PER_CHUNK = 64


@dataclass
class SymbolProcessResult:
    symbol: str
    observed_days: int
    valid_days: int
    invalid_grid_days: int
    missing_factor_days: int
    bad_price_days: int
    rows_written: int
    years_written: List[int]
    skipped: bool
    skip_reason: Optional[str] = None


@dataclass
class ChunkProcessResult:
    chunk_id: int
    column_start: int
    column_count: int
    symbols: List[str]
    signal_paths: Dict[int, str]
    ret_paths: Dict[int, str]
    symbol_results: List[SymbolProcessResult]


def _ensure_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _resolve_default_raw_root() -> Path:
    if DEFAULT_RAW_ROOT.exists():
        return DEFAULT_RAW_ROOT.resolve()
    raise FileNotFoundError(
        f"Expected raw kline directory at {DEFAULT_RAW_ROOT} "
        "(Data/kline_Data/EXTRA_STOCK_A), but it does not exist."
    )


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def _normalize_worker_count(workers: Optional[int]) -> int:
    if workers is None:
        return _default_worker_count()
    return max(1, int(workers))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_raw_files(raw_root: Path) -> List[Path]:
    files = sorted(raw_root.rglob("data.bz2"))
    if not files:
        raise FileNotFoundError(f"No data.bz2 files found under {raw_root}")
    return files


def _split_files_into_chunks(files: Sequence[Path], chunk_count: int) -> List[Tuple[int, int, List[Path]]]:
    total = len(files)
    if total == 0:
        return []
    chunk_count = max(1, min(int(chunk_count), total))
    base, remainder = divmod(total, chunk_count)
    chunks: List[Tuple[int, int, List[Path]]] = []
    start = 0
    for chunk_id in range(chunk_count):
        stop = start + base + (1 if chunk_id < remainder else 0)
        if start >= stop:
            break
        chunk_files = list(files[start:stop])
        chunks.append((chunk_id, start, chunk_files))
        start = stop
    return chunks


def _load_cn_symbol_frame(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
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


def _discover_years_from_file_chunk(files: Sequence[Path]) -> List[int]:
    years: set[int] = set()
    for file_path in files:
        raw = pd.read_pickle(file_path, compression="bz2")
        if "kline_time" not in raw.columns:
            raise ValueError(f"{file_path} is missing required column: kline_time")
        timestamps = pd.to_datetime(raw["kline_time"], errors="coerce").dropna()
        if timestamps.empty:
            continue
        years.update(timestamps.dt.year.astype(int).unique().tolist())
    if not years:
        raise ValueError(f"No valid kline_time values found under {files[0].parent.parent if files else 'raw root'}")
    return sorted(years)


def _discover_years_from_raw_files(files: Sequence[Path], workers: Optional[int] = None) -> List[int]:
    if not files:
        raise ValueError("No raw files provided for year discovery")
    worker_count = max(1, min(_normalize_worker_count(workers), len(files)))
    chunk_count = max(worker_count, math.ceil(len(files) / TARGET_SYMBOLS_PER_CHUNK))
    if worker_count == 1:
        return _discover_years_from_file_chunk(files)

    chunks = _split_files_into_chunks(files, chunk_count)
    years: set[int] = set()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_discover_years_from_file_chunk, chunk_files): (chunk_id, start, len(chunk_files))
            for chunk_id, start, chunk_files in chunks
        }
        for future in as_completed(futures):
            chunk_id, start, chunk_size = futures[future]
            try:
                years.update(future.result())
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Year discovery failed for chunk {chunk_id} "
                    f"(columns {start + 1}-{start + chunk_size}): {exc}"
                ) from exc
    if not years:
        raise ValueError("No valid kline_time values found in any raw files")
    return sorted(years)


def _load_factor_dates(factor_path: Path, years: Optional[Sequence[int]]) -> List[str]:
    header = pd.read_csv(factor_path, nrows=0).columns.tolist()
    date_column = header[0]
    date_frame = pd.read_csv(factor_path, usecols=[date_column])
    dates = pd.to_datetime(date_frame[date_column], errors="coerce").dropna()
    if years is not None:
        year_set = set(int(year) for year in years)
        dates = dates[dates.dt.year.isin(year_set)]
    return dates.dt.strftime("%Y-%m-%d").tolist()


def load_backward_factor_matrix(factor_path: str | Path, symbols: Sequence[str], dates: Sequence[str]) -> pd.DataFrame:
    factor_path = _ensure_path(factor_path)
    header = pd.read_csv(factor_path, nrows=0).columns.tolist()
    date_column = header[0]
    requested_symbols = [str(symbol) for symbol in symbols]
    available_symbols = [symbol for symbol in requested_symbols if symbol in header]
    usecols = [date_column] + available_symbols
    factor = pd.read_csv(factor_path, usecols=usecols, index_col=0)
    factor.index = pd.to_datetime(factor.index, errors="coerce")
    factor = factor.loc[factor.index.notna()]
    factor.index = factor.index.strftime("%Y-%m-%d")
    factor = factor[~factor.index.duplicated(keep="last")]
    factor = factor.reindex(index=list(dates), columns=requested_symbols)
    return factor.apply(pd.to_numeric, errors="coerce")


def _build_year_calendars(years: Sequence[int], factor_dates: Sequence[str]) -> Tuple[Dict[int, List[str]], Dict[int, np.ndarray]]:
    dates_by_year: Dict[int, List[str]] = {int(year): [] for year in years}
    year_set = set(int(year) for year in years)
    for date_text in factor_dates:
        year = int(pd.Timestamp(date_text).year)
        if year in year_set:
            dates_by_year[year].append(date_text)
    times_by_year: Dict[int, np.ndarray] = {}
    for year, date_list in dates_by_year.items():
        stamps = [pd.Timestamp(f"{date_text} {time_text}") for date_text in date_list for time_text in CN_5MIN_BAR_TIMES]
        times_by_year[year] = np.array(stamps, dtype="datetime64[ns]")
    return dates_by_year, times_by_year


def _create_memmaps(cache_root: Path, years: Sequence[int], times_by_year: Dict[int, np.ndarray], n_symbols: int) -> Tuple[Dict[int, np.memmap], Dict[int, np.memmap]]:
    signal_maps: Dict[int, np.memmap] = {}
    ret_maps: Dict[int, np.memmap] = {}
    for year in years:
        year_dir = cache_root / f"year_{int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        shape = (len(times_by_year[int(year)]), int(n_symbols))
        signal = np.memmap(year_dir / "signal.dat", mode="w+", dtype="float32", shape=shape)
        ret = np.memmap(year_dir / "ret.dat", mode="w+", dtype="float32", shape=shape)
        signal[:] = np.nan
        ret[:] = np.nan
        signal.flush()
        ret.flush()
        signal_maps[int(year)] = signal
        ret_maps[int(year)] = ret
    return signal_maps, ret_maps


def _day_start_maps(dates_by_year: Dict[int, List[str]]) -> Dict[int, Dict[str, int]]:
    starts: Dict[int, Dict[str, int]] = {}
    bar_count = len(CN_5MIN_BAR_TIMES)
    for year, date_list in dates_by_year.items():
        starts[year] = {date_text: idx * bar_count for idx, date_text in enumerate(date_list)}
    return starts


def _process_symbol_into_cache(
    file_path: Path,
    factor_by_date: pd.Series,
    symbol_idx: int,
    lookback: int,
    skip: int,
    years: Sequence[int],
    day_start_by_year: Dict[int, Dict[str, int]],
    signal_maps: Dict[int, np.memmap],
    ret_maps: Dict[int, np.memmap],
) -> SymbolProcessResult:
    symbol = file_path.parent.name
    df, _ = _load_cn_symbol_frame(file_path)
    year_filter = set(int(year) for year in years)
    timestamps = df["kline_time"]
    dates = timestamps.to_numpy(dtype="datetime64[D]")
    time_codes = (timestamps.dt.hour.to_numpy(dtype=np.int16) * 100 + timestamps.dt.minute.to_numpy(dtype=np.int16)).astype(np.int32)
    close_raw = df["close"].to_numpy(dtype=np.float64)
    price_ok_rows = df[["open", "high", "low", "close"]].gt(0.0).to_numpy().all(axis=1)
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(df)]

    ordered_bar_times: List[pd.Timestamp] = []
    ordered_closes: List[float] = []
    bar_time_to_slot: Dict[pd.Timestamp, Tuple[int, int]] = {}
    observed_days = 0
    valid_days = 0
    invalid_grid_days = 0
    missing_factor_days = 0
    bad_price_days = 0
    rows_written = 0
    years_written: set[int] = set()

    for start, end in zip(starts, ends):
        date_text = str(np.datetime_as_string(dates[start], unit="D"))
        date_ts = pd.Timestamp(date_text)
        year = int(date_ts.year)
        if year not in year_filter:
            continue
        observed_days += 1
        factor_value = factor_by_date.get(date_text)
        factor_ok = bool(pd.notna(factor_value) and np.isfinite(float(factor_value)) and float(factor_value) > 0.0)
        grid_ok = (end - start) == len(CN_5MIN_BAR_TIMES) and np.array_equal(time_codes[start:end], CN_5MIN_BAR_CODES)
        prices_ok = bool(price_ok_rows[start:end].all())
        if not factor_ok:
            missing_factor_days += 1
            continue
        if not grid_ok:
            invalid_grid_days += 1
            continue
        if not prices_ok:
            bad_price_days += 1
            continue

        day_start = day_start_by_year.get(year, {}).get(date_text)
        if day_start is None:
            continue
        valid_days += 1
        years_written.add(year)
        adjusted_close = close_raw[start:end] * float(factor_value)
        for bar_idx, close_value in enumerate(adjusted_close):
            bar_time = pd.Timestamp(f"{date_text} {CN_5MIN_BAR_TIMES[bar_idx]}")
            global_idx = day_start + bar_idx
            ordered_bar_times.append(bar_time)
            ordered_closes.append(float(close_value))
            bar_time_to_slot[bar_time] = (year, global_idx)
            rows_written += 1

    if not ordered_bar_times:
        return SymbolProcessResult(symbol=symbol, observed_days=observed_days, valid_days=valid_days, invalid_grid_days=invalid_grid_days, missing_factor_days=missing_factor_days, bad_price_days=bad_price_days, rows_written=0, years_written=[], skipped=True, skip_reason="no_valid_rows_after_adjustment")

    close_series = np.asarray(ordered_closes, dtype=np.float64)
    signal_values = np.full(len(close_series), np.nan, dtype=np.float64)
    ret_values = np.full(len(close_series), np.nan, dtype=np.float64)
    if len(close_series) > skip + lookback:
        denominator = close_series[: -(skip + lookback)]
        numerator = close_series[lookback : -skip]
        valid = denominator > 0.0
        signal_slice = np.full(len(denominator), np.nan, dtype=np.float64)
        signal_slice[valid] = numerator[valid] / denominator[valid] - 1.0
        signal_values[skip + lookback :] = signal_slice
    if len(close_series) > 1:
        ret_values[1:] = close_series[1:] / close_series[:-1] - 1.0

    for idx, bar_time in enumerate(ordered_bar_times):
        year, global_idx = bar_time_to_slot[bar_time]
        if int(bar_time.strftime("%H%M")) in SESSION_OPEN_CODES:
            ret_values[idx] = np.nan
        signal_maps[year][global_idx, symbol_idx] = np.float32(signal_values[idx])
        ret_maps[year][global_idx, symbol_idx] = np.float32(ret_values[idx])

    return SymbolProcessResult(symbol=symbol, observed_days=observed_days, valid_days=valid_days, invalid_grid_days=invalid_grid_days, missing_factor_days=missing_factor_days, bad_price_days=bad_price_days, rows_written=rows_written, years_written=sorted(years_written), skipped=False, skip_reason=None)


def _process_symbol_chunk_task(
    chunk_id: int,
    column_start: int,
    file_paths: Sequence[str],
    factor_path: str,
    years: Sequence[int],
    factor_dates: Sequence[str],
    lookback: int,
    skip: int,
    times_by_year: Dict[int, np.ndarray],
    day_start_by_year: Dict[int, Dict[str, int]],
    shard_root: str,
) -> ChunkProcessResult:
    chunk_root = Path(shard_root) / f"chunk_{int(chunk_id):04d}"
    chunk_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(path_text) for path_text in file_paths]
    symbols = [path.parent.name for path in paths]
    factor = load_backward_factor_matrix(factor_path=factor_path, symbols=symbols, dates=factor_dates)
    signal_maps, ret_maps = _create_memmaps(cache_root=chunk_root, years=years, times_by_year=times_by_year, n_symbols=len(paths))
    signal_paths = {int(year): str((chunk_root / f"year_{int(year)}" / "signal.dat").resolve()) for year in years}
    ret_paths = {int(year): str((chunk_root / f"year_{int(year)}" / "ret.dat").resolve()) for year in years}
    symbol_results: List[SymbolProcessResult] = []
    try:
        for local_idx, file_path in enumerate(paths):
            factor_by_date = factor[file_path.parent.name] if file_path.parent.name in factor.columns else pd.Series(dtype="float64")
            result = _process_symbol_into_cache(
                file_path=file_path,
                factor_by_date=factor_by_date,
                symbol_idx=local_idx,
                lookback=lookback,
                skip=skip,
                years=years,
                day_start_by_year=day_start_by_year,
                signal_maps=signal_maps,
                ret_maps=ret_maps,
            )
            symbol_results.append(result)
    finally:
        _close_memmaps(signal_maps)
        _close_memmaps(ret_maps)
    return ChunkProcessResult(
        chunk_id=int(chunk_id),
        column_start=int(column_start),
        column_count=int(len(paths)),
        symbols=symbols,
        signal_paths=signal_paths,
        ret_paths=ret_paths,
        symbol_results=symbol_results,
    )


def _merge_chunk_result_into_cache(
    chunk_result: ChunkProcessResult,
    years: Sequence[int],
    times_by_year: Dict[int, np.ndarray],
    signal_maps: Dict[int, np.memmap],
    ret_maps: Dict[int, np.memmap],
) -> None:
    column_start = int(chunk_result.column_start)
    column_stop = column_start + int(chunk_result.column_count)
    for year in years:
        year = int(year)
        shard_shape = (len(times_by_year[year]), int(chunk_result.column_count))
        signal_shard = np.memmap(chunk_result.signal_paths[year], mode="r", dtype="float32", shape=shard_shape)
        ret_shard = np.memmap(chunk_result.ret_paths[year], mode="r", dtype="float32", shape=shard_shape)
        try:
            signal_maps[year][:, column_start:column_stop] = signal_shard
            ret_maps[year][:, column_start:column_stop] = ret_shard
            signal_maps[year].flush()
            ret_maps[year].flush()
        finally:
            base = getattr(signal_shard, "_mmap", None)
            if base is not None:
                base.close()
            base = getattr(ret_shard, "_mmap", None)
            if base is not None:
                base.close()


def _process_symbol_files_serial(
    files: Sequence[Path],
    factor: pd.DataFrame,
    years: Sequence[int],
    lookback: int,
    skip: int,
    day_start_by_year: Dict[int, Dict[str, int]],
    signal_maps: Dict[int, np.memmap],
    ret_maps: Dict[int, np.memmap],
) -> List[SymbolProcessResult]:
    symbol_results: List[SymbolProcessResult] = []
    for symbol_idx, file_path in enumerate(files):
        symbol = file_path.parent.name
        print(f"[INFO] processing {symbol_idx + 1}/{len(files)} {symbol}")
        factor_by_date = factor[symbol] if symbol in factor.columns else pd.Series(dtype="float64")
        result = _process_symbol_into_cache(
            file_path=file_path,
            factor_by_date=factor_by_date,
            symbol_idx=symbol_idx,
            lookback=lookback,
            skip=skip,
            years=years,
            day_start_by_year=day_start_by_year,
            signal_maps=signal_maps,
            ret_maps=ret_maps,
        )
        symbol_results.append(result)
    return symbol_results


def _process_symbol_files_parallel(
    files: Sequence[Path],
    factor_path: Path,
    years: Sequence[int],
    factor_dates: Sequence[str],
    lookback: int,
    skip: int,
    workers: int,
    cache_root: Path,
    times_by_year: Dict[int, np.ndarray],
    day_start_by_year: Dict[int, Dict[str, int]],
    signal_maps: Dict[int, np.memmap],
    ret_maps: Dict[int, np.memmap],
) -> List[SymbolProcessResult]:
    worker_count = max(1, min(int(workers), len(files)))
    chunk_count = max(worker_count, math.ceil(len(files) / TARGET_SYMBOLS_PER_CHUNK))
    chunks = _split_files_into_chunks(files, chunk_count)
    if worker_count == 1 or len(chunks) == 1:
        factor = load_backward_factor_matrix(factor_path=factor_path, symbols=[path.parent.name for path in files], dates=factor_dates)
        return _process_symbol_files_serial(
            files=files,
            factor=factor,
            years=years,
            lookback=lookback,
            skip=skip,
            day_start_by_year=day_start_by_year,
            signal_maps=signal_maps,
            ret_maps=ret_maps,
        )

    shard_root = cache_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    chunk_results: List[ChunkProcessResult] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _process_symbol_chunk_task,
                chunk_id,
                start,
                [str(path) for path in chunk_files],
                str(factor_path),
                [int(year) for year in years],
                list(factor_dates),
                int(lookback),
                int(skip),
                times_by_year,
                day_start_by_year,
                str(shard_root),
            ): (chunk_id, start, len(chunk_files))
            for chunk_id, start, chunk_files in chunks
        }
        completed = 0
        for future in as_completed(futures):
            chunk_id, start, chunk_size = futures[future]
            completed += 1
            try:
                chunk_result = future.result()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Parallel processing failed for chunk {chunk_id} "
                    f"(columns {start + 1}-{start + chunk_size}): {exc}"
                ) from exc
            print(
                f"[INFO] completed chunk {completed}/{len(chunks)} "
                f"(columns {chunk_result.column_start + 1}-{chunk_result.column_start + chunk_result.column_count})"
            )
            _merge_chunk_result_into_cache(chunk_result, years, times_by_year, signal_maps, ret_maps)
            chunk_results.append(chunk_result)

    chunk_results.sort(key=lambda result: result.column_start)
    symbol_results: List[SymbolProcessResult] = []
    for chunk_result in chunk_results:
        symbol_results.extend(chunk_result.symbol_results)
    return symbol_results


def _aggregate_year_from_cache(
    times: np.ndarray,
    signal_map: np.memmap,
    ret_map: np.memmap,
    winner_pct: float,
    loser_pct: float,
    min_stocks: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    signal = np.asarray(signal_map, dtype=np.float32)
    ret = np.asarray(ret_map, dtype=np.float32)
    for i, kline_time in enumerate(pd.to_datetime(times)):
        sig_row = signal[i]
        ret_row = ret[i]
        valid_mask = np.isfinite(sig_row) & np.isfinite(ret_row)
        n_stocks = int(valid_mask.sum())
        if n_stocks < min_stocks:
            continue
        sig_valid = sig_row[valid_mask]
        ret_valid = ret_row[valid_mask]
        n_winners = max(1, int(math.ceil(n_stocks * winner_pct)))
        n_losers = max(1, int(math.ceil(n_stocks * loser_pct)))
        if n_winners + n_losers > n_stocks:
            continue
        winner_idx = np.argpartition(sig_valid, n_stocks - n_winners)[-n_winners:]
        loser_idx = np.argpartition(sig_valid, n_losers - 1)[:n_losers]
        winner_ret = float(np.mean(ret_valid[winner_idx]))
        loser_ret = float(np.mean(ret_valid[loser_idx]))
        rows.append({"kline_time": kline_time, "MOM": winner_ret - loser_ret, "winner_ret": winner_ret, "loser_ret": loser_ret, "n_stocks": n_stocks, "n_winners": int(n_winners), "n_losers": int(n_losers)})
    if not rows:
        return pd.DataFrame(columns=["kline_time", "MOM", "winner_ret", "loser_ret", "n_stocks", "n_winners", "n_losers"])
    return pd.DataFrame(rows).sort_values("kline_time").reset_index(drop=True)


def _close_memmaps(memmaps: Dict[int, np.memmap]) -> None:
    for mmap_obj in memmaps.values():
        mmap_obj.flush()
        base = getattr(mmap_obj, "_mmap", None)
        if base is not None:
            base.close()


def _build_metadata(
    args: argparse.Namespace,
    raw_root: Path,
    factor_path: Path,
    files: Sequence[Path],
    years: Sequence[int],
    symbol_results: Sequence[SymbolProcessResult],
    factor_date_count: int,
    factor_symbol_count: int,
    worker_count: int,
) -> Dict[str, Any]:
    parallel_strategy = "process_pool_symbol_shards" if int(worker_count) > 1 else "single_process"
    return {
        "raw_root": str(raw_root),
        "factor_path": str(factor_path),
        "proc_root": str(_ensure_path(args.proc_root)),
        "lookback_bars": int(args.lookback_bars),
        "skip_bars": int(args.skip_bars),
        "winner_pct": float(args.winner_pct),
        "loser_pct": float(args.loser_pct),
        "min_stocks": int(args.min_stocks),
        "years": [int(year) for year in years],
        "run_mode": "full_universe_default",
        "workers": int(worker_count),
        "parallel_strategy": parallel_strategy,
        "raw_symbol_files": int(len(files)),
        "factor_dates": int(factor_date_count),
        "factor_symbols_loaded": int(factor_symbol_count),
        "processed_symbols": int(sum(1 for r in symbol_results if not r.skipped)),
        "skipped_symbols": int(sum(1 for r in symbol_results if r.skipped)),
        "symbols": [
            {"symbol": result.symbol, "observed_days": result.observed_days, "valid_days": result.valid_days, "invalid_grid_days": result.invalid_grid_days, "missing_factor_days": result.missing_factor_days, "bad_price_days": result.bad_price_days, "rows_written": result.rows_written, "years_written": result.years_written, "skipped": result.skipped, "skip_reason": result.skip_reason}
            for result in symbol_results
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 5-minute momentum factor from the China A-share 5-minute K-line library.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Raw kline root directory")
    parser.add_argument("--factor-path", default=str(DEFAULT_FACTOR_PATH), help="Backward factor CSV path")
    parser.add_argument("--proc-root", default=str(DEFAULT_PROC_ROOT), help="Output directory for MOM factor files")
    parser.add_argument("--lookback-bars", type=int, default=48, help="Signal lookback in 5-minute bars")
    parser.add_argument("--skip-bars", type=int, default=1, help="Number of most recent bars skipped in the signal")
    parser.add_argument("--winner-pct", type=float, default=0.30, help="Top percentile used as winners")
    parser.add_argument("--loser-pct", type=float, default=0.30, help="Bottom percentile used as losers")
    parser.add_argument("--min-stocks", type=int, default=5, help="Minimum valid cross-sectional stock count per timestamp")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count; default min(cpu_count - 1, 8)")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.winner_pct < 1.0):
        raise ValueError("--winner-pct must be between 0 and 1")
    if not (0.0 < args.loser_pct < 1.0):
        raise ValueError("--loser-pct must be between 0 and 1")
    if args.lookback_bars <= 0:
        raise ValueError("--lookback-bars must be positive")
    if args.skip_bars <= 0:
        raise ValueError("--skip-bars must be positive")
    if args.min_stocks <= 0:
        raise ValueError("--min-stocks must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    raw_root = _resolve_default_raw_root() if Path(args.raw_root) == DEFAULT_RAW_ROOT else _ensure_path(args.raw_root)
    factor_path = _ensure_path(args.factor_path)
    proc_root = _ensure_path(args.proc_root)
    proc_root.mkdir(parents=True, exist_ok=True)

    files = _find_raw_files(raw_root)
    symbols = [path.parent.name for path in files]
    worker_count = max(1, min(_normalize_worker_count(args.workers), len(files)))
    years = _discover_years_from_raw_files(files, workers=worker_count)
    factor_dates = _load_factor_dates(factor_path, years)
    dates_by_year, times_by_year = _build_year_calendars(years, factor_dates)
    day_start_by_year = _day_start_maps(dates_by_year)
    factor: Optional[pd.DataFrame] = None
    if worker_count == 1:
        factor = load_backward_factor_matrix(factor_path=factor_path, symbols=symbols, dates=factor_dates)

    with tempfile.TemporaryDirectory(prefix="mom5m_", dir=str(proc_root)) as tmp_dir:
        cache_root = Path(tmp_dir)
        signal_maps, ret_maps = _create_memmaps(cache_root=cache_root, years=years, times_by_year=times_by_year, n_symbols=len(symbols))
        csv_path = proc_root / "mom_factor_5min.csv"
        pkl_path = proc_root / "mom_factor_5min.pkl"
        pq_path = proc_root / "mom_factor_5min.parquet"
        metadata_path = proc_root / "metadata.json"
        factor_df = pd.DataFrame(columns=["kline_time", "MOM", "winner_ret", "loser_ret", "n_stocks", "n_winners", "n_losers"])
        try:
            if worker_count == 1:
                assert factor is not None
                symbol_results = _process_symbol_files_serial(
                    files=files,
                    factor=factor,
                    years=years,
                    lookback=args.lookback_bars,
                    skip=args.skip_bars,
                    day_start_by_year=day_start_by_year,
                    signal_maps=signal_maps,
                    ret_maps=ret_maps,
                )
            else:
                symbol_results = _process_symbol_files_parallel(
                    files=files,
                    factor_path=factor_path,
                    years=years,
                    factor_dates=factor_dates,
                    lookback=args.lookback_bars,
                    skip=args.skip_bars,
                    workers=worker_count,
                    cache_root=cache_root,
                    times_by_year=times_by_year,
                    day_start_by_year=day_start_by_year,
                    signal_maps=signal_maps,
                    ret_maps=ret_maps,
                )

            print("[INFO] aggregating yearly cross-sections ...")
            factor_frames = [
                _aggregate_year_from_cache(times=times_by_year[year], signal_map=signal_maps[year], ret_map=ret_maps[year], winner_pct=args.winner_pct, loser_pct=args.loser_pct, min_stocks=args.min_stocks)
                for year in years
            ]
            factor_frames = [frame for frame in factor_frames if not frame.empty]
            factor_df = pd.concat(factor_frames, ignore_index=True) if factor_frames else factor_df
            factor_df = factor_df.drop_duplicates(subset=["kline_time"], keep="last").sort_values("kline_time").reset_index(drop=True)

            factor_df.to_csv(csv_path, index=False)
            factor_df.to_pickle(pkl_path)
            try:
                factor_df.to_parquet(pq_path, index=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] parquet skipped: {exc}")

            metadata = _build_metadata(
                args=args,
                raw_root=raw_root,
                factor_path=factor_path,
                files=files,
                years=years,
                symbol_results=symbol_results,
                factor_date_count=len(factor_dates),
                factor_symbol_count=len(symbols),
                worker_count=worker_count,
            )
            _write_json(metadata_path, metadata)
        finally:
            _close_memmaps(signal_maps)
            _close_memmaps(ret_maps)

    print(f"[OK ] factor rows = {len(factor_df):,}")
    if not factor_df.empty:
        print(factor_df.describe(percentiles=[0.01, 0.5, 0.99]).round(6))
    print(f"[OK ] saved {csv_path}")
    print(f"[OK ] saved {pkl_path}")
    if pq_path.exists():
        print(f"[OK ] saved {pq_path}")
    print(f"[OK ] saved {metadata_path}")


if __name__ == "__main__":
    main()
