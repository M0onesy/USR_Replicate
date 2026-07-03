from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# 论文保真选项（env 驱动；这些都只影响 paper_tail“尾部/视图”层，复用既有
# 重结果缓存即可生效，不会让 replication_result_*.pkl 失效）。
# RunConfig 的同名字段可通过 RunConfig.export_fidelity_env() 写入这些 env。
# ======================================================================
def _fid_env(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def _industry_info_filename() -> str:
    # 新版 11 桶映射放到 external/.../industry/ 下后，把文件名指过去即可。
    return _fid_env("PELGER_INDUSTRY_INFO_FILENAME", "stock_full_info_with_std_industry.csv")


def _industry_info_filename() -> str:
    return _fid_env("PELGER_INDUSTRY_INFO_FILENAME", "stock_full_info_std_industry_final.csv")


def _industry_mapping_filename() -> str:
    return _fid_env("PELGER_INDUSTRY_MAPPING_FILENAME", "行业映射表_终版.csv")


def _strict_final_export() -> bool:
    return _fid_env("PELGER_STRICT_FINAL_EXPORT", "0").lower() not in {"0", "false", "no"}


def _industry_frozen() -> Optional[List[str]]:
    """P5/D1：事先冻结的行业因子桶（std_industry，逗号分隔）。空 = 暂用占位自动挑选。"""
    raw = _fid_env("PELGER_INDUSTRY_FROZEN", "")
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or None


def _annualization_days() -> int:
    try:
        return int(_fid_env("PELGER_ANNUALIZATION_DAYS", "252"))
    except Exception:
        return 252


def _ffc_mom_mode() -> str:
    """P6/D4：'carhart_daily'（自建日频 12-1 月）或 'legacy_hf'（旧高频 1 日动量）。"""
    return _fid_env("PELGER_FFC_MOM_MODE", "carhart_daily").lower()


def _size_value_full_market() -> bool:
    """P9/D5：2×3 是否用全市场 symbol_returns 重建。"""
    return _fid_env("PELGER_SIZE_VALUE_FULL_MARKET", "1").lower() not in {"0", "false", "no"}


def _size_value_start() -> Optional[pd.Timestamp]:
    """P10/D5：size/value 与分段 FFC 的样本起点（2012 账面缺失 -> 固定 2014-07-01）。"""
    raw = _fid_env("PELGER_SIZE_VALUE_START", "2014-07-01")
    try:
        return pd.Timestamp(raw) if raw else None
    except Exception:
        return None


# PAPER_TAIL_VERSION 提升以触发尾部缓存重建（应用本批论文保真修复）。
PAPER_TAIL_VERSION = 6
PAPER_TAIL_ALGORITHM_VERSION = "paper_faithful_v4_submission"
PORTFOLIO_ORDER = ["SL", "SM", "SH", "BL", "BM", "BH"]
# N7：Figure 12 分组标题里的行业/规模组合数改为动态填充（新映射 11 桶）。
GROUP_TITLES = {
    "balanced_panel_individual_stocks": "Balanced Panel Individual Stocks",
    "all_stocks": "All Stocks",
    "industry_portfolios": "{n_industry} Industry Portfolios",
    "size_value_portfolios": "6 Size/Value Portfolios",
}
FIG13_FACTORSET_ORDER = [
    "Continuous PCA",
    "Continuous PCA (unbalanced, yearly aligned)",
    "FFC 4-factor",
]
SEGMENT_ORDER = ["intraday", "overnight", "daily"]
OFFICIAL_FFC_FACTORS = ["MKT_excess", "SMB", "HML", "MOM"]
FIGURE12_MIN_ALL_STOCK_OBS = 60
SHORT_LABEL_MAP = {
    "公用事业与交运": "公用/交运",
    "可选消费与服务": "可选消费",
    "房地产与建筑": "地产/建筑",
    "电力设备与新能源": "电新",
    "基础化工": "基础化工",
    "钢铁有色": "钢铁/有色",
    "机械设备": "机械设备",
    "食品饮料": "食饮",
    "传媒通信与计算机": "传媒/通信/计算机",
    "农林牧渔": "农林牧渔",
    "石油石化与煤炭": "油气/煤炭",
    "汽车": "汽车",
    "医药生物": "医药",
    "大金融": "大金融",
    "周期资源": "周期资源",
}



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_external_data_root() -> Path:
    return _repo_root() / "Data" / "external_Data" / "pelger_tail"


def default_paper_tail_root(proc_root: str | Path | None = None) -> Path:
    if proc_root is None:
        return _repo_root() / "Data" / "proc_Data" / "pelger_cn_adjusted" / "paper_tail"
    return Path(proc_root).resolve() / "paper_tail"


def _ensure_path(value: str | Path | None, *, default: Path) -> Path:
    if value is None:
        return default.resolve()
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(path)


def _safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    encodings = [kwargs.pop("encoding", None), "utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_exc: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:  # pragma: no cover - fallback path
            last_exc = exc
            continue
    raise RuntimeError(f"Unable to read CSV file: {path}") from last_exc


def _find_file(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Expected file {filename!r} under {root}")
    return matches[0]


def _normalize_code(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.upper()
    out = out.str.replace(r"\.0+$", "", regex=True)
    out = out.str.replace(r"\.XSHE$", ".SZ", regex=True)
    out = out.str.replace(r"\.XSHG$", ".SH", regex=True)
    digits_only = out.str.fullmatch(r"\d+", na=False)
    out.loc[digits_only] = out.loc[digits_only].str.zfill(6)
    needs_suffix = out.str.fullmatch(r"\d{6}", na=False)
    out.loc[needs_suffix] = np.where(
        out.loc[needs_suffix].str.startswith(("6", "9")),
        out.loc[needs_suffix] + ".SH",
        out.loc[needs_suffix] + ".SZ",
    )
    return out


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D matrix for weighted averaging.")
    valid = np.isfinite(values)
    weighted_sum = np.nansum(values * weights[None, :], axis=1)
    active_weight = np.sum(valid * weights[None, :], axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    good = active_weight > 0
    out[good] = weighted_sum[good] / active_weight[good]
    return out


def _sha1_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_record(path: Path, *, base: Optional[Path] = None) -> Dict[str, Any]:
    stat = path.stat()
    rel = path.relative_to(base) if base is not None else path
    return {
        "path": str(rel).replace("\\", "/"),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _preferred_rf_file(external_root: Path) -> Path:
    env_path = os.environ.get("PELGER_RF_FILE", "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(external_root) / "factors" / "rf" / "risk_free.csv")
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "No RF csv file found. Expected Data/external_Data/pelger_tail/factors/rf/risk_free.csv "
        f"or an explicit PELGER_RF_FILE. Checked: {', '.join(str(path) for path in candidates)}"
    )


def _discovered_paths(proc_root: Path, external_root: Path) -> Dict[str, Path]:
    reference_root = external_root / "size_value" / "reference"
    return {
        "proc_manifest": proc_root / "manifest.json",
        "universe_summary": proc_root / "metadata" / "universe_summary.json",
        "mom_daily": proc_root.parent / "mom_5min" / "mom_factor_5min.csv",
        "ff3": _find_file(external_root / "factors" / "ff3", "STK_MKT_THRFACDAY.csv"),
        "ff5": _find_file(external_root / "factors" / "ff5", "STK_MKT_FIVEFACDAY.csv"),
        "rf": _preferred_rf_file(external_root),
        "industry_info": external_root / "industry" / _industry_info_filename(),
        "industry_mapping": external_root / "industry" / _industry_mapping_filename(),
        "size_value_assignments": _find_file(reference_root, "size_value_2x3_assignments.csv"),
        "size_value_breakpoints": _find_file(reference_root, "size_value_2x3_breakpoints.csv"),
        "size_value_reference_vw": _find_file(reference_root, "size_value_2x3_daily_returns_value_weighted.csv"),
        "size_value_reference_ew": _find_file(reference_root, "size_value_2x3_daily_returns_equal_weighted.csv"),
        "size_value_reference_long": _find_file(reference_root, "size_value_2x3_daily_returns_long.csv"),
    }


def _raw_mcap_files(external_root: Path) -> List[Path]:
    raw_root = external_root / "size_value" / "raw"
    files = sorted(raw_root.rglob("TRD_Dalyr*.csv"))
    if not files:
        raise FileNotFoundError(f"No TRD_Dalyr*.csv files found under {raw_root}")
    return files


def _paper_tail_output_paths(root: Path) -> Dict[str, Path]:
    return {
        "manifest": root / "manifest.json",
        "figure12": root / "figures" / "figure12_data.csv",
        "figure13": root / "figures" / "figure13_data.csv",
        "pricing_industry": root / "figures" / "pricing_industry.csv",
        "pricing_size_value": root / "figures" / "pricing_size_value.csv",
        "table_iii": root / "tables" / "table_iii.csv",
        "table_v": root / "tables" / "table_v.csv",
        "industry_assets": root / "assets" / "industry_portfolios.csv",
        "size_value_assets": root / "assets" / "size_value_portfolios.csv",
        "ffc_external": root / "factors" / "ffc_external_daily.csv",
        "ffc_segmented_raw": root / "factors" / "ffc_segmented_raw_returns.csv",
        "ffc_segmented": root / "factors" / "ffc_segmented_returns.csv",
        "industry_factors": root / "factors" / "industry_factor_returns.csv",
        "industry_selection": root / "diagnostics" / "industry_selection.json",
        "factor_matrix_diagnostics": root / "diagnostics" / "factor_matrix_diagnostics.json",
        "figure12_filter": root / "diagnostics" / "figure12_all_stocks_filter.json",
        "figure13_alignment": root / "diagnostics" / "figure13_yearly_alignment.json",
        "validation_size_value": root / "validation" / "size_value_daily_parity.csv",
        "validation_size_value_summary": root / "validation" / "size_value_daily_parity_summary.json",
        "validation_ffc": root / "validation" / "ffc_daily_validation.csv",
        "validation_ffc_summary": root / "validation" / "ffc_daily_validation_summary.json",
        "validation_ffc_segment_reconciliation": root / "validation" / "ffc_segment_reconciliation.csv",
    }


def _scope_info(result: Any, proc_root: Path, external_root: Path, weighting: str) -> Dict[str, Any]:
    panel = result.panel
    display_k = 0
    if getattr(result.pipeline, "pca_cont_display", None) is not None:
        display_k = int(result.pipeline.pca_cont_display.Lambda.shape[1])
    return {
        "paper_tail_version": PAPER_TAIL_VERSION,
        "paper_tail_algorithm_version": PAPER_TAIL_ALGORITHM_VERSION,
        "proc_root": str(proc_root),
        "external_data_root": str(external_root),
        "paper_tail_weighting": weighting,
        "industry_info_filename": _industry_info_filename(),
        "industry_mapping_filename": _industry_mapping_filename(),
        "strict_final_export": bool(_strict_final_export()),
        "panel_start": str(panel.dates[0].date()) if panel.dates else None,
        "panel_end": str(panel.dates[-1].date()) if panel.dates else None,
        "panel_days": int(panel.D),
        "panel_symbols": int(panel.N),
        "display_factor_count": int(display_k),
        "sample_mode": str(getattr(panel, "sample_mode", "")),
        "requested_return_mode": str(getattr(panel, "requested_return_mode", "")),
    }


def _load_payload(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    outputs = _paper_tail_output_paths(root)
    payload = {
        "manifest": dict(manifest),
        "paths": {key: str(path) for key, path in outputs.items()},
        "figure12_data": pd.read_csv(outputs["figure12"], parse_dates=["date"]),
        "figure13_data": pd.read_csv(outputs["figure13"], parse_dates=["date"]),
        "pricing_industry": pd.read_csv(outputs["pricing_industry"]),
        "pricing_size_value": pd.read_csv(outputs["pricing_size_value"]),
        "table_iii": pd.read_csv(outputs["table_iii"]),
        "table_v": pd.read_csv(outputs["table_v"]),
        "industry_assets": pd.read_csv(outputs["industry_assets"], parse_dates=["date"]),
        "size_value_assets": pd.read_csv(outputs["size_value_assets"], parse_dates=["date"]),
        "ffc_external": pd.read_csv(outputs["ffc_external"], parse_dates=["date"]),
        "ffc_external_daily": pd.read_csv(outputs["ffc_external"], parse_dates=["date"]),
        "ffc_segmented_raw": pd.read_csv(outputs["ffc_segmented_raw"], parse_dates=["date"]),
        "ffc_segmented": pd.read_csv(outputs["ffc_segmented"], parse_dates=["date"]),
        "industry_factors": pd.read_csv(outputs["industry_factors"], parse_dates=["date"]),
        "industry_selection": _load_json(outputs["industry_selection"]),
        "diagnostics": {
            "factor_matrix_diagnostics": _load_json(outputs["factor_matrix_diagnostics"]),
            "figure12_all_stocks_filter": _load_json(outputs["figure12_filter"]),
            "figure13_yearly_alignment": _load_json(outputs["figure13_alignment"]),
        },
        "validation": {
            "size_value_daily_parity": pd.read_csv(outputs["validation_size_value"], parse_dates=["date"]),
            "size_value_daily_parity_summary": _load_json(outputs["validation_size_value_summary"]),
            "ffc_daily_validation": pd.read_csv(outputs["validation_ffc"], parse_dates=["date"]),
            "ffc_daily_validation_summary": _load_json(outputs["validation_ffc_summary"]),
            "ffc_segment_reconciliation": pd.read_csv(outputs["validation_ffc_segment_reconciliation"], parse_dates=["date"]),
        },
    }
    return payload


def _manifest_is_compatible(
    manifest: Mapping[str, Any],
    *,
    external_root: Path,
    paper_tail_root: Path,
    weighting: str,
    scope_signature: Optional[str] = None,
) -> bool:
    if manifest.get("paper_tail_version") != PAPER_TAIL_VERSION:
        return False
    if manifest.get("paper_tail_algorithm_version") != PAPER_TAIL_ALGORITHM_VERSION:
        return False
    if str(manifest.get("external_data_root")) != str(external_root):
        return False
    if str(manifest.get("paper_tail_root")) != str(paper_tail_root):
        return False
    if manifest.get("paper_tail_weighting") != weighting:
        return False
    if scope_signature is not None and manifest.get("scope_signature") != scope_signature:
        return False

    outputs = _paper_tail_output_paths(paper_tail_root)
    expected_outputs = {key for key in outputs if key != "manifest"}
    recorded_outputs = set(manifest.get("outputs", {}).keys())
    return recorded_outputs == expected_outputs


def _discover_or_build_payload(
    result: Any,
    *,
    proc_root: Path,
    external_root: Path,
    paper_tail_root: Path,
    weighting: str,
) -> Dict[str, Any]:
    paper_tail_root.mkdir(parents=True, exist_ok=True)
    paths = _paper_tail_output_paths(paper_tail_root)
    discovered = _discovered_paths(proc_root, external_root)
    mcap_files = _raw_mcap_files(external_root)
    scope = _scope_info(result, proc_root, external_root, weighting)
    source_files = [discovered[key] for key in sorted(discovered.keys())] + mcap_files
    source_records = [_file_record(path, base=_repo_root()) for path in source_files]
    signature_payload = {
        "paper_tail_version": PAPER_TAIL_VERSION,
        "paper_tail_algorithm_version": PAPER_TAIL_ALGORITHM_VERSION,
        "external_data_root": str(external_root),
        "paper_tail_root": str(paper_tail_root),
        "paper_tail_weighting": weighting,
        "scope": scope,
        "source_files": source_records,
    }
    signature = _sha1_payload(signature_payload)

    if paths["manifest"].exists():
        try:
            manifest = _load_json(paths["manifest"])
            expected = manifest.get("outputs", {})
            expected_ok = all((paper_tail_root / rel_path).exists() for rel_path in expected.values())
            if _manifest_is_compatible(
                manifest,
                external_root=external_root,
                paper_tail_root=paper_tail_root,
                weighting=weighting,
                scope_signature=signature,
            ) and expected_ok:
                return _load_payload(paper_tail_root, manifest)
        except Exception:
            pass

    payload = _build_payload(
        result,
        proc_root=proc_root,
        external_root=external_root,
        paper_tail_root=paper_tail_root,
        weighting=weighting,
        discovered=discovered,
        mcap_files=mcap_files,
        scope=scope,
        scope_signature=signature,
    )

    outputs = _paper_tail_output_paths(paper_tail_root)
    _write_csv(outputs["figure12"], payload["figure12_data"])
    _write_csv(outputs["figure13"], payload["figure13_data"])
    _write_csv(outputs["pricing_industry"], payload["pricing_industry"])
    _write_csv(outputs["pricing_size_value"], payload["pricing_size_value"])
    _write_csv(outputs["table_iii"], payload["table_iii"])
    _write_csv(outputs["table_v"], payload["table_v"])
    _write_csv(outputs["industry_assets"], payload["industry_assets"])
    _write_csv(outputs["size_value_assets"], payload["size_value_assets"])
    _write_csv(outputs["ffc_external"], payload["ffc_external"])
    _write_csv(outputs["ffc_segmented_raw"], payload["ffc_segmented_raw"])
    _write_csv(outputs["ffc_segmented"], payload["ffc_segmented"])
    _write_csv(outputs["industry_factors"], payload["industry_factors"])
    _write_json(outputs["industry_selection"], payload["industry_selection"])
    _write_json(outputs["factor_matrix_diagnostics"], payload["diagnostics"]["factor_matrix_diagnostics"])
    _write_json(outputs["figure12_filter"], payload["diagnostics"]["figure12_all_stocks_filter"])
    _write_json(outputs["figure13_alignment"], payload["diagnostics"]["figure13_yearly_alignment"])
    _write_csv(outputs["validation_size_value"], payload["validation"]["size_value_daily_parity"])
    _write_json(outputs["validation_size_value_summary"], payload["validation"]["size_value_daily_parity_summary"])
    _write_csv(outputs["validation_ffc"], payload["validation"]["ffc_daily_validation"])
    _write_json(outputs["validation_ffc_summary"], payload["validation"]["ffc_daily_validation_summary"])
    _write_csv(outputs["validation_ffc_segment_reconciliation"], payload["validation"]["ffc_segment_reconciliation"])

    manifest = {
        "paper_tail_version": PAPER_TAIL_VERSION,
        "paper_tail_algorithm_version": PAPER_TAIL_ALGORITHM_VERSION,
        "scope_signature": signature,
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "external_data_root": str(external_root),
        "paper_tail_root": str(paper_tail_root),
        "paper_tail_weighting": weighting,
        "scope": scope,
        "source_files": source_records,
        "outputs": {key: str(path.relative_to(paper_tail_root)).replace("\\", "/") for key, path in outputs.items() if key != "manifest"},
    }
    _write_json(paths["manifest"], manifest)
    payload["manifest"] = manifest
    payload["paths"] = {key: str(path) for key, path in outputs.items()}
    return payload


def _load_rf_series(path: Path) -> pd.Series:
    df = _safe_read_csv(path)
    if {"date", "rf_log_daily"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["rf_log_daily"] = pd.to_numeric(df["rf_log_daily"], errors="coerce")
    elif {"Clsdt", "Nrrdaydt"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["Clsdt"], errors="coerce")
        daily_pct = pd.to_numeric(df["Nrrdaydt"], errors="coerce")
        df["rf_log_daily"] = np.log1p(daily_pct / 100.0)
    elif {"Clsdt", "Nrrdata"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["Clsdt"], errors="coerce")
        annual_pct = pd.to_numeric(df["Nrrdata"], errors="coerce")
        df["rf_log_daily"] = np.log1p((annual_pct / 100.0) / 365.0)
    else:
        raise ValueError(
            f"Unsupported RF file format: {path}. Expected date/rf_log_daily or CSMAR Clsdt/Nrrdaydt columns."
        )
    df = df.dropna(subset=["date", "rf_log_daily"]).sort_values("date")
    return df.set_index("date")["rf_log_daily"]


def _load_mom_daily(path: Path) -> pd.Series:
    df = _safe_read_csv(path)
    df["kline_time"] = pd.to_datetime(df["kline_time"], errors="coerce")
    df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce")
    df = df.dropna(subset=["kline_time", "MOM"]).copy()
    df["date"] = df["kline_time"].dt.normalize()
    daily = df.groupby("date", sort=True)["MOM"].sum(min_count=1)
    return daily.astype(float)


def _load_ffc_daily(ff3_path: Path, rf_daily: pd.Series, mom_daily: pd.Series) -> pd.DataFrame:
    ff3 = _safe_read_csv(ff3_path)
    if "MarkettypeID" in ff3.columns:
        market_ids = sorted(ff3["MarkettypeID"].dropna().astype(str).unique().tolist())
        chosen = "P9714" if "P9714" in market_ids else market_ids[0]
        ff3 = ff3.loc[ff3["MarkettypeID"].astype(str) == chosen].copy()
    ff3["date"] = pd.to_datetime(ff3["TradingDate"], errors="coerce")
    ff3["RiskPremium1"] = pd.to_numeric(ff3["RiskPremium1"], errors="coerce")
    ff3["SMB1"] = pd.to_numeric(ff3["SMB1"], errors="coerce")
    ff3["HML1"] = pd.to_numeric(ff3["HML1"], errors="coerce")
    ff3 = ff3.dropna(subset=["date", "RiskPremium1", "SMB1", "HML1"]).copy()
    ff3 = ff3[["date", "RiskPremium1", "SMB1", "HML1"]].drop_duplicates("date").sort_values("date")

    df = ff3.set_index("date").join(rf_daily.rename("RF"), how="left")
    df = df.join(mom_daily.rename("MOM"), how="left")
    df["MKT_excess"] = df["RiskPremium1"]
    df["SMB"] = df["SMB1"]
    df["HML"] = df["HML1"]
    out = df.reset_index()[["date", "MKT_excess", "SMB", "HML", "MOM", "RF"]]
    return out.sort_values("date").reset_index(drop=True)


def _load_industry_mapping(path: Path) -> pd.DataFrame:
    df = _safe_read_csv(path)
    df["ts_code"] = _normalize_code(df["ts_code"])
    df["std_industry"] = df["std_industry"].astype(str).str.strip()
    df = df.dropna(subset=["ts_code", "std_industry"])
    return df[["ts_code", "std_industry"]].drop_duplicates("ts_code").sort_values("ts_code").reset_index(drop=True)


def _build_market_cap_matrix(
    mcap_files: Sequence[Path],
    *,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> np.ndarray:
    date_index = {int(ts.strftime("%Y%m%d")): idx for idx, ts in enumerate(dates)}
    symbol_index = {str(symbol): idx for idx, symbol in enumerate(symbols)}
    matrix = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)

    for path in mcap_files:
        for chunk in pd.read_csv(path, usecols=["Stkcd", "Trddt", "Dsmvosd"], chunksize=500_000):
            chunk = chunk.rename(columns={"Stkcd": "code", "Trddt": "date", "Dsmvosd": "mcap"})
            chunk["code"] = _normalize_code(chunk["code"])
            chunk["mcap"] = pd.to_numeric(chunk["mcap"], errors="coerce")
            chunk["date_code"] = (
                chunk["date"].astype(str).str.slice(0, 10).str.replace("-", "", regex=False).str.replace("/", "", regex=False)
            )
            chunk = chunk[chunk["date_code"].str.fullmatch(r"\d{8}", na=False)].copy()
            chunk["date_code"] = chunk["date_code"].astype(np.int32)
            row_idx = chunk["date_code"].map(date_index)
            col_idx = chunk["code"].map(symbol_index)
            valid = row_idx.notna() & col_idx.notna() & np.isfinite(chunk["mcap"].to_numpy())
            if not bool(valid.any()):
                continue
            rows = row_idx.loc[valid].to_numpy(dtype=np.int32, copy=False)
            cols = col_idx.loc[valid].to_numpy(dtype=np.int32, copy=False)
            vals = chunk.loc[valid, "mcap"].to_numpy(dtype=np.float64, copy=False)
            matrix[rows, cols] = vals
    return matrix


def _build_full_market_assets(
    proc_root: Path,
    *,
    global_dates: pd.DatetimeIndex,
    sample_dates: pd.DatetimeIndex,
    rf_daily: pd.Series,
    industry_map: pd.DataFrame,
    mcap_matrix: np.ndarray,
    weighting: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol_files = sorted((proc_root / "symbol_returns").glob("*.npz"))
    symbols = [path.stem for path in symbol_files]
    symbol_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    date_index = {int(ts.strftime("%Y%m%d")): idx for idx, ts in enumerate(global_dates)}
    sample_mask = global_dates.isin(sample_dates)
    rf_daily_arr = rf_daily.reindex(global_dates, fill_value=0.0).to_numpy(dtype=float)
    industry_lookup = dict(zip(industry_map["ts_code"], industry_map["std_industry"]))
    industry_names = sorted(industry_map["std_industry"].dropna().unique().tolist())
    industry_index = {name: idx for idx, name in enumerate(industry_names)}

    T = len(global_dates)
    G = len(industry_names)
    sum_eq = {segment: np.zeros((T, G), dtype=np.float64) for segment in SEGMENT_ORDER}
    cnt_eq = {segment: np.zeros((T, G), dtype=np.float64) for segment in SEGMENT_ORDER}
    sum_vw = {segment: np.zeros((T, G), dtype=np.float64) for segment in SEGMENT_ORDER}
    wgt_vw = {segment: np.zeros((T, G), dtype=np.float64) for segment in SEGMENT_ORDER}
    market_sum_eq = {segment: np.zeros(T, dtype=np.float64) for segment in SEGMENT_ORDER}
    market_cnt_eq = {segment: np.zeros(T, dtype=np.float64) for segment in SEGMENT_ORDER}
    market_sum_vw = {segment: np.zeros(T, dtype=np.float64) for segment in SEGMENT_ORDER}
    market_wgt_vw = {segment: np.zeros(T, dtype=np.float64) for segment in SEGMENT_ORDER}
    all_stock_rows: List[Dict[str, Any]] = []

    for path in symbol_files:
        symbol = path.stem
        arrays = np.load(path)
        raw_date_codes = arrays["date_codes"].astype(np.int32, copy=False)
        row_idx = np.array([date_index.get(int(code), -1) for code in raw_date_codes], dtype=np.int32)
        valid_rows = row_idx >= 0
        if not valid_rows.any():
            continue
        row_idx = row_idx[valid_rows]
        intra = arrays["intraday_returns"][valid_rows].astype(np.float64, copy=False)
        night = arrays["overnight_returns"][valid_rows].astype(np.float64, copy=False)
        daily = arrays["daily_returns"][valid_rows].astype(np.float64, copy=False)

        sample_valid = sample_mask[row_idx]
        if sample_valid.any():
            rows_sample = row_idx[sample_valid]
            rf_sample = rf_daily_arr[rows_sample]
            all_stock_rows.append(
                {
                    "group": "all_stocks",
                    "asset": symbol,
                    "mean_intraday_excess": float(np.nanmean(intra[sample_valid])),
                    "mean_overnight_excess": float(np.nanmean(night[sample_valid] - rf_sample)),
                    "mean_daily_excess": float(np.nanmean(daily[sample_valid] - rf_sample)),
                    "n_obs": int(sample_valid.sum()),
                }
            )

        industry_name = industry_lookup.get(symbol)
        industry_col = industry_index.get(industry_name)
        mcap_col = symbol_index.get(symbol)
        weights = mcap_matrix[row_idx, mcap_col] if mcap_col is not None else None

        segment_values = {
            "intraday": intra,
            "overnight": night,
            "daily": daily,
        }
        for segment, values in segment_values.items():
            finite = np.isfinite(values)
            if finite.any():
                np.add.at(market_sum_eq[segment], row_idx[finite], values[finite])
                np.add.at(market_cnt_eq[segment], row_idx[finite], 1.0)
            if weights is not None:
                finite_w = finite & np.isfinite(weights) & (weights > 0)
                if finite_w.any():
                    np.add.at(market_sum_vw[segment], row_idx[finite_w], values[finite_w] * weights[finite_w])
                    np.add.at(market_wgt_vw[segment], row_idx[finite_w], weights[finite_w])

            if industry_col is None:
                continue
            if finite.any():
                np.add.at(sum_eq[segment][:, industry_col], row_idx[finite], values[finite])
                np.add.at(cnt_eq[segment][:, industry_col], row_idx[finite], 1.0)
            if weights is not None:
                finite_w = finite & np.isfinite(weights) & (weights > 0)
                if finite_w.any():
                    np.add.at(sum_vw[segment][:, industry_col], row_idx[finite_w], values[finite_w] * weights[finite_w])
                    np.add.at(wgt_vw[segment][:, industry_col], row_idx[finite_w], weights[finite_w])

    market_returns_rows: List[Dict[str, Any]] = []
    industry_rows: List[Dict[str, Any]] = []
    for segment in SEGMENT_ORDER:
        eq = np.divide(
            market_sum_eq[segment],
            market_cnt_eq[segment],
            out=np.full(T, np.nan, dtype=float),
            where=market_cnt_eq[segment] > 0,
        )
        vw = np.divide(
            market_sum_vw[segment],
            market_wgt_vw[segment],
            out=np.full(T, np.nan, dtype=float),
            where=market_wgt_vw[segment] > 0,
        )
        for current_weighting, values in [("equal_weighted", eq), ("value_weighted", vw)]:
            market_returns_rows.extend(
                {
                    "date": global_dates[idx],
                    "segment_kind": segment,
                    "weighting": current_weighting,
                    "ret": float(values[idx]) if np.isfinite(values[idx]) else np.nan,
                }
                for idx in range(T)
            )

        eq_matrix = np.divide(
            sum_eq[segment],
            cnt_eq[segment],
            out=np.full((T, G), np.nan, dtype=float),
            where=cnt_eq[segment] > 0,
        )
        vw_matrix = np.divide(
            sum_vw[segment],
            wgt_vw[segment],
            out=np.full((T, G), np.nan, dtype=float),
            where=wgt_vw[segment] > 0,
        )
        for current_weighting, matrix in [("equal_weighted", eq_matrix), ("value_weighted", vw_matrix)]:
            for industry_name, col_idx in industry_index.items():
                values = matrix[:, col_idx]
                industry_rows.extend(
                    {
                        "date": global_dates[idx],
                        "portfolio": industry_name,
                        "segment_kind": segment,
                        "weighting": current_weighting,
                        "ret": float(values[idx]) if np.isfinite(values[idx]) else np.nan,
                    }
                    for idx in range(T)
                )

    all_stocks_df = pd.DataFrame(all_stock_rows).sort_values("asset").reset_index(drop=True)
    industry_assets_df = pd.DataFrame(industry_rows).sort_values(["date", "portfolio", "segment_kind", "weighting"]).reset_index(drop=True)
    market_returns_df = pd.DataFrame(market_returns_rows).sort_values(["date", "segment_kind", "weighting"]).reset_index(drop=True)
    return all_stocks_df, industry_assets_df, market_returns_df


def _load_assignments(path: Path) -> pd.DataFrame:
    df = _safe_read_csv(path, parse_dates=["trade_date", "acc_date", "hold_start", "hold_end"], low_memory=False)
    df["code"] = _normalize_code(df["code"])
    df["portfolio"] = df["portfolio"].astype(str).str.strip().str.upper()
    df["sort_year"] = pd.to_numeric(df["sort_year"], errors="coerce").astype("Int64")
    df["float_mv_adj"] = pd.to_numeric(df["float_mv_adj"], errors="coerce")
    df = df[df["portfolio"].isin(PORTFOLIO_ORDER)].copy()
    df = df.dropna(subset=["sort_year", "code", "portfolio", "float_mv_adj"])
    df["sort_year"] = df["sort_year"].astype(int)
    return df.sort_values(["sort_year", "portfolio", "code"]).reset_index(drop=True)


def _load_year_panel(
    proc_root: Path,
    year: int,
    sample_mode: str = "strict_balanced",
) -> Tuple[pd.DatetimeIndex, List[str], Dict[str, np.ndarray]]:
    meta = _load_json(proc_root / "panels" / sample_mode / f"year_{year}.json")
    dates = pd.to_datetime(meta["dates"])
    tickers = [str(value).strip().upper() for value in meta["tickers"]]
    year_dir = proc_root / "panels" / sample_mode / f"year_{year}"
    arrays = {
        "intraday": np.load(year_dir / "R_intra.npy", mmap_mode="r"),
        "overnight": np.load(year_dir / "R_night.npy", mmap_mode="r"),
        "daily": np.load(year_dir / "R_daily.npy", mmap_mode="r"),
    }
    return pd.DatetimeIndex(dates), tickers, arrays


def _build_size_value_assets(
    proc_root: Path,
    assignments: pd.DataFrame,
    sample_mode: str = "strict_balanced",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    assignments_by_year = {year: frame.reset_index(drop=True) for year, frame in assignments.groupby("sort_year", sort=True)}
    strict_root = proc_root / "panels" / sample_mode
    year_meta_files = sorted(strict_root.glob("year_*.json"))
    years = [int(path.stem.split("_")[1]) for path in year_meta_files if path.stem != "full"]

    long_rows: List[Dict[str, Any]] = []
    daily_wide_vw_parts: List[pd.DataFrame] = []
    daily_wide_ew_parts: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, Any]] = []

    for panel_year in years:
        dates, tickers, arrays = _load_year_panel(proc_root, panel_year, sample_mode=sample_mode)
        ticker_lookup = {ticker: idx for idx, ticker in enumerate(tickers)}
        segments = [
            ("H1", panel_year - 1, np.flatnonzero(dates.month <= 6)),
            ("H2", panel_year, np.flatnonzero(dates.month >= 7)),
        ]
        for half, active_sort_year, row_idx in segments:
            if active_sort_year not in assignments_by_year or row_idx.size == 0:
                continue
            active_assignments = assignments_by_year[active_sort_year]
            meta_row = {
                "date": dates[row_idx],
                "panel_year": panel_year,
                "active_sort_year": active_sort_year,
                "segment": half,
            }
            daily_vw_frame = pd.DataFrame(meta_row)
            daily_ew_frame = pd.DataFrame(meta_row)
            for portfolio in PORTFOLIO_ORDER:
                subset = active_assignments.loc[active_assignments["portfolio"].eq(portfolio)].copy()
                matched = subset.loc[subset["code"].isin(ticker_lookup)].copy()
                col_idx = [ticker_lookup[code] for code in matched["code"].tolist()]
                n_assigned = int(len(subset))
                n_matched = int(len(matched))
                assigned_weight_sum = float(subset["float_mv_adj"].sum())
                matched_weight_sum = float(matched["float_mv_adj"].sum())
                matched_weight_share = matched_weight_sum / assigned_weight_sum if assigned_weight_sum > 0 else np.nan
                coverage_rows.append(
                    {
                        "panel_year": panel_year,
                        "active_sort_year": active_sort_year,
                        "segment": half,
                        "portfolio": portfolio,
                        "n_dates": int(len(row_idx)),
                        "n_assigned": n_assigned,
                        "n_matched": n_matched,
                        "matched_weight_share": matched_weight_share,
                    }
                )

                if not col_idx:
                    daily_vw_frame[portfolio] = np.nan
                    daily_ew_frame[portfolio] = np.nan
                    continue

                weights = matched["float_mv_adj"].to_numpy(dtype=np.float64)
                weights = weights / weights.sum()
                for segment_kind, panel_array in arrays.items():
                    values = np.asarray(panel_array[np.ix_(row_idx, col_idx)], dtype=np.float64)
                    ret_ew = np.nanmean(values, axis=1)
                    ret_vw = _weighted_average(values, weights)
                    long_rows.extend(
                        {
                            "date": dates[row_idx[pos]],
                            "panel_year": panel_year,
                            "active_sort_year": active_sort_year,
                            "segment": half,
                            "portfolio": portfolio,
                            "segment_kind": segment_kind,
                            "weighting": "equal_weighted",
                            "ret": float(ret_ew[pos]) if np.isfinite(ret_ew[pos]) else np.nan,
                        }
                        for pos in range(len(row_idx))
                    )
                    long_rows.extend(
                        {
                            "date": dates[row_idx[pos]],
                            "panel_year": panel_year,
                            "active_sort_year": active_sort_year,
                            "segment": half,
                            "portfolio": portfolio,
                            "segment_kind": segment_kind,
                            "weighting": "value_weighted",
                            "ret": float(ret_vw[pos]) if np.isfinite(ret_vw[pos]) else np.nan,
                        }
                        for pos in range(len(row_idx))
                    )
                    if segment_kind == "daily":
                        daily_vw_frame[portfolio] = ret_vw
                        daily_ew_frame[portfolio] = ret_ew

            daily_wide_vw_parts.append(daily_vw_frame)
            daily_wide_ew_parts.append(daily_ew_frame)

    long_df = pd.DataFrame(long_rows).sort_values(["date", "portfolio", "segment_kind", "weighting"]).reset_index(drop=True)
    daily_wide_vw = pd.concat(daily_wide_vw_parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    daily_wide_ew = pd.concat(daily_wide_ew_parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage_rows).sort_values(["panel_year", "segment", "portfolio"]).reset_index(drop=True)
    summary = {
        "n_rows_long": int(len(long_df)),
        "n_rows_daily": int(len(daily_wide_vw)),
        "panel_years": years,
        "coverage_mean_match_ratio": float(coverage_df["n_matched"].sum() / max(coverage_df["n_assigned"].sum(), 1)) if not coverage_df.empty else np.nan,
    }
    return long_df, daily_wide_vw, daily_wide_ew, {"coverage": coverage_df, "summary": summary}


def _validate_size_value(
    daily_wide_vw: pd.DataFrame,
    daily_wide_ew: pd.DataFrame,
    ref_vw_path: Path,
    ref_ew_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ref_vw = _safe_read_csv(ref_vw_path, parse_dates=["date"])
    ref_ew = _safe_read_csv(ref_ew_path, parse_dates=["date"])
    key_cols = ["date", "panel_year", "active_sort_year", "segment"]
    value_long = daily_wide_vw.melt(id_vars=key_cols, value_vars=PORTFOLIO_ORDER, var_name="portfolio", value_name="ret_rebuilt")
    ref_vw_long = ref_vw.melt(id_vars=key_cols, value_vars=PORTFOLIO_ORDER, var_name="portfolio", value_name="ret_reference")
    value_long["weighting"] = "value_weighted"
    ref_vw_long["weighting"] = "value_weighted"

    equal_long = daily_wide_ew.melt(id_vars=key_cols, value_vars=PORTFOLIO_ORDER, var_name="portfolio", value_name="ret_rebuilt")
    ref_ew_long = ref_ew.melt(id_vars=key_cols, value_vars=PORTFOLIO_ORDER, var_name="portfolio", value_name="ret_reference")
    equal_long["weighting"] = "equal_weighted"
    ref_ew_long["weighting"] = "equal_weighted"

    merged = pd.concat([value_long.merge(ref_vw_long, on=key_cols + ["portfolio", "weighting"], how="outer"), equal_long.merge(ref_ew_long, on=key_cols + ["portfolio", "weighting"], how="outer")], ignore_index=True)
    merged["abs_diff"] = (merged["ret_rebuilt"] - merged["ret_reference"]).abs()
    summary = {
        "max_abs_diff": float(np.nanmax(merged["abs_diff"].to_numpy(dtype=float))) if not merged.empty else np.nan,
        "mean_abs_diff": float(np.nanmean(merged["abs_diff"].to_numpy(dtype=float))) if not merged.empty else np.nan,
        "rows_compared": int(len(merged)),
    }
    return merged.sort_values(["date", "weighting", "portfolio"]).reset_index(drop=True), summary


def _summarize_balanced_panel(result: Any, rf_daily_sample: np.ndarray) -> pd.DataFrame:
    panel = result.panel
    rows: List[Dict[str, Any]] = []
    for col_idx, symbol in enumerate(panel.tickers):
        rows.append(
            {
                "group": "balanced_panel_individual_stocks",
                "asset": symbol,
                "mean_intraday_excess": float(np.nanmean(panel.R_intra[:, col_idx])),
                "mean_overnight_excess": float(np.nanmean(panel.R_night[:, col_idx] - rf_daily_sample)),
                "mean_daily_excess": float(np.nanmean(panel.R_daily[:, col_idx] - rf_daily_sample)),
                "n_obs": int(panel.D),
            }
        )
    return pd.DataFrame(rows)


def _series_to_long_frame(date_index: pd.DatetimeIndex, series_map: Mapping[str, np.ndarray], *, factor_names: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment, matrix in series_map.items():
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        for factor_idx, factor_name in enumerate(factor_names):
            column = matrix[:, factor_idx]
            rows.extend(
                {
                    "date": date_index[pos],
                    "factor": factor_name,
                    "segment_kind": segment,
                    "ret": float(column[pos]) if np.isfinite(column[pos]) else np.nan,
                }
                for pos in range(len(date_index))
            )
    return pd.DataFrame(rows)


def _pivot_factor_matrix(
    factor_df: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    factor_names: Sequence[str],
    segment_kind: str,
) -> np.ndarray:
    subset = factor_df.loc[factor_df["segment_kind"].eq(segment_kind)].copy()
    if subset.empty:
        return np.full((len(dates), len(factor_names)), np.nan, dtype=float)
    pivot = subset.pivot(index="date", columns="factor", values="ret").reindex(dates)
    return pivot.reindex(columns=factor_names).to_numpy(dtype=float)


def _select_industries_from_pca(result: Any, industry_map: pd.DataFrame) -> Dict[str, Any]:
    import prepareCore.engine as eng

    mapping = dict(zip(industry_map["ts_code"], industry_map["std_industry"]))
    available = set(str(x) for x in industry_map["std_industry"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # P5/D1（论文 III.D 的 ex-ante 口径）：行业因子应“看一次修好后的 CN
    # Figure 4 → 冻结 3 个桶 → 写死”，而不是每次从被解释对象里按集中度反推。
    # 若 PELGER_INDUSTRY_FROZEN 给出了冻结桶（决策 D1 落地），直接采用，
    # 不再做样本内反推；否则退回旧的“按集中度自动挑选”占位逻辑，并在
    # selection_rule 里标注其为“待冻结的占位”。
    frozen = _industry_frozen()
    if frozen:
        chosen = [b for b in frozen if b in available]
        selected = [
            {"factor": int(i + 2), "industry": str(b), "concentration": np.nan, "fallback_rank": 0, "frozen": True}
            for i, b in enumerate(chosen)
        ]
        return {
            "selected_industries": selected,
            "market_factor_definition": "Equal-weighted full-market return across all stocks in the sample dates.",
            "selection_rule": "EX-ANTE FROZEN buckets (paper III.D): fixed before estimation via PELGER_INDUSTRY_FROZEN.",
            "frozen_requested": frozen,
            "frozen_missing": [b for b in frozen if b not in available],
        }

    display_pca = result.pipeline.pca_cont_display if getattr(result.pipeline, "pca_cont_display", None) is not None else result.pipeline.pca_cont
    weights = eng.factor_portfolio_weights(display_pca)
    factor_count = min(weights.shape[1], 4)
    selected: List[Dict[str, Any]] = []
    used: set[str] = set()
    for factor_idx in range(1, factor_count):
        abs_weights = np.abs(weights[:, factor_idx])
        bucket_rows: List[Tuple[str, float]] = []
        for symbol, value in zip(result.panel.tickers, abs_weights):
            industry = mapping.get(symbol)
            if industry:
                bucket_rows.append((industry, float(value)))
        if not bucket_rows:
            continue
        bucket_df = pd.DataFrame(bucket_rows, columns=["industry", "abs_weight"])
        grouped = bucket_df.groupby("industry", as_index=False)["abs_weight"].sum().sort_values("abs_weight", ascending=False).reset_index(drop=True)
        total_abs = float(grouped["abs_weight"].sum()) or np.nan
        chosen_rank = None
        chosen_row: Optional[pd.Series] = None
        for rank, row in grouped.iterrows():
            if row["industry"] not in used:
                chosen_rank = rank + 1
                chosen_row = row
                break
        if chosen_row is None:
            chosen_rank = 1
            chosen_row = grouped.iloc[0]
        used.add(str(chosen_row["industry"]))
        selected.append(
            {
                "factor": int(factor_idx + 1),
                "industry": str(chosen_row["industry"]),
                "concentration": float(chosen_row["abs_weight"] / total_abs) if total_abs and np.isfinite(total_abs) else np.nan,
                "fallback_rank": int(chosen_rank),
            }
        )

    return {
        "selected_industries": selected,
        "market_factor_definition": "Equal-weighted full-market return across all stocks in the sample dates.",
        "selection_rule": "PLACEHOLDER (pending D1 freeze): highest-concentration unique std_industry by aggregated |PCA weight|. After one corrected run, freeze the 3 buckets via PELGER_INDUSTRY_FROZEN.",
    }


def _build_industry_factor_frame(
    market_returns_df: pd.DataFrame,
    industry_assets_df: pd.DataFrame,
    industry_selection: Mapping[str, Any],
    *,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    selected = [item["industry"] for item in industry_selection.get("selected_industries", [])]
    market_subset = market_returns_df.loc[market_returns_df["weighting"].eq("equal_weighted"), ["date", "segment_kind", "ret"]].copy()
    market_subset["factor"] = "Market"
    industry_subset = industry_assets_df.loc[
        industry_assets_df["weighting"].eq("equal_weighted") & industry_assets_df["portfolio"].isin(selected),
        ["date", "portfolio", "segment_kind", "ret"],
    ].copy()
    industry_subset = industry_subset.rename(columns={"portfolio": "factor"})
    frame = pd.concat([market_subset[["date", "factor", "segment_kind", "ret"]], industry_subset[["date", "factor", "segment_kind", "ret"]]], ignore_index=True)
    frame["factor"] = pd.Categorical(frame["factor"], categories=["Market"] + selected, ordered=True)
    frame = frame.sort_values(["date", "segment_kind", "factor"]).reset_index(drop=True)
    return frame


def _build_ffc_segmented_frames(
    market_returns_df: pd.DataFrame,
    size_value_assets_df: pd.DataFrame,
    ffc_external_daily: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_vw = (
        market_returns_df.loc[market_returns_df["weighting"].eq("value_weighted"), ["date", "segment_kind", "ret"]]
        .pivot(index="date", columns="segment_kind", values="ret")
        .reindex(dates)
        .reindex(columns=SEGMENT_ORDER)
    )
    vw_assets = size_value_assets_df.loc[size_value_assets_df["weighting"].eq("value_weighted")].copy()
    official_daily = ffc_external_daily.set_index("date").reindex(dates)
    rf_daily = official_daily["RF"].fillna(0.0)
    mom_daily = official_daily["MOM"].fillna(0.0)

    raw_mats: Dict[str, np.ndarray] = {}
    for segment_kind in ("intraday", "overnight"):
        pivot = (
            vw_assets.loc[vw_assets["segment_kind"].eq(segment_kind)]
            .pivot(index="date", columns="portfolio", values="ret")
            .reindex(dates)
        )
        smb = pivot[["SL", "SM", "SH"]].mean(axis=1) - pivot[["BL", "BM", "BH"]].mean(axis=1)
        hml = 0.5 * (pivot["SH"] + pivot["BH"]) - 0.5 * (pivot["SL"] + pivot["BL"])
        if segment_kind == "intraday":
            market_factor = market_vw["intraday"]
            mom = mom_daily
        else:
            market_factor = market_vw["overnight"] - rf_daily
            mom = pd.Series(0.0, index=dates)
        raw_mats[segment_kind] = np.column_stack(
            [
                market_factor.to_numpy(dtype=float),
                smb.to_numpy(dtype=float),
                hml.to_numpy(dtype=float),
                mom.to_numpy(dtype=float),
            ]
        )

    raw_daily = raw_mats["intraday"] + raw_mats["overnight"]
    official_daily_matrix = official_daily.reindex(columns=OFFICIAL_FFC_FACTORS).to_numpy(dtype=float)

    final_intraday = raw_mats["intraday"].copy()
    final_overnight = raw_mats["overnight"].copy()
    final_daily = np.full_like(official_daily_matrix, np.nan, dtype=float)
    recon_rows: List[Dict[str, Any]] = []

    for factor_idx, factor_name in enumerate(OFFICIAL_FFC_FACTORS):
        raw_intra = raw_mats["intraday"][:, factor_idx]
        raw_night = raw_mats["overnight"][:, factor_idx]
        raw_day = raw_daily[:, factor_idx]
        official_day = official_daily_matrix[:, factor_idx]
        valid = np.isfinite(raw_intra) & np.isfinite(raw_night) & np.isfinite(official_day)
        delta = np.full(len(dates), np.nan, dtype=float)
        delta[valid] = official_day[valid] - raw_day[valid]
        denom = np.abs(raw_intra) + np.abs(raw_night)
        intra_share = np.full(len(dates), 0.5, dtype=float)
        weighted = valid & (denom > 1e-12)
        intra_share[weighted] = np.abs(raw_intra[weighted]) / denom[weighted]
        final_intraday[valid, factor_idx] = raw_intra[valid] + delta[valid] * intra_share[valid]
        final_overnight[valid, factor_idx] = raw_night[valid] + delta[valid] * (1.0 - intra_share[valid])
        final_daily[valid, factor_idx] = final_intraday[valid, factor_idx] + final_overnight[valid, factor_idx]

        recon_rows.extend(
            {
                "date": dates[pos],
                "factor": factor_name,
                "raw_intraday": float(raw_intra[pos]) if np.isfinite(raw_intra[pos]) else np.nan,
                "raw_overnight": float(raw_night[pos]) if np.isfinite(raw_night[pos]) else np.nan,
                "raw_daily": float(raw_day[pos]) if np.isfinite(raw_day[pos]) else np.nan,
                "official_daily": float(official_day[pos]) if np.isfinite(official_day[pos]) else np.nan,
                "final_intraday": float(final_intraday[pos, factor_idx]) if np.isfinite(final_intraday[pos, factor_idx]) else np.nan,
                "final_overnight": float(final_overnight[pos, factor_idx]) if np.isfinite(final_overnight[pos, factor_idx]) else np.nan,
                "final_daily": float(final_daily[pos, factor_idx]) if np.isfinite(final_daily[pos, factor_idx]) else np.nan,
            }
            for pos in range(len(dates))
        )

    raw_frame = _series_to_long_frame(
        dates,
        {
            "intraday": raw_mats["intraday"],
            "overnight": raw_mats["overnight"],
            "daily": raw_daily,
        },
        factor_names=OFFICIAL_FFC_FACTORS,
    )
    final_frame = _series_to_long_frame(
        dates,
        {
            "intraday": final_intraday,
            "overnight": final_overnight,
            "daily": final_daily,
        },
        factor_names=OFFICIAL_FFC_FACTORS,
    )
    return (
        raw_frame.sort_values(["date", "segment_kind", "factor"]).reset_index(drop=True),
        final_frame.sort_values(["date", "segment_kind", "factor"]).reset_index(drop=True),
        pd.DataFrame(recon_rows).sort_values(["date", "factor"]).reset_index(drop=True),
    )


def _validate_ffc_daily(
    ffc_segmented: pd.DataFrame,
    ffc_external: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    daily_matrix = (
        ffc_segmented.loc[ffc_segmented["segment_kind"].eq("daily")]
        .pivot(index="date", columns="factor", values="ret")
        .reindex(dates)
        .reindex(columns=OFFICIAL_FFC_FACTORS)
    )
    external = ffc_external.set_index("date").reindex(dates).reindex(columns=OFFICIAL_FFC_FACTORS)
    rows: List[Dict[str, Any]] = []
    for factor_name in OFFICIAL_FFC_FACTORS:
        internal = daily_matrix[factor_name]
        official = external[factor_name]
        valid = internal.notna() & official.notna()
        diff = (internal - official).abs()
        rows.extend(
            {
                "date": dates[pos],
                "factor": factor_name,
                "internal": float(internal.iloc[pos]) if pd.notna(internal.iloc[pos]) else np.nan,
                "external": float(official.iloc[pos]) if pd.notna(official.iloc[pos]) else np.nan,
                "abs_diff": float(diff.iloc[pos]) if valid.iloc[pos] else np.nan,
            }
            for pos in range(len(dates))
            if valid.iloc[pos]
        )
    diff_df = pd.DataFrame(rows).sort_values(["date", "factor"]).reset_index(drop=True)
    summary = {
        "rows_compared": int(len(diff_df)),
        "max_abs_diff": float(np.nanmax(diff_df["abs_diff"].to_numpy(dtype=float))) if not diff_df.empty else np.nan,
        "mean_abs_diff": float(np.nanmean(diff_df["abs_diff"].to_numpy(dtype=float))) if not diff_df.empty else np.nan,
    }
    return diff_df, summary


def _align_matrix(frame: pd.DataFrame, *, dates: pd.DatetimeIndex, factor_names: Sequence[str]) -> np.ndarray:
    pivot = frame.pivot(index="date", columns="factor", values="ret").reindex(dates)
    return pivot.reindex(columns=factor_names).to_numpy(dtype=float)


def _drop_invalid_rows(lhs: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lhs_use = lhs if lhs.ndim == 2 else lhs[:, None]
    rhs_use = rhs if rhs.ndim == 2 else rhs[:, None]
    mask = np.isfinite(lhs_use).all(axis=1) & np.isfinite(rhs_use).all(axis=1)
    return lhs_use[mask], rhs_use[mask]


def _build_table_iii(
    result: Any,
    *,
    sample_dates: pd.DatetimeIndex,
    industry_factors: pd.DataFrame,
    ffc_external: pd.DataFrame,
    selected_industries: Sequence[str],
) -> pd.DataFrame:
    import prepareCore.engine as eng

    continuous = np.asarray(result.pipeline.F_cont_display_daily_total, dtype=float)
    factor_order = ["Market", *list(selected_industries)]

    industry_daily = industry_factors.loc[industry_factors["segment_kind"].eq("daily"), ["date", "factor", "ret"]].copy()
    industry_matrix_full = _align_matrix(industry_daily, dates=sample_dates, factor_names=factor_order)
    external_daily = ffc_external.set_index("date").reindex(sample_dates)
    comparisons: List[Tuple[str, np.ndarray]] = [
        ("Industry full", industry_matrix_full),
        ("FFC 4-factor", external_daily[OFFICIAL_FFC_FACTORS].to_numpy(dtype=float)),
        ("FF3", external_daily[OFFICIAL_FFC_FACTORS[:3]].to_numpy(dtype=float)),
        ("Market only", external_daily[["MKT_excess"]].to_numpy(dtype=float)),
    ]
    for drop_idx, drop_name in enumerate(selected_industries):
        keep_cols = [0] + [idx + 1 for idx, _name in enumerate(selected_industries) if idx != drop_idx]
        comparisons.insert(
            drop_idx + 1,
            (f"Industry ablation: drop {drop_name}", industry_matrix_full[:, keep_cols]),
        )

    rows: List[Dict[str, Any]] = []
    for label, matrix in comparisons:
        lhs, rhs = _drop_invalid_rows(continuous, matrix)
        gc = eng.generalized_correlations(lhs, rhs) if lhs.size and rhs.size else np.array([], dtype=float)
        row: Dict[str, Any] = {
            "comparison": label,
            "panel": "top_returns_gc",
            "n_factors": int(matrix.shape[1]) if matrix.ndim == 2 else 1,
            "sample_days": int(lhs.shape[0]) if lhs.ndim == 2 else 0,
            "gc_mean": float(np.nanmean(gc)) if len(gc) else np.nan,
        }
        for idx in range(1, 5):
            row[f"gc_{idx}"] = float(gc[idx - 1]) if idx <= len(gc) else np.nan
        rows.append(row)

    # ------------------------------------------------------------------
    # N9（论文 Table III）：补上半部 HF PCA / PCA Proxy 两行 + 下半部
    # ω HF/Jump/Overnight/Daily/Week/Month 的【载荷 GC】。全部 best-effort、
    # 逐项 try/except；若复用的 pickle 缺少所需数组则跳过该行（不崩溃）。
    try:
        _extend_table_iii(rows, result, sample_dates=sample_dates)
    except Exception as exc:  # pragma: no cover
        print(f"[paper_tail] N9 Table III 扩展部分跳过: {exc!r}")

    df = pd.DataFrame(rows)
    if "panel" not in df.columns:
        df["panel"] = "top_returns_gc"
    return df


def _gc_safe(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    import prepareCore.engine as eng
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.size == 0 or B.size == 0:
        return np.array([], dtype=float)
    return eng.generalized_correlations(A, B)


def _pca_loadings(R: Optional[np.ndarray], K: int) -> Optional[np.ndarray]:
    """对一个 (T×N) 收益矩阵做 PCA，返回 N×K 载荷（用于载荷空间 GC）。"""
    import prepareCore.engine as eng
    if R is None:
        return None
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] < 3:
        return None
    res = eng._panel_pca(R, K=min(K, R.shape[1]), use_corr=True) if hasattr(eng, "_panel_pca") else eng.pca_factors(R, K=min(K, R.shape[1]), use_corr=True)
    return np.asarray(res.Lambda, dtype=float)


def _aggregate_daily_to_period(R_daily: np.ndarray, dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """把日频 (T×N) 聚合到 weekly / monthly（对数收益求和）。"""
    df = pd.DataFrame(np.asarray(R_daily, dtype=float), index=pd.DatetimeIndex(dates))
    key = df.index.to_period("W") if freq == "weekly" else df.index.to_period("M")
    return df.groupby(key).sum(min_count=1).to_numpy(dtype=float)


def _extend_table_iii(rows: List[Dict[str, Any]], result: Any, *, sample_dates: pd.DatetimeIndex) -> None:
    """N9 的扩展行（top: HF PCA / PCA Proxy；bottom: ω 频率载荷 GC）。"""
    import prepareCore.engine as eng
    pipe = result.pipeline
    panel = result.panel
    display_k = int(np.asarray(pipe.F_cont_display_daily_total).shape[1]) if getattr(pipe, "F_cont_display_daily_total", None) is not None else 4
    display_k = max(1, min(4, display_k))

    # ---- 上半部 HF PCA 行：HF(连续+跳跃)PCA 因子收益 vs 连续 display 因子收益 ----
    cont_ret = np.asarray(pipe.F_cont_display_daily_total, dtype=float)
    try:
        W_hf = eng.factor_portfolio_weights(_truncate_or_pca(pipe, "hf", display_k))
        hf_ret = np.asarray(panel.R_daily, dtype=float) @ W_hf
        lhs, rhs = _drop_invalid_rows(cont_ret, hf_ret)
        gc = _gc_safe(lhs, rhs)
        rows.append(_gc_row("HF PCA", "top_returns_gc", rhs.shape[1] if rhs.ndim == 2 else 1, lhs.shape[0] if lhs.ndim == 2 else 0, gc))
    except Exception as exc:
        print(f"[paper_tail] HF PCA 行跳过: {exc!r}")

    # ---- 上半部 PCA Proxy 行：proxy 因子收益 vs 连续 display 因子收益 ----
    try:
        W_disp = eng.factor_portfolio_weights(pipe.pca_cont_display)
        proxy_w, _ = eng.build_proxy_factors(W_disp, pipe.R_cont)
        proxy_ret = np.asarray(panel.R_daily, dtype=float) @ proxy_w
        lhs, rhs = _drop_invalid_rows(cont_ret, proxy_ret)
        gc = _gc_safe(lhs, rhs)
        rows.append(_gc_row("PCA Proxy", "top_returns_gc", rhs.shape[1] if rhs.ndim == 2 else 1, lhs.shape[0] if lhs.ndim == 2 else 0, gc))
    except Exception as exc:
        print(f"[paper_tail] PCA Proxy 行跳过: {exc!r}")

    # ---- 下半部：ω 频率的【载荷 GC】（vs 连续 display 载荷）----
    cont_load = np.asarray(pipe.pca_cont_display.Lambda, dtype=float)
    freq_loadings: List[Tuple[str, Optional[np.ndarray]]] = [
        ("omega HF", _pca_loadings(getattr(panel, "R_5min_full", None), display_k)),
        ("omega Jump", _pca_loadings(getattr(pipe, "R_jump", None), display_k)),
        ("omega Overnight", _pca_loadings(getattr(panel, "R_night", None), display_k)),
        ("omega Daily", _pca_loadings(getattr(panel, "R_daily", None), display_k)),
    ]
    try:
        wk = _aggregate_daily_to_period(panel.R_daily, panel.dates, "weekly")
        freq_loadings.append(("omega Week", _pca_loadings(wk, display_k)))
        mo = _aggregate_daily_to_period(panel.R_daily, panel.dates, "monthly")
        freq_loadings.append(("omega Month", _pca_loadings(mo, display_k)))
    except Exception as exc:
        print(f"[paper_tail] weekly/monthly 载荷跳过: {exc!r}")

    for label, Lam in freq_loadings:
        if Lam is None:
            continue
        try:
            gc = _gc_safe(cont_load, Lam)  # 载荷空间 GC（论文 footnote 19 同式）
            rows.append(_gc_row(label, "bottom_loadings_gc", Lam.shape[1] if Lam.ndim == 2 else 1, int(cont_load.shape[0]), gc))
        except Exception as exc:
            print(f"[paper_tail] {label} 载荷 GC 跳过: {exc!r}")


def _truncate_or_pca(pipe: Any, which: str, k: int):
    import prepareCore.engine as eng
    src = getattr(pipe, f"pca_{which}", None)
    if src is not None and int(src.Lambda.shape[1]) >= k:
        return eng._truncate_pca_result(src, k)
    R = getattr(pipe, "R_cont", None) if which == "cont" else getattr(pipe.panel, "R_5min_full", None) if which == "hf" else getattr(pipe, "R_jump", None)
    return eng.pca_factors(np.asarray(R, dtype=float), K=k, use_corr=getattr(pipe, "use_corr", True))


def _gc_row(label: str, panel_name: str, n_factors: int, sample_n: int, gc: np.ndarray) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "comparison": label,
        "panel": panel_name,
        "n_factors": int(n_factors),
        "sample_days": int(sample_n),
        "gc_mean": float(np.nanmean(gc)) if len(gc) else np.nan,
    }
    for idx in range(1, 5):
        row[f"gc_{idx}"] = float(gc[idx - 1]) if idx <= len(gc) else np.nan
    return row


def _segment_rf_vectors(rf_daily_sample: np.ndarray) -> Dict[str, np.ndarray]:
    zeros = np.zeros(len(rf_daily_sample), dtype=float)
    daily_rf = np.asarray(rf_daily_sample, dtype=float)
    return {
        "intraday": zeros,
        "overnight": daily_rf,
        "daily": daily_rf,
    }


def _to_excess_matrices(
    matrix_by_segment: Mapping[str, np.ndarray],
    *,
    rf_daily_sample: np.ndarray,
) -> Dict[str, np.ndarray]:
    rf_by_segment = _segment_rf_vectors(rf_daily_sample)
    out: Dict[str, np.ndarray] = {}
    for segment in SEGMENT_ORDER:
        matrix = np.asarray(matrix_by_segment[segment], dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        out[segment] = matrix - rf_by_segment[segment][:, None]
    return out


def _portfolio_sharpes_from_excess_matrices(matrix_by_segment: Mapping[str, np.ndarray]) -> Dict[str, float]:
    import prepareCore.engine as eng

    scale = np.sqrt(float(_annualization_days()))  # N6: 论文 252；A 股可设 PELGER_ANNUALIZATION_DAYS=243
    intra = np.asarray(matrix_by_segment["intraday"], dtype=float)
    night = np.asarray(matrix_by_segment["overnight"], dtype=float)
    daily = np.asarray(matrix_by_segment["daily"], dtype=float)

    intra_mask = np.isfinite(intra).all(axis=1)
    night_mask = np.isfinite(night).all(axis=1)
    daily_mask = np.isfinite(daily).all(axis=1)

    _, sr_intra = eng.tangency_portfolio(intra[intra_mask], np.zeros(int(intra_mask.sum()), dtype=float)) if intra_mask.any() else (None, np.nan)
    _, sr_night = eng.tangency_portfolio(night[night_mask], np.zeros(int(night_mask.sum()), dtype=float)) if night_mask.any() else (None, np.nan)
    _, sr_daily = eng.tangency_portfolio(daily[daily_mask], np.zeros(int(daily_mask.sum()), dtype=float)) if daily_mask.any() else (None, np.nan)
    return {
        "SR_intraday": float(sr_intra * scale) if np.isfinite(sr_intra) else np.nan,
        "SR_overnight": float(sr_night * scale) if np.isfinite(sr_night) else np.nan,
        "SR_daily": float(sr_daily * scale) if np.isfinite(sr_daily) else np.nan,
    }


def _matrix_diagnostics(matrix: np.ndarray, factor_names: Sequence[str]) -> Dict[str, Any]:
    use = np.asarray(matrix, dtype=float)
    if use.ndim == 1:
        use = use[:, None]
    mask = np.isfinite(use).all(axis=1)
    aligned = use[mask]
    means = {factor_names[idx]: float(np.nanmean(use[:, idx])) for idx in range(min(use.shape[1], len(factor_names)))}
    stds = {factor_names[idx]: float(np.nanstd(use[:, idx], ddof=1)) for idx in range(min(use.shape[1], len(factor_names)))}
    if aligned.shape[0] < 2 or aligned.shape[1] == 0:
        return {
            "n_obs": int(aligned.shape[0]),
            "n_factors": int(use.shape[1]),
            "rank": 0,
            "condition_number": np.nan,
            "mean": means,
            "std": stds,
        }
    cov = np.atleast_2d(np.cov(aligned, rowvar=False, ddof=1))
    return {
        "n_obs": int(aligned.shape[0]),
        "n_factors": int(use.shape[1]),
        "rank": int(np.linalg.matrix_rank(cov)),
        "condition_number": float(np.linalg.cond(cov)) if cov.size else np.nan,
        "mean": means,
        "std": stds,
    }


def _build_factor_sets(
    result: Any,
    *,
    sample_dates: pd.DatetimeIndex,
    rf_daily_sample: np.ndarray,
    industry_factors: pd.DataFrame,
    ffc_segmented: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
    import prepareCore.engine as eng

    display_pca = result.pipeline.pca_cont_display if getattr(result.pipeline, "pca_cont_display", None) is not None else result.pipeline.pca_cont
    weights = eng.factor_portfolio_weights(display_pca)
    proxy_weights, _ = eng.build_proxy_factors(weights, result.pipeline.R_cont)
    panel = result.panel
    continuous_raw = {
        "intraday": np.asarray(result.pipeline.F_cont_display_daily_intra, dtype=float),
        "overnight": np.asarray(result.pipeline.F_cont_display_daily_night, dtype=float),
        "daily": np.asarray(result.pipeline.F_cont_display_daily_total, dtype=float),
    }
    proxy_raw = {
        "intraday": np.asarray(panel.R_intra @ proxy_weights, dtype=float),
        "overnight": np.asarray(panel.R_night @ proxy_weights, dtype=float),
        "daily": np.asarray(panel.R_daily @ proxy_weights, dtype=float),
    }
    industry_factor_names = sorted(industry_factors["factor"].dropna().unique().tolist(), key=lambda name: (0 if name == "Market" else 1, name))
    industry_raw = {
        segment: _align_matrix(
            industry_factors.loc[industry_factors["segment_kind"].eq(segment), ["date", "factor", "ret"]],
            dates=sample_dates,
            factor_names=industry_factor_names,
        )
        for segment in SEGMENT_ORDER
    }
    ffc = {
        segment: _align_matrix(
            ffc_segmented.loc[ffc_segmented["segment_kind"].eq(segment), ["date", "factor", "ret"]],
            dates=sample_dates,
            factor_names=OFFICIAL_FFC_FACTORS,
        )
        for segment in SEGMENT_ORDER
    }

    factor_sets = {
        "Continuous PCA": _to_excess_matrices(continuous_raw, rf_daily_sample=rf_daily_sample),
        "Proxy PCA": _to_excess_matrices(proxy_raw, rf_daily_sample=rf_daily_sample),
        "Industry full": _to_excess_matrices(industry_raw, rf_daily_sample=rf_daily_sample),
        "FFC 4-factor": ffc,
        "FF3": {segment: ffc[segment][:, :3] for segment in SEGMENT_ORDER},
        "Market only": {segment: ffc[segment][:, :1] for segment in SEGMENT_ORDER},
    }
    factor_names = {
        "Continuous PCA": [f"Continuous PCA Factor {idx}" for idx in range(1, factor_sets["Continuous PCA"]["daily"].shape[1] + 1)],
        "Proxy PCA": [f"Proxy PCA Factor {idx}" for idx in range(1, factor_sets["Proxy PCA"]["daily"].shape[1] + 1)],
        "Industry full": industry_factor_names,
        "FFC 4-factor": OFFICIAL_FFC_FACTORS,
        "FF3": OFFICIAL_FFC_FACTORS[:3],
        "Market only": ["MKT_excess"],
    }
    diagnostics = {
        label: {
            segment: _matrix_diagnostics(factor_sets[label][segment], factor_names[label])
            for segment in SEGMENT_ORDER
        }
        for label in factor_sets
    }
    return factor_sets, diagnostics


def _build_table_v(
    result: Any,
    *,
    sample_dates: pd.DatetimeIndex,
    rf_daily_sample: np.ndarray,
    industry_factors: pd.DataFrame,
    ffc_segmented: pd.DataFrame,
) -> pd.DataFrame:
    factor_sets, _ = _build_factor_sets(
        result,
        sample_dates=sample_dates,
        rf_daily_sample=rf_daily_sample,
        industry_factors=industry_factors,
        ffc_segmented=ffc_segmented,
    )
    factor_names_cont = [f"Continuous PCA Factor {idx}" for idx in range(1, factor_sets["Continuous PCA"]["daily"].shape[1] + 1)]

    upper_rows: List[Dict[str, Any]] = []
    for label, mats in [
        ("Continuous PCA", factor_sets["Continuous PCA"]),
        ("Proxy PCA", factor_sets["Proxy PCA"]),
        ("Industry full", factor_sets["Industry full"]),
        ("FFC 4-factor", factor_sets["FFC 4-factor"]),
        ("FF3", factor_sets["FF3"]),
        ("Market only", factor_sets["Market only"]),
    ]:
        sharpes = _portfolio_sharpes_from_excess_matrices(mats)
        upper_rows.append(
            {
                "section": "factor_set_tangency",
                "portfolio": label,
                "SR_intraday": sharpes["SR_intraday"],
                "SR_overnight": sharpes["SR_overnight"],
                "SR_daily": sharpes["SR_daily"],
            }
        )

    lower_rows: List[Dict[str, Any]] = []
    ann = float(_annualization_days())

    def _seg_sr(series: np.ndarray) -> float:
        # footnote 33：因子按自身 std 归一化 -> 均值即 Sharpe（这里直接 mean/std×√ann）。
        s = np.asarray(series, dtype=float)
        sd = np.nanstd(s, ddof=1)
        return float(np.nanmean(s) / sd * np.sqrt(ann)) if np.isfinite(sd) and sd > 0 else np.nan

    for idx, name in enumerate(factor_names_cont[:4], start=1):
        lower_rows.append(
            {
                "section": "continuous_individual_factors",
                "portfolio": name,
                "SR_intraday": _seg_sr(factor_sets["Continuous PCA"]["intraday"][:, idx - 1]),
                "SR_overnight": _seg_sr(factor_sets["Continuous PCA"]["overnight"][:, idx - 1]),
                "SR_daily": _seg_sr(factor_sets["Continuous PCA"]["daily"][:, idx - 1]),
            }
        )

    # N10（论文 Table V 下半部）：补 Market / Size / Value / Momentum 的个体夏普，
    # 用于展示特征因子“隔夜赚、盘中亏”的反转。取自（修好的）分段 FFC 四列。
    ffc_mats = factor_sets["FFC 4-factor"]
    for col, label in enumerate(["Market", "Size", "Value", "Momentum"]):
        if col >= ffc_mats["daily"].shape[1]:
            break
        lower_rows.append(
            {
                "section": "characteristic_individual_factors",
                "portfolio": label,
                "SR_intraday": _seg_sr(ffc_mats["intraday"][:, col]),
                "SR_overnight": _seg_sr(ffc_mats["overnight"][:, col]),
                "SR_daily": _seg_sr(ffc_mats["daily"][:, col]),
            }
        )

    return pd.DataFrame(upper_rows + lower_rows)


def _build_figure12_data(
    balanced_summary: pd.DataFrame,
    all_stock_summary: pd.DataFrame,
    industry_assets: pd.DataFrame,
    size_value_assets: pd.DataFrame,
    *,
    sample_dates: pd.DatetimeIndex,
    rf_daily_sample: np.ndarray,
    weighting: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = [balanced_summary, all_stock_summary]
    official_industry = industry_assets.loc[industry_assets["weighting"].eq(weighting) & industry_assets["date"].isin(sample_dates)].copy()
    rf_by_date = pd.Series(rf_daily_sample, index=sample_dates)
    official_industry["rf"] = official_industry["date"].map(rf_by_date).fillna(0.0)
    official_industry["ret_excess"] = official_industry["ret"]
    mask_non_intra = official_industry["segment_kind"].isin(["overnight", "daily"])
    official_industry.loc[mask_non_intra, "ret_excess"] = official_industry.loc[mask_non_intra, "ret"] - official_industry.loc[mask_non_intra, "rf"]

    industry_summary = (
        official_industry.pivot_table(index=["portfolio"], columns="segment_kind", values="ret_excess", aggfunc="mean")
        .rename(columns={"intraday": "mean_intraday_excess", "overnight": "mean_overnight_excess", "daily": "mean_daily_excess"})
        .reset_index()
        .rename(columns={"portfolio": "asset"})
    )
    industry_summary.insert(0, "group", "industry_portfolios")
    # P13a：n_obs 用真实有效观测数（按 daily 段非缺失计），不再写 len(sample_dates)。
    _ind_obs = (
        official_industry.loc[official_industry["segment_kind"].eq("daily")]
        .assign(_ok=lambda d: d["ret"].notna())
        .groupby("portfolio")["_ok"].sum()
    )
    industry_summary["n_obs"] = industry_summary["asset"].map(_ind_obs).fillna(0).astype(int)
    rows.append(industry_summary)

    official_sv = size_value_assets.loc[size_value_assets["weighting"].eq(weighting) & size_value_assets["date"].isin(sample_dates)].copy()
    official_sv["rf"] = official_sv["date"].map(rf_by_date).fillna(0.0)
    official_sv["ret_excess"] = official_sv["ret"]
    mask_non_intra_sv = official_sv["segment_kind"].isin(["overnight", "daily"])
    official_sv.loc[mask_non_intra_sv, "ret_excess"] = official_sv.loc[mask_non_intra_sv, "ret"] - official_sv.loc[mask_non_intra_sv, "rf"]
    size_summary = (
        official_sv.pivot_table(index=["portfolio"], columns="segment_kind", values="ret_excess", aggfunc="mean")
        .rename(columns={"intraday": "mean_intraday_excess", "overnight": "mean_overnight_excess", "daily": "mean_daily_excess"})
        .reset_index()
        .rename(columns={"portfolio": "asset"})
    )
    size_summary.insert(0, "group", "size_value_portfolios")
    _sv_obs = (
        official_sv.loc[official_sv["segment_kind"].eq("daily")]
        .assign(_ok=lambda d: d["ret"].notna())
        .groupby("portfolio")["_ok"].sum()
    )
    size_summary["n_obs"] = size_summary["asset"].map(_sv_obs).fillna(0).astype(int)
    rows.append(size_summary)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out["eligible_for_plot"] = True
    out["plot_exclusion_reason"] = ""
    all_stock_mask = out["group"].eq("all_stocks")
    out.loc[all_stock_mask & out["n_obs"].lt(FIGURE12_MIN_ALL_STOCK_OBS), "eligible_for_plot"] = False
    out.loc[all_stock_mask & out["n_obs"].lt(FIGURE12_MIN_ALL_STOCK_OBS), "plot_exclusion_reason"] = f"n_obs_lt_{FIGURE12_MIN_ALL_STOCK_OBS}"

    all_stock_rows = out.loc[all_stock_mask].copy()
    all_stock_rows["abs_daily"] = all_stock_rows["mean_daily_excess"].abs()
    filter_diag = {
        "min_all_stock_obs": FIGURE12_MIN_ALL_STOCK_OBS,
        "all_stocks_total": int(all_stock_rows.shape[0]),
        "all_stocks_excluded": int((~all_stock_rows["eligible_for_plot"]).sum()),
        "all_stocks_included": int(all_stock_rows["eligible_for_plot"].sum()),
        "excluded_n_obs_summary": {
            "min": float(all_stock_rows.loc[~all_stock_rows["eligible_for_plot"], "n_obs"].min()) if (~all_stock_rows["eligible_for_plot"]).any() else np.nan,
            "median": float(all_stock_rows.loc[~all_stock_rows["eligible_for_plot"], "n_obs"].median()) if (~all_stock_rows["eligible_for_plot"]).any() else np.nan,
            "max": float(all_stock_rows.loc[~all_stock_rows["eligible_for_plot"], "n_obs"].max()) if (~all_stock_rows["eligible_for_plot"]).any() else np.nan,
        },
        "largest_abs_daily_mean_outliers": all_stock_rows.sort_values("abs_daily", ascending=False).head(15)[
            ["asset", "n_obs", "mean_intraday_excess", "mean_overnight_excess", "mean_daily_excess", "eligible_for_plot"]
        ].to_dict(orient="records"),
    }
    return (
        out[
            [
                "group",
                "asset",
                "mean_intraday_excess",
                "mean_overnight_excess",
                "mean_daily_excess",
                "n_obs",
                "eligible_for_plot",
                "plot_exclusion_reason",
            ]
        ],
        filter_diag,
    )


def _orthogonal_rotation(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if src.ndim == 1:
        src = src[:, None]
    if tgt.ndim == 1:
        tgt = tgt[:, None]
    k = min(src.shape[1], tgt.shape[1])
    src = src[:, :k]
    tgt = tgt[:, :k]
    valid = np.isfinite(src).all(axis=1) & np.isfinite(tgt).all(axis=1)
    if valid.sum() < max(2, k):
        q = np.eye(k, dtype=float)
        return q, {
            "valid_rows": int(valid.sum()),
            "orthogonality_error": 0.0,
            "alignment_rmse": np.nan,
            "used_identity": True,
        }
    x = src[valid]
    y = tgt[valid]
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    q = u @ vt
    aligned = x @ q
    return q, {
        "valid_rows": int(valid.sum()),
        "orthogonality_error": float(np.linalg.norm(q.T @ q - np.eye(k), ord="fro")),
        "alignment_rmse": float(np.sqrt(np.nanmean((aligned - y) ** 2))),
        "used_identity": False,
    }


def _continuous_factor_segments_from_panel(panel: Any, *, jump_a: float, k: int) -> Dict[str, Any]:
    import prepareCore.engine as eng

    r_cont, _ = eng.detect_jumps(panel, a=float(jump_a))
    pca = eng._panel_pca(r_cont, K=int(k), use_corr=True)
    eng.orient_pca_result(pca)
    weights = eng.factor_portfolio_weights(pca)
    return {
        "pca": pca,
        "weights": weights,
        "intraday": np.asarray(panel.R_intra @ weights, dtype=float),
        "overnight": np.asarray(panel.R_night @ weights, dtype=float),
        "daily": np.asarray(panel.R_daily @ weights, dtype=float),
        "n_symbols": int(panel.N),
        "n_days": int(panel.D),
    }


def _build_yearly_aligned_unbalanced_pca_segments(
    result: Any,
    *,
    proc_root: Path,
    sample_dates: pd.DatetimeIndex,
    k: int = 4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    import prepareCore.engine as eng

    baseline = {
        "intraday": np.asarray(result.pipeline.F_cont_display_daily_intra, dtype=float),
        "overnight": np.asarray(result.pipeline.F_cont_display_daily_night, dtype=float),
        "daily": np.asarray(result.pipeline.F_cont_display_daily_total, dtype=float),
    }
    k_use = min(int(k), int(baseline["daily"].shape[1]))
    out = {
        segment: np.full((len(sample_dates), k_use), np.nan, dtype=float)
        for segment in SEGMENT_ORDER
    }
    diagnostics: Dict[str, Any] = {
        "factor_set_label": "Continuous PCA (unbalanced, yearly aligned)",
        "alignment_steps": [
            "year-specific balanced changing-universe -> same-year fixed intersection",
            "same-year fixed intersection -> global fixed-intersection baseline",
        ],
        "years": [],
    }

    years = sorted({int(date.year) for date in sample_dates})
    for year in years:
        year_mask = np.array([int(date.year) == int(year) for date in sample_dates], dtype=bool)
        fixed_year_panel = eng.subset_panel_by_years(result.panel, [int(year)])
        changing_year_panel = eng.load_proc_hf_panel(
            proc_root=proc_root,
            sample_mode=eng.STRICT_BALANCED_SAMPLE,
            years=[int(year)],
            return_mode=result.panel.requested_return_mode or "open_close",
        )
        fixed_segments = _continuous_factor_segments_from_panel(
            fixed_year_panel,
            jump_a=float(result.pipeline.jump_a),
            k=k_use,
        )
        changing_segments = _continuous_factor_segments_from_panel(
            changing_year_panel,
            jump_a=float(result.pipeline.jump_a),
            k=k_use,
        )
        global_year_daily = baseline["daily"][year_mask, :k_use]
        q1, diag1 = _orthogonal_rotation(changing_segments["daily"][:, :k_use], fixed_segments["daily"][:, :k_use])
        q2, diag2 = _orthogonal_rotation(fixed_segments["daily"][:, :k_use], global_year_daily)
        q_total = q1 @ q2
        for segment in SEGMENT_ORDER:
            out[segment][year_mask, :] = np.asarray(changing_segments[segment][:, :k_use] @ q_total, dtype=float)
        diagnostics["years"].append(
            {
                "year": int(year),
                "n_symbols_unbalanced": int(changing_segments["n_symbols"]),
                "n_symbols_fixed": int(fixed_segments["n_symbols"]),
                "n_days": int(changing_segments["n_days"]),
                "step1": diag1,
                "step2": diag2,
                "composite_orthogonality_error": float(np.linalg.norm(q_total.T @ q_total - np.eye(q_total.shape[1]), ord="fro")),
            }
        )
    return out, diagnostics


def _build_figure13_data(
    result: Any,
    *,
    sample_dates: pd.DatetimeIndex,
    ffc_segmented: pd.DataFrame,
    pca_unbalanced: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    continuous = {
        "intraday": np.asarray(result.pipeline.F_cont_display_daily_intra, dtype=float),
        "overnight": np.asarray(result.pipeline.F_cont_display_daily_night, dtype=float),
        "daily": np.asarray(result.pipeline.F_cont_display_daily_total, dtype=float),
    }
    ffc_names = OFFICIAL_FFC_FACTORS
    ffc = {
        segment: _align_matrix(ffc_segmented.loc[ffc_segmented["segment_kind"].eq(segment), ["date", "factor", "ret"]], dates=sample_dates, factor_names=ffc_names)
        for segment in SEGMENT_ORDER
    }

    rows: List[Dict[str, Any]] = []
    continuous_names = [f"Factor {idx}" for idx in range(1, min(4, continuous["daily"].shape[1]) + 1)]
    factor_set_specs = [
        ("Continuous PCA", continuous, continuous_names),
    ]
    # Figure 13 second row: yearly estimated, yearly aligned changing-universe factors.
    if pca_unbalanced is not None:
        ub_names = [f"Factor {idx}" for idx in range(1, min(4, pca_unbalanced["daily"].shape[1]) + 1)]
        factor_set_specs.append(("Continuous PCA (unbalanced, yearly aligned)", pca_unbalanced, ub_names))
    factor_set_specs.append(("FFC 4-factor", ffc, ffc_names))
    for factor_set, mats, factor_names in factor_set_specs:
        # P11（论文 Figure 13 caption：“normalized by their DAILY standard deviation”）：
        # 同一因子的 intraday/overnight/daily 三段统一用【该因子日频 std】归一化，
        # 而不是各段用各自 std；前导缺失不再当 0（对齐首个有效日后再累计）。
        daily_mat = mats["daily"]
        factor_count = min(daily_mat.shape[1], len(factor_names), 4)
        daily_std = np.nanstd(daily_mat[:, :factor_count], axis=0, ddof=1)
        for segment in SEGMENT_ORDER:
            matrix = mats[segment]
            for factor_idx in range(factor_count):
                series = matrix[:, factor_idx]
                std = float(daily_std[factor_idx]) if factor_idx < daily_std.size else np.nan
                if not np.isfinite(std) or std <= 0:
                    normalized = np.zeros_like(series)
                else:
                    normalized = series / std
                # 前导/中间缺失：用 0 增量参与累计（不影响累计曲线的水平），
                # 但仅在该因子首个有效日之后开始累计，避免“缺失期当 0 收益”。
                valid = np.isfinite(normalized)
                cumulative = np.full(series.shape, np.nan, dtype=float)
                if valid.any():
                    first = int(np.argmax(valid))
                    filled = np.where(valid, normalized, 0.0)
                    cumulative[first:] = np.cumsum(filled[first:])
                rows.extend(
                    {
                        "date": sample_dates[pos],
                        "factor_set": factor_set,
                        "segment_kind": segment,
                        "factor": factor_names[factor_idx],
                        "normalized_cumulative_return": float(cumulative[pos]) if np.isfinite(cumulative[pos]) else np.nan,
                    }
                    for pos in range(len(sample_dates))
                )
    return pd.DataFrame(rows)


def _build_pricing_frame(
    *,
    asset_df: pd.DataFrame,
    sample_dates: pd.DatetimeIndex,
    factor_sets: Mapping[str, Mapping[str, np.ndarray]],
    rf_daily_sample: np.ndarray,
) -> pd.DataFrame:
    import prepareCore.engine as eng

    assets = sorted(asset_df["portfolio"].dropna().unique().tolist())
    rows: List[Dict[str, Any]] = []
    for segment in SEGMENT_ORDER:
        asset_matrix = (
            asset_df.loc[asset_df["segment_kind"].eq(segment)]
            .pivot(index="date", columns="portfolio", values="ret")
            .reindex(sample_dates)
            .reindex(columns=assets)
            .to_numpy(dtype=float)
        )
        if segment in {"overnight", "daily"}:
            asset_matrix = asset_matrix - np.asarray(rf_daily_sample, dtype=float)[:, None]
        for factor_set_name, matrices in factor_sets.items():
            factor_matrix = matrices[segment]
            Y, F = _drop_invalid_rows(asset_matrix, factor_matrix)
            if Y.size == 0 or F.size == 0:
                continue
            stats = eng.time_series_pricing(Y, F)
            for asset_idx, asset_name in enumerate(assets):
                rows.append(
                    {
                        "factor_set": factor_set_name,
                        "segment_kind": segment,
                        "asset": asset_name,
                        "expected_return": float(stats["avg_ret"][asset_idx]),
                        "predicted_return": float(stats["pred"][asset_idx]),
                        "alpha": float(stats["alpha"][asset_idx]),
                        "abs_alpha": float(abs(stats["alpha"][asset_idx])),
                        "r2": float(stats["R2"][asset_idx]),
                    }
                )
    return pd.DataFrame(rows)


def _build_payload(
    result: Any,
    *,
    proc_root: Path,
    external_root: Path,
    paper_tail_root: Path,
    weighting: str,
    discovered: Mapping[str, Path],
    mcap_files: Sequence[Path],
    scope: Mapping[str, Any],
    scope_signature: str,
) -> Dict[str, Any]:
    universe_summary = _load_json(discovered["universe_summary"])
    global_dates = pd.to_datetime(universe_summary["global_dates"])
    sample_dates = pd.DatetimeIndex(result.panel.dates)
    rf_daily = _load_rf_series(discovered["rf"])
    rf_daily_sample = rf_daily.reindex(sample_dates, fill_value=0.0).to_numpy(dtype=float)
    mom_daily = _load_mom_daily(discovered["mom_daily"]).reindex(global_dates, fill_value=0.0)
    ffc_external = _load_ffc_daily(discovered["ff3"], rf_daily, mom_daily)
    industry_map = _load_industry_mapping(discovered["industry_info"])

    symbol_files = sorted((proc_root / "symbol_returns").glob("*.npz"))
    symbols = [path.stem for path in symbol_files]
    mcap_matrix = _build_market_cap_matrix(mcap_files, dates=global_dates, symbols=symbols)

    all_stock_summary, industry_assets, market_returns = _build_full_market_assets(
        proc_root,
        global_dates=global_dates,
        sample_dates=sample_dates,
        rf_daily=rf_daily,
        industry_map=industry_map,
        mcap_matrix=mcap_matrix,
        weighting=weighting,
    )

    assignments = _load_assignments(discovered["size_value_assignments"])
    # legacy（平衡子集）保留：用于旧 validation 工件与回退。
    sv_legacy, daily_wide_vw, daily_wide_ew, size_value_meta = _build_size_value_assets(
        proc_root,
        assignments,
        sample_mode=str(getattr(result.panel, "sample_mode", "strict_balanced") or "strict_balanced"),
    )
    size_value_assets = sv_legacy
    size_value_source = "legacy_balanced_subset"
    if _size_value_full_market():
        try:
            from prepareCore.paper_fidelity import build_full_market_size_value
            size_value_assets = build_full_market_size_value(
                proc_root, assignments, global_dates=global_dates, mcap_matrix=mcap_matrix, symbols=symbols
            )
            size_value_source = "full_market"  # P9
        except Exception as exc:  # pragma: no cover - refresh 仅分钟级，失败回退不丢重结果
            print(f"[paper_fidelity] P9 full-market 2x3 失败，回退旧实现: {exc!r}")
            if _strict_final_export():
                raise RuntimeError("paper-tail size/value full-market build failed under strict final export mode") from exc
            size_value_assets = sv_legacy
            size_value_source = "legacy_fallback_after_error"
    # P10/D5：按 size/value 起点裁剪（2012 账面缺失 -> 2014-07-01）。
    _sv_start = _size_value_start()
    if _sv_start is not None and "date" in size_value_assets.columns:
        size_value_assets = size_value_assets.loc[size_value_assets["date"] >= _sv_start].reset_index(drop=True)
    size_value_validation, size_value_validation_summary = _validate_size_value(
        daily_wide_vw,
        daily_wide_ew,
        discovered["size_value_reference_vw"],
        discovered["size_value_reference_ew"],
    )
    # N8：把校验重定位为“与旧平衡口径的差异报告”，不再当作正确性证明。
    if isinstance(size_value_validation_summary, dict):
        size_value_validation_summary["size_value_source"] = size_value_source
        size_value_validation_summary["note"] = (
            "full_market 模式下，重建口径已不同于参考（平衡子集），差异预期非零，本表仅作差异报告。"
        )

    industry_selection = _select_industries_from_pca(result, industry_map)
    industry_factors = _build_industry_factor_frame(market_returns, industry_assets, industry_selection, dates=global_dates)

    # P7/P6/P8：分段 FFC。默认用股票级直接构造（carhart_daily）；失败回退旧残差法。
    ffc_source = "legacy_residual_allocation"
    try:
        if _ffc_mom_mode() == "carhart_daily":
            from prepareCore.paper_fidelity import build_full_market_momentum, build_ffc_segmented_clean, split_daily_rf
            rf_global = rf_daily.reindex(global_dates, fill_value=0.0).to_numpy(dtype=float)
            rf_split_global = split_daily_rf(rf_global)
            mom_segmented = build_full_market_momentum(
                proc_root, global_dates=global_dates, mcap_matrix=mcap_matrix, symbols=symbols
            )
            ffc_segmented_raw, ffc_segmented = build_ffc_segmented_clean(
                market_returns, size_value_assets, mom_segmented,
                dates=global_dates, rf_split={seg: rf_split_global[seg] for seg in SEGMENT_ORDER},
            )
            ffc_segment_reconciliation = pd.DataFrame()
            ffc_source = "stocklevel_carhart_clean"  # P7+P6+P8
        else:
            raise RuntimeError("ffc_mom_mode != carhart_daily -> 用旧实现")
    except Exception as exc:  # pragma: no cover
        if _ffc_mom_mode() == "carhart_daily":
            print(f"[paper_fidelity] P7 分段 FFC 失败，回退旧实现: {exc!r}")
        if _strict_final_export():
            raise RuntimeError("paper-tail FFC segmented build failed under strict final export mode") from exc
        ffc_segmented_raw, ffc_segmented, ffc_segment_reconciliation = _build_ffc_segmented_frames(
            market_returns,
            size_value_assets,
            ffc_external,
            dates=global_dates,
        )
    ffc_validation, ffc_validation_summary = _validate_ffc_daily(
        ffc_segmented.loc[ffc_segmented["date"].isin(sample_dates)],
        ffc_external,
        dates=sample_dates,
    )
    if isinstance(ffc_validation_summary, dict):
        ffc_validation_summary["ffc_source"] = ffc_source
        ffc_validation_summary["note"] = (
            "stocklevel 口径下 daily 不再被强制等于官方日频；本表是与官方日频的差异报告（N8）。"
        )

    balanced_summary = _summarize_balanced_panel(result, rf_daily_sample)
    figure12_data, figure12_filter = _build_figure12_data(
        balanced_summary,
        all_stock_summary,
        industry_assets,
        size_value_assets,
        sample_dates=sample_dates,
        rf_daily_sample=rf_daily_sample,
        weighting=weighting,
    )
    figure13_alignment_diag: Dict[str, Any] = {"factor_set_label": "Continuous PCA (unbalanced, yearly aligned)", "years": [], "status": "not_built"}
    pca_unbalanced = None
    try:
        pca_unbalanced, figure13_alignment_diag = _build_yearly_aligned_unbalanced_pca_segments(
            result,
            proc_root=proc_root,
            sample_dates=sample_dates,
            k=4,
        )
        figure13_alignment_diag["status"] = "built"
    except Exception as exc:  # pragma: no cover
        figure13_alignment_diag = {
            "factor_set_label": "Continuous PCA (unbalanced, yearly aligned)",
            "years": [],
            "status": "failed",
            "error": repr(exc),
        }
        print(f"[paper_tail] Figure 13 yearly aligned row unavailable, fallback to 2 rows: {exc!r}")
    figure13_data = _build_figure13_data(
        result,
        sample_dates=sample_dates,
        ffc_segmented=ffc_segmented.loc[ffc_segmented["date"].isin(sample_dates)],
        pca_unbalanced=pca_unbalanced,
    )

    result.paper_tail = {"industry_selection": industry_selection}
    table_iii = _build_table_iii(
        result,
        sample_dates=sample_dates,
        industry_factors=industry_factors.loc[industry_factors["date"].isin(sample_dates)],
        ffc_external=ffc_external.loc[ffc_external["date"].isin(sample_dates)],
        selected_industries=[item["industry"] for item in industry_selection.get("selected_industries", [])],
    )
    table_v = _build_table_v(
        result,
        sample_dates=sample_dates,
        rf_daily_sample=rf_daily_sample,
        industry_factors=industry_factors.loc[industry_factors["date"].isin(sample_dates)],
        ffc_segmented=ffc_segmented.loc[ffc_segmented["date"].isin(sample_dates)],
    )

    official_industry_assets = industry_assets.loc[industry_assets["weighting"].eq(weighting) & industry_assets["date"].isin(sample_dates)].copy()
    official_size_assets = size_value_assets.loc[size_value_assets["weighting"].eq(weighting) & size_value_assets["date"].isin(sample_dates)].copy()
    factor_sets, factor_matrix_diagnostics = _build_factor_sets(
        result,
        sample_dates=sample_dates,
        rf_daily_sample=rf_daily_sample,
        industry_factors=industry_factors.loc[industry_factors["date"].isin(sample_dates)],
        ffc_segmented=ffc_segmented.loc[ffc_segmented["date"].isin(sample_dates)],
    )
    pricing_factor_sets = {
        "Continuous PCA": factor_sets["Continuous PCA"],
        "FFC 4-factor": factor_sets["FFC 4-factor"],
    }
    pricing_industry = _build_pricing_frame(
        asset_df=official_industry_assets,
        sample_dates=sample_dates,
        factor_sets=pricing_factor_sets,
        rf_daily_sample=rf_daily_sample,
    )
    pricing_size_value = _build_pricing_frame(
        asset_df=official_size_assets,
        sample_dates=sample_dates,
        factor_sets=pricing_factor_sets,
        rf_daily_sample=rf_daily_sample,
    )

    return {
        "manifest": {
            "paper_tail_version": PAPER_TAIL_VERSION,
            "paper_tail_algorithm_version": PAPER_TAIL_ALGORITHM_VERSION,
            "scope_signature": scope_signature,
            "scope": dict(scope),
        },
        "figure12_data": figure12_data,
        "figure13_data": figure13_data,
        "pricing_industry": pricing_industry,
        "pricing_size_value": pricing_size_value,
        "table_iii": table_iii,
        "table_v": table_v,
        "industry_assets": industry_assets,
        "size_value_assets": size_value_assets,
        "ffc_external": ffc_external,
        "ffc_external_daily": ffc_external,
        "ffc_segmented_raw": ffc_segmented_raw,
        "ffc_segmented": ffc_segmented,
        "industry_factors": industry_factors,
        "industry_selection": industry_selection,
        "diagnostics": {
            "factor_matrix_diagnostics": factor_matrix_diagnostics,
            "figure12_all_stocks_filter": figure12_filter,
            "figure13_yearly_alignment": figure13_alignment_diag,
        },
        "validation": {
            "size_value_daily_parity": size_value_validation,
            "size_value_daily_parity_summary": {
                **size_value_validation_summary,
                "coverage_summary": size_value_meta["summary"],
            },
            "ffc_daily_validation": ffc_validation,
            "ffc_daily_validation_summary": ffc_validation_summary,
            "ffc_segment_reconciliation": ffc_segment_reconciliation,
        },
    }


def refresh_paper_tail_views(
    result: Any,
    *,
    proc_root: str | Path | None = None,
    external_data_root: str | Path | None = None,
    paper_tail_root: str | Path | None = None,
    paper_tail_weighting: str = "value_weighted",
    refresh_paper_tail: bool = True,
    strict_final_export: bool | None = None,
) -> Dict[str, Any]:
    if not refresh_paper_tail:
        return getattr(result, "paper_tail", {}) if hasattr(result, "paper_tail") else {}
    if strict_final_export is not None:
        os.environ["PELGER_STRICT_FINAL_EXPORT"] = "1" if strict_final_export else "0"
    proc_root_path = _ensure_path(proc_root, default=_repo_root() / "Data" / "proc_Data" / "pelger_cn_adjusted")
    external_root_path = _ensure_path(external_data_root, default=default_external_data_root())
    paper_tail_root_path = _ensure_path(paper_tail_root, default=default_paper_tail_root(proc_root_path))
    if paper_tail_weighting not in {"value_weighted", "equal_weighted"}:
        raise ValueError(f"Unsupported paper_tail_weighting: {paper_tail_weighting}")
    existing = getattr(result, "paper_tail", None)
    existing_manifest = existing.get("manifest", {}) if isinstance(existing, dict) else {}
    existing_scope = dict(existing_manifest.get("scope", {})) if isinstance(existing_manifest, dict) else {}
    if isinstance(existing, dict) and _manifest_is_compatible(
        existing_manifest,
        external_root=external_root_path,
        paper_tail_root=paper_tail_root_path,
        weighting=paper_tail_weighting,
    ) and existing_scope.get("industry_info_filename") == _industry_info_filename() and existing_scope.get(
        "industry_mapping_filename"
    ) == _industry_mapping_filename() and bool(existing_scope.get("strict_final_export", False)) == bool(_strict_final_export()):
        return existing

    payload = _discover_or_build_payload(
        result,
        proc_root=proc_root_path,
        external_root=external_root_path,
        paper_tail_root=paper_tail_root_path,
        weighting=paper_tail_weighting,
    )
    result.paper_tail = payload
    return payload


def _short_asset_label(name: Any) -> str:
    text = str(name)
    if text in SHORT_LABEL_MAP:
        return SHORT_LABEL_MAP[text]
    if text in {"BH", "BL", "BM", "SH", "SL", "SM"}:
        return text
    return _short_asset_label_ascii(text)


def _short_asset_label_ascii(text: str) -> str:
    compact = (
        text.replace("Industry Portfolios", "")
        .replace("Portfolio", "")
        .replace("portfolio", "")
        .replace("industry", "")
        .replace("Industry", "")
        .replace(" & ", "/")
        .replace(" and ", "/")
        .strip(" /")
    )
    return compact if len(compact) <= 12 else compact[:12]
    """
    compact = (
        text.replace("行业组合", "")
        .replace("组合", "")
        .replace("行业", "")
        .replace("与", "/")
        .replace("及", "/")
        .strip()
    )
    return compact if len(compact) <= 12 else compact[:12]


"""

def render_figure_12(result: Any, output_path: Path, title: str) -> None:
    from prepareCore.engine import _atomic_save_figure, _save_placeholder_figure

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(exc) from exc

    payload = getattr(result, "paper_tail", {}) or {}
    df = payload.get("figure12_data")
    if df is None or df.empty:
        _save_placeholder_figure(output_path, title, "Paper-tail Figure 12 data are not available.")
        return

    paper_groups = [
        ("all_stocks", "All Stocks"),
        ("industry_portfolios", "{n_industry} Industry Portfolios"),
        ("size_value_portfolios", "6 Size/Value Portfolios"),
    ]
    n_industry = 0
    if isinstance(df, pd.DataFrame) and "group" in df.columns:
        ind = df.loc[df["group"].eq("industry_portfolios"), "asset"]
        n_industry = int(ind.nunique()) if not ind.empty else 0

    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.0), sharex=False, sharey=False)
    for ax, (group, gtitle) in zip(np.atleast_1d(axes).flat, paper_groups):
        sub = df.loc[df["group"].eq(group)].copy()
        if group == "all_stocks" and "eligible_for_plot" in sub.columns:
            keep = sub["eligible_for_plot"]
            if keep.dtype != bool:
                keep = keep.astype(str).str.lower().isin({"1", "true", "yes"})
            sub = sub.loc[keep].copy()
        title_group = gtitle.format(n_industry=n_industry) if "{n_industry}" in gtitle else gtitle
        if sub.empty:
            ax.set_title(title_group)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        xs = sub["mean_intraday_excess"].to_numpy(dtype=float)
        ys = sub["mean_overnight_excess"].to_numpy(dtype=float)
        ax.scatter(
            xs,
            ys,
            facecolors="none",
            edgecolors="#1f5fbf",
            linewidths=1.0,
            s=30 if len(sub) > 20 else 55,
            alpha=0.85,
        )
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)
        good = np.isfinite(xs) & np.isfinite(ys)
        if good.sum() >= 2 and np.ptp(xs[good]) > 0:
            b1, b0 = np.polyfit(xs[good], ys[good], 1)
            xline = np.array([xs[good].min(), xs[good].max()])
            ax.plot(xline, b0 + b1 * xline, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_title(title_group)
        ax.set_xlabel("Expected intraday excess return")
        ax.set_ylabel("Expected overnight excess return")
        label_map = _asset_label_map(sub["asset"]) if group == "industry_portfolios" else {}
        if group != "all_stocks" or len(sub) <= 16:
            for label_idx, (_, row) in enumerate(sub.iterrows()):
                _annotate_scatter_label(
                    ax,
                    _plot_asset_label(row["asset"], label_map),
                    float(row["mean_intraday_excess"]),
                    float(row["mean_overnight_excess"]),
                    label_idx,
                    center=(
                        float(np.nanmedian(xs[good])) if good.any() else float(np.nanmean(xs)),
                        float(np.nanmedian(ys[good])) if good.any() else float(np.nanmean(ys)),
                    ),
                    fontsize=7,
                )
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=2.0)
    _atomic_save_figure(fig, output_path, dpi=170)
    plt.close(fig)


def render_figure_13(result: Any, output_path: Path, title: str) -> None:
    from prepareCore.engine import _atomic_save_figure, _save_placeholder_figure

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(exc) from exc

    payload = getattr(result, "paper_tail", {}) or {}
    df = payload.get("figure13_data")
    if df is None or df.empty:
        _save_placeholder_figure(output_path, title, "Paper-tail Figure 13 data are not available.")
        return

    seg_cols = ["intraday", "overnight", "daily"]
    seg_label = {"intraday": "Intraday", "overnight": "Overnight", "daily": "Daily"}
    preferred = [
        "Continuous PCA",
        "Continuous PCA (unbalanced, yearly aligned)",
        "FFC 4-factor",
    ]
    present = list(df["factor_set"].unique()) if "factor_set" in df.columns else []
    factor_sets = [factor_set for factor_set in preferred if factor_set in present] or preferred
    fig, axes = plt.subplots(len(factor_sets), 3, figsize=(16.8, 4.8 * len(factor_sets)), sharex=True)
    axes = np.atleast_2d(axes)
    for row_idx, factor_set in enumerate(factor_sets):
        for col_idx, segment in enumerate(seg_cols):
            ax = axes[row_idx, col_idx]
            sub = df.loc[df["segment_kind"].eq(segment) & df["factor_set"].eq(factor_set)].copy()
            ax.set_title(f"{factor_set} | {seg_label[segment]}", fontsize=9)
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            for factor_name, factor_df in sub.groupby("factor", sort=False):
                ax.plot(
                    pd.to_datetime(factor_df["date"]),
                    factor_df["normalized_cumulative_return"],
                    linewidth=1.4,
                    label=str(factor_name),
                )
            ax.grid(True, alpha=0.2)
            if row_idx == len(factor_sets) - 1:
                ax.set_xlabel("Time")
            if col_idx == 0:
                ax.set_ylabel("Return")
            ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _atomic_save_figure(fig, output_path, dpi=170)
    plt.close(fig)


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


def _asset_label_map(values: Any) -> Dict[str, str]:
    try:
        unique = sorted({str(x) for x in pd.Series(values).dropna().tolist()})
    except Exception:
        unique = []
    mapping: Dict[str, str] = {}
    for idx, raw in enumerate(unique, start=1):
        if raw in {"BH", "BL", "BM", "SH", "SL", "SM"}:
            mapping[raw] = raw
        elif raw in _INDUSTRY_EN_LABELS:
            mapping[raw] = f"I{idx:02d}"
        else:
            try:
                raw.encode("ascii")
                mapping[raw] = _short_asset_label(raw)
            except UnicodeEncodeError:
                mapping[raw] = f"IND{idx:02d}"
    return mapping


def _plot_asset_label(asset: Any, label_map: Mapping[str, str]) -> str:
    raw = str(asset)
    if raw in label_map:
        return label_map[raw]
    return _short_asset_label(raw)


def _annotate_scatter_label(
    ax: Any,
    label: str,
    x: float,
    y: float,
    index: int,
    *,
    fontsize: float = 7.0,
    center: Optional[Tuple[float, float]] = None,
) -> None:
    if center is None or not np.isfinite(center[0]) or not np.isfinite(center[1]):
        angle = (int(index) % 12) * (2.0 * np.pi / 12.0)
    else:
        x0, y0 = center
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xscale = max(abs(xlim[1] - xlim[0]), 1e-12)
        yscale = max(abs(ylim[1] - ylim[0]), 1e-12)
        vx = (float(x) - float(x0)) / xscale
        vy = (float(y) - float(y0)) / yscale
        if abs(vx) + abs(vy) < 1e-9:
            angle = (int(index) % 12) * (2.0 * np.pi / 12.0)
        else:
            angle = float(np.arctan2(vy, vx))
    radius = 16.0 + 3.0 * (int(index) % 3)
    tangent = [-8.0, 0.0, 8.0, -13.0, 13.0][int(index) % 5]
    dx = float(np.cos(angle) * radius - np.sin(angle) * tangent)
    dy = float(np.sin(angle) * radius + np.cos(angle) * tangent)
    ax.annotate(
        str(label),
        (x, y),
        fontsize=fontsize,
        alpha=0.88,
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        arrowprops={
            "arrowstyle": "-",
            "color": "0.55",
            "linewidth": 0.45,
            "alpha": 0.65,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.62},
    )


def _render_pricing_figure(df: pd.DataFrame, output_path: Path, title: str) -> None:
    from prepareCore.engine import _atomic_save_figure, _save_placeholder_figure

    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(exc) from exc

    if df.empty:
        _save_placeholder_figure(output_path, title, "No pricing data are available.")
        return

    factor_sets = ["Continuous PCA", "FFC 4-factor"]
    seg_cols = ["daily", "intraday", "overnight"]
    seg_label = {"daily": "Daily", "intraday": "Intraday", "overnight": "Overnight"}
    set_label = {"Continuous PCA": "Continuous PCA", "FFC 4-factor": "FFC"}
    asset_label_map = _asset_label_map(df["asset"]) if "asset" in df.columns else {}

    fig = plt.figure(figsize=(17.2, 16.8))
    grid = GridSpec(5, 3, figure=fig, height_ratios=[1.0, 1.0, 0.12, 1.0, 1.0], hspace=0.55, wspace=0.32)
    axes_a = [[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)]
    axes_b = [[fig.add_subplot(grid[r + 3, c]) for c in range(3)] for r in range(2)]

    for r, factor_set in enumerate(factor_sets):
        for c, segment in enumerate(seg_cols):
            ax = axes_a[r][c]
            sub = df.loc[df["segment_kind"].eq(segment) & df["factor_set"].eq(factor_set)].copy()
            ax.set_title(f"{seg_label[segment]} | {set_label[factor_set]}", fontsize=9)
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            ex = sub["expected_return"].to_numpy(dtype=float)
            pr = sub["predicted_return"].to_numpy(dtype=float)
            lo = float(np.nanmin(np.concatenate([ex, pr])))
            hi = float(np.nanmax(np.concatenate([ex, pr])))
            pad = max(1e-4, 0.08 * max(abs(lo), abs(hi), 1e-4))
            ax.scatter(ex, pr, facecolors="none", edgecolors="#1f5fbf", linewidths=1.0, s=42, alpha=0.85)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Expected return")
            ax.set_ylabel("Predicted return")
            ax.grid(True, alpha=0.2)
            if len(sub) <= 12:
                for label_idx, (_, row) in enumerate(sub.iterrows()):
                    _annotate_scatter_label(
                        ax,
                        _plot_asset_label(row["asset"], asset_label_map),
                        float(row["expected_return"]),
                        float(row["predicted_return"]),
                        label_idx,
                        center=(float(np.nanmedian(ex)), float(np.nanmedian(pr))),
                        fontsize=6.5,
                    )

    for r, factor_set in enumerate(factor_sets):
        for c, segment in enumerate(seg_cols):
            ax = axes_b[r][c]
            sub = df.loc[df["segment_kind"].eq(segment) & df["factor_set"].eq(factor_set)].copy()
            ax.set_title(f"{seg_label[segment]} | {set_label[factor_set]}", fontsize=9)
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            if set(sub["asset"].astype(str)) == {"BH", "BL", "BM", "SH", "SL", "SM"}:
                order = ["BH", "BL", "BM", "SH", "SL", "SM"]
                sub = sub.set_index("asset").reindex(order).reset_index()
            else:
                sub = sub.sort_values("asset").reset_index(drop=True)
            xpos = np.arange(len(sub))
            ax.bar(xpos, sub["alpha"].to_numpy(dtype=float), color="#1f5fbf", alpha=0.85)
            ax.axhline(0.0, color="0.4", linewidth=0.9)
            ax.set_ylabel("Pricing error")
            ax.set_xticks(xpos)
            ax.set_xticklabels([_plot_asset_label(asset, asset_label_map) for asset in sub["asset"].tolist()], fontsize=7, rotation=40 if len(sub) > 8 else 0, ha="right" if len(sub) > 8 else "center")
            ax.grid(True, axis="y", alpha=0.2)

    title_lines = str(title).count("\n") + 1
    panel_a_y = 0.940 if title_lines > 1 else 0.965
    top_rect = 0.900 if title_lines > 1 else 0.935
    fig.text(0.5, panel_a_y, "Panel A: Predicted Returns", ha="center", fontsize=12, weight="bold")
    fig.text(0.5, 0.462, "Panel B: Pricing Errors", ha="center", fontsize=12, weight="bold")
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.06, top=top_rect, hspace=0.75, wspace=0.32)
    _atomic_save_figure(fig, output_path, dpi=170)
    plt.close(fig)


def render_figure_14(result: Any, output_path: Path, title: str) -> None:
    payload = getattr(result, "paper_tail", {}) or {}
    df = payload.get("pricing_industry")
    _render_pricing_figure(df if isinstance(df, pd.DataFrame) else pd.DataFrame(), output_path, title)


def render_figure_15(result: Any, output_path: Path, title: str) -> None:
    payload = getattr(result, "paper_tail", {}) or {}
    df = payload.get("pricing_size_value")
    full_title = f"{title}\nA-share reconstructed 2x3 portfolios; sample starts 2014-07-01"
    _render_pricing_figure(df if isinstance(df, pd.DataFrame) else pd.DataFrame(), output_path, full_title)


def build_replication_coverage_report() -> pd.DataFrame:
    rows = [
        ("Table I", "Summary Statistics for Continuous and Jump Returns", "implemented_adapted", "Balanced and unbalanced yearly jump/continuous tables come from the main replication cache."),
        ("Table II", "Balanced and Unbalanced Panel Results", "implemented_adapted", "Balanced vs unbalanced generalized-correlation tables come from the main replication cache."),
        ("Table III", "Generalized Correlations with Industry and FFC Factors", "implemented", "Uses supplement-backed paper_tail industry portfolios and FFC factor comparisons."),
        ("Table IV", "Time-Variation Decomposition across Frequencies", "implemented_adapted", "Rolling generalized-correlation and explained-variation diagnostics come from the main replication cache."),
        ("Table V", "Intraday / Overnight / Daily Sharpe Ratios", "implemented", "Uses canonical excess-return factor sets plus the first four continuous PCA factors."),
        ("Figure 1", "Number of HF Factors, Unbalanced Panel", "implemented_adapted", "Submission version uses cross-year unbalanced but within-year balanced yearly panels."),
        ("Figure 2", "Number of HF Factors, Balanced Panel", "implemented_adapted", "Submission version uses fixed full-sample intersection yearly slices with constant N."),
        ("Figure 3", "Proxy Factor Portfolio Weights", "omitted_from_submission", "Removed from the submission-core export profile."),
        ("Figure 4", "Continuous PCA Factor Portfolio Weights", "implemented_adapted", "Rendered from refreshed display-layer continuous PCA weights."),
        ("Figure 5", "Monthly PCA Factor Portfolio Weights", "omitted_from_submission", "Removed from the submission-core export profile."),
        ("Figure 6", "Time Variation in Loadings", "omitted_from_submission", "Removed from the submission-core export profile."),
        ("Figure 7", "Locally Estimated Continuous Factors", "implemented_adapted", "Submission version emphasizes 7 continuous factors over a 21-trading-day local window."),
        ("Figure 8", "Time-Varying Portfolio Weights", "omitted_from_submission", "Removed from the submission-core export profile."),
        ("Figure 9", "Time-Varying Explained Variation", "omitted_from_submission", "Removed from the submission-core export profile."),
        ("Figure 10", "Factor-Structure Time Variation Decomposition", "implemented_adapted", "Rendered from rolling generalized-correlation and explained-variation diagnostics."),
        ("Figure 11", "Continuous Factor-Structure Decomposition", "omitted_from_submission", "Removed from the submission-core export profile."),
        ("Figure 12", "Expected Intraday and Overnight Returns", "implemented", "Uses balanced-panel stocks, filtered all-stocks scatter points, 14 industry portfolios, and 6 size/value portfolios."),
        ("Figure 13", "Normalized Cumulative Factor Returns", "implemented", "Uses yearly estimated and yearly aligned changing-universe continuous PCA factors plus canonical segmented FFC returns."),
        ("Figure 14", "Asset Pricing of Industry Portfolios", "implemented", "Uses supplement-backed industry portfolios with continuous PCA and FFC pricing tests."),
        ("Figure 15", "Asset Pricing of Size- and Value-Sorted Portfolios", "implemented", "Uses supplement-backed 2x3 size/value portfolios with continuous PCA and FFC pricing tests."),
    ]
    return pd.DataFrame(rows, columns=["paper_item", "paper_content", "status", "notes"])


__all__ = [
    "build_replication_coverage_report",
    "default_external_data_root",
    "default_paper_tail_root",
    "refresh_paper_tail_views",
    "render_figure_12",
    "render_figure_13",
    "render_figure_14",
    "render_figure_15",
]
