from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.config import RunConfig
from core.logging_utils import log_warn


TABLE_FILES = {
    "I": "Table_I_summary_statistics_for_continuous_and_jump_returns.csv",
    "II": "Table_II_balanced_and_unbalanced_panel_results.csv",
    "III": "Table_III_generalized_correlations_with_industry_and_ffc_factors.csv",
    "V": "Table_V_intraday_overnight_daily_sharpe_ratios.csv",
}


def diagnostics_dir(cfg: RunConfig) -> Path:
    from core.submission_fast import submission_fast_runtime_root

    path = submission_fast_runtime_root(cfg) / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(cfg: RunConfig, roman: str) -> Path:
    path = Path(cfg.final_result_root) / "tables" / TABLE_FILES[roman]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_table_with_fallback(path: Path, df: pd.DataFrame, fallback_suffix: str, tag: str) -> Path:
    try:
        write_csv(path, df)
        return path
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}_{fallback_suffix}.csv")
        write_csv(fallback_path, df)
        log_warn(tag, f"Canonical table appears to be locked; wrote fallback table to {fallback_path}.")
        return fallback_path
