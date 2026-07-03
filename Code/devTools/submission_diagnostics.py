from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from prepareCore.config import RunConfig
from prepareCore.engine import (
    PAPER_LENIENT_SAMPLE,
    STRICT_BALANCED_SAMPLE,
    PelgerPipeline,
    ReplicationResult,
    _factor_count_rows_for_panel,
    detect_jumps,
    load_proc_hf_panel,
)
from prepareCore.paper_tail import (
    _build_figure13_data,
    _build_table_v,
    _load_rf_series,
    _build_table_iii,
)
from prepareCore.submission_fast import (
    _rf_file,
    _submission_fast_diagnostics_dir,
    load_legacy_paper_tail_assets,
    write_strict_panel_industry_composition,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {path}")


def _representative_years(cfg: RunConfig) -> List[int]:
    if cfg.years:
        years = sorted(int(year) for year in cfg.years)
    else:
        years = sorted({pd.Timestamp(date).year for date in load_proc_hf_panel(
            proc_root=cfg.proc_root,
            sample_mode=STRICT_BALANCED_SAMPLE,
            years=None,
            return_mode=cfg.return_mode,
            max_stocks=cfg.max_stocks,
        ).dates})
    wanted = [2015, 2020, 2023]
    return [year for year in wanted if year in years] or years[:3]


def build_gfn_sensitivity(cfg: RunConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for year in _representative_years(cfg):
        panel = load_proc_hf_panel(
            proc_root=cfg.proc_root,
            sample_mode=STRICT_BALANCED_SAMPLE,
            years=[int(year)],
            return_mode=cfg.return_mode,
            max_stocks=cfg.max_stocks,
        )
        for g_fn in ("median_sqrtN", "logN", "none"):
            rows.extend(
                _factor_count_rows_for_panel(
                    panel,
                    panel_block="Yearwise balanced, changing universe",
                    jump_a=float(cfg.jump_a),
                    k_max=int(cfg.k_max),
                    gamma=float(cfg.gamma),
                    g_fn=g_fn,
                )
            )
    return pd.DataFrame(rows).sort_values(["year", "return_component", "g_fn"]).reset_index(drop=True)


def build_jump_proxy_diagnostics(cfg: RunConfig) -> pd.DataFrame:
    panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        years=cfg.years,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    _, jump = detect_jumps(panel, a=float(cfg.jump_a))
    jump_mask = np.isfinite(jump) & (np.abs(jump) > 0)
    returns = np.asarray(panel.R_5min_full, dtype=float)
    proxy_threshold = float(np.log1p(0.095))
    proxy_mask = np.isfinite(returns) & (np.abs(returns) >= proxy_threshold)
    rows: List[Dict[str, Any]] = []
    for day_idx, date in enumerate(panel.dates):
        bar_mask = panel.day_ids == int(day_idx)
        jm = jump_mask[bar_mask]
        pm = proxy_mask[bar_mask]
        jump_count = int(jm.sum())
        rows.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "year": int(pd.Timestamp(date).year),
                "jump_increment_count": jump_count,
                "proxy_limit_like_count": int((jm & pm).sum()),
                "proxy_share_of_jumps": float((jm & pm).sum() / jump_count) if jump_count else np.nan,
                "proxy_method": "abs_5min_log_return_ge_log1p_0.095",
            }
        )
    daily = pd.DataFrame(rows)
    yearly = (
        daily.groupby("year", as_index=False)
        .agg(
            jump_increment_count=("jump_increment_count", "sum"),
            proxy_limit_like_count=("proxy_limit_like_count", "sum"),
        )
        .reset_index(drop=True)
    )
    yearly["proxy_share_of_jumps"] = yearly["proxy_limit_like_count"] / yearly["jump_increment_count"].replace(0, np.nan)
    yearly["proxy_method"] = "abs_5min_log_return_ge_log1p_0.095"
    return yearly


def build_lenient_robustness(cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=PAPER_LENIENT_SAMPLE,
        years=cfg.years,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    pipeline = PelgerPipeline(
        panel=panel,
        jump_a=cfg.jump_a,
        K_max=cfg.k_max,
        gamma=cfg.gamma,
        g_fn=cfg.g_fn,
    ).run_full()
    print(f"[INFO] built lenient robustness pipeline: N={panel.N}, D={panel.D}, K_cont={pipeline.K_cont_hat}")
    result = ReplicationResult(
        universe=pd.DataFrame(),
        universe_summary={},
        panel=panel,
        pipeline=pipeline,
        rolling_gc=np.zeros((0, 0), dtype=float),
        rolling_explained_variation=np.zeros(0, dtype=float),
        robustness=pd.DataFrame(),
        output_root=Path(cfg.final_result_root),
        runtime_root=Path(cfg.runtime_root) / "submission_fast",
    )
    legacy = load_legacy_paper_tail_assets(cfg.paper_tail_root, weighting=cfg.paper_tail_weighting)
    sample_dates = pd.DatetimeIndex(panel.dates)
    rf_daily = _load_rf_series(_rf_file(Path(cfg.external_data_root)))
    rf_daily_sample = rf_daily.reindex(sample_dates, fill_value=0.0).to_numpy(dtype=float)
    figure13_data = _build_figure13_data(
        result,
        sample_dates=sample_dates,
        ffc_segmented=legacy["ffc_segmented"],
        pca_unbalanced=None,
    )
    result.paper_tail = {
        "figure13_data": figure13_data,
        "industry_selection": legacy.get("industry_selection", {}),
    }
    ffc_external = legacy.get("ffc_external")
    if ffc_external is None:
        ffc_external = legacy.get("ffc_external_daily")
    if ffc_external is None:
        raise KeyError("ffc_external or ffc_external_daily")
    table_iii = _build_table_iii(
        result,
        sample_dates=sample_dates,
        industry_factors=legacy["industry_factors"].loc[legacy["industry_factors"]["date"].isin(sample_dates)],
        ffc_external=ffc_external.loc[ffc_external["date"].isin(sample_dates)],
        selected_industries=list(cfg.industry_factors_frozen or ()),
    )
    table_v = _build_table_v(
        result,
        sample_dates=sample_dates,
        rf_daily_sample=rf_daily_sample,
        industry_factors=legacy["industry_factors"].loc[legacy["industry_factors"]["date"].isin(sample_dates)],
        ffc_segmented=legacy["ffc_segmented"].loc[legacy["ffc_segmented"]["date"].isin(sample_dates)],
    )
    table_iii.insert(0, "robustness_sample", f"{PAPER_LENIENT_SAMPLE}_N{panel.N}")
    table_v.insert(0, "robustness_sample", f"{PAPER_LENIENT_SAMPLE}_N{panel.N}")
    table_iii.insert(1, "diagnostic_note", "lenient minimal PCA; legacy paper_tail assets reused")
    table_v.insert(1, "diagnostic_note", "lenient minimal PCA; legacy paper_tail assets reused")
    return table_iii, table_v


def main() -> int:
    cfg = RunConfig()
    cfg.export_fidelity_env()
    diag_dir = _submission_fast_diagnostics_dir(cfg)
    strict_panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        years=cfg.years,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    write_strict_panel_industry_composition(cfg, strict_panel, diag_dir)
    _write_csv(diag_dir / "g_fn_sensitivity_3x3.csv", build_gfn_sensitivity(cfg))
    _write_csv(diag_dir / "jump_limit_proxy_diagnostics.csv", build_jump_proxy_diagnostics(cfg))
    try:
        table_iii, table_v = build_lenient_robustness(cfg)
        _write_csv(diag_dir / "Table_III_robustness_lenient391.csv", table_iii)
        _write_csv(diag_dir / "Table_V_robustness_lenient391.csv", table_v)
    except Exception as exc:
        (diag_dir / "lenient_robustness_error.txt").write_text(repr(exc), encoding="utf-8")
        print(f"[WARN] lenient robustness skipped: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
