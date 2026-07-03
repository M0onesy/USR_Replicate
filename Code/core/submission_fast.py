from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from core.config import RunConfig
from core.engine import (
    DEFAULT_PROC_ROOT,
    STRICT_BALANCED_SAMPLE,
    PelgerPipeline,
    ReplicationResult,
    _atomic_to_csv,
    _panel_pca,
    _table_i_rows_for_panel,
    _rolling_output_frames,
    build_submission_figure1_factor_counts,
    build_submission_figure2_factor_counts,
    detect_jumps,
    generalized_correlations,
    load_proc_hf_panel,
    refresh_replication_result_views,
    rolling_gc_and_explained_variation,
    subset_panel_by_years,
)
from core.logging_utils import log_info, log_step, log_warn
from core.paper_tail import (
    OFFICIAL_FFC_FACTORS,
    SEGMENT_ORDER,
    _annualization_days,
    _align_matrix,
    _build_figure13_data,
    _build_pricing_frame,
    _build_yearly_aligned_unbalanced_pca_segments,
    _drop_invalid_rows,
    _load_rf_series,
    _portfolio_sharpes_from_excess_matrices,
    _to_excess_matrices,
)
from core.runner import run_generator


SUBMISSION_FAST_FIGURES: Tuple[Tuple[str, str], ...] = (
    ("fig1", "figCode.figure_01"),
    ("fig2", "figCode.figure_02"),
    ("fig4", "figCode.figure_04"),
    ("fig7", "figCode.figure_07"),
    ("fig10", "figCode.figure_10"),
    ("fig12", "figCode.figure_12"),
    ("fig13", "figCode.figure_13"),
    ("fig14", "figCode.figure_14"),
    ("fig15", "figCode.figure_15"),
)

SUBMISSION_TABLE_I_FILE = "Table_I_summary_statistics_for_continuous_and_jump_returns.csv"
SUBMISSION_TABLE_II_FILE = "Table_II_balanced_and_unbalanced_panel_results.csv"
SUBMISSION_TABLE_III_FILE = "Table_III_generalized_correlations_with_industry_and_ffc_factors.csv"
SUBMISSION_TABLE_V_FILE = "Table_V_intraday_overnight_daily_sharpe_ratios.csv"


def submission_fast_runtime_root(cfg: RunConfig) -> Path:
    return Path(cfg.runtime_root) / "submission_fast"


def _submission_fast_diagnostics_dir(cfg: RunConfig) -> Path:
    path = submission_fast_runtime_root(cfg) / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _load_csv(path: Path, *, parse_dates: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=list(parse_dates or ()))


def _table_i_output_path(cfg: RunConfig) -> Path:
    path = Path(cfg.final_result_root) / "tables" / SUBMISSION_TABLE_I_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _table_ii_output_path(cfg: RunConfig) -> Path:
    path = Path(cfg.final_result_root) / "tables" / SUBMISSION_TABLE_II_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _table_iii_output_path(cfg: RunConfig) -> Path:
    path = Path(cfg.final_result_root) / "tables" / SUBMISSION_TABLE_III_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _table_v_output_path(cfg: RunConfig) -> Path:
    path = Path(cfg.final_result_root) / "tables" / SUBMISSION_TABLE_V_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _rf_file(external_root: Path) -> Path:
    import os as _os

    candidates: List[Path] = []
    env_path = str(_os.environ.get("PELGER_RF_FILE", "")).strip()
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


def _long_to_factor_mats(sample_dates: pd.DatetimeIndex, factor_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    mats: Dict[str, np.ndarray] = {}
    subset = factor_df.loc[:, ["date", "factor", "segment_kind", "ret"]].copy()
    for segment in SEGMENT_ORDER:
        seg = subset.loc[subset["segment_kind"].eq(segment), ["date", "factor", "ret"]].copy()
        mats[segment] = _align_matrix(seg, dates=sample_dates, factor_names=OFFICIAL_FFC_FACTORS)
    return mats


def _rolling_signature_matches(cfg: RunConfig) -> bool:
    run_state_path = Path(cfg.runtime_root) / "checkpoints" / "run_state.json"
    if not run_state_path.exists():
        return False
    try:
        state = _read_json(run_state_path)
    except Exception:
        return False
    payload = (((state.get("signature") or {}).get("semantic_payload")) or {})
    expected = {
        "proc_root": str(Path(cfg.proc_root).resolve()),
        "balanced_mode": str(cfg.balanced_mode),
        "jump_a": float(cfg.jump_a),
        "k_max": int(cfg.k_max),
        "gamma": float(cfg.gamma),
        "g_fn": str(cfg.g_fn),
        "return_mode": str(cfg.return_mode),
    }
    try:
        actual = {
            "proc_root": str(Path(payload.get("proc_root", "")).resolve()),
            "balanced_mode": str(payload.get("balanced_mode")),
            "jump_a": float(payload.get("jump_a")),
            "k_max": int(payload.get("k_max")),
            "gamma": float(payload.get("gamma")),
            "g_fn": str(payload.get("g_fn")),
            "return_mode": str(payload.get("return_mode")),
        }
    except Exception:
        return False
    if actual != expected:
        return False
    completed = state.get("completed_rolling_chunks") or []
    total = state.get("rolling_total_chunks") or 0
    return bool(completed) and len(completed) == int(total)


def _persist_rolling_outputs(
    diagnostics_dir: Path,
    rolling_gc: np.ndarray,
    rolling_ev: np.ndarray,
) -> None:
    rolling_gc_df, rolling_ev_df = _rolling_output_frames(rolling_gc, rolling_ev)
    _atomic_to_csv(rolling_gc_df, diagnostics_dir / "rolling_gc.csv", index=False, encoding="utf-8-sig")
    _atomic_to_csv(rolling_ev_df, diagnostics_dir / "rolling_explained_variation.csv", index=False, encoding="utf-8-sig")


def _load_or_build_rolling_outputs(
    cfg: RunConfig,
    *,
    diagnostics_dir: Path,
    pipeline: PelgerPipeline,
) -> Tuple[np.ndarray, np.ndarray, str]:
    rolling_dir = Path(cfg.runtime_root) / "checkpoints" / "rolling"
    if not _rolling_signature_matches(cfg) or not rolling_dir.exists():
        log_warn("submission_fast", "No compatible strict rolling checkpoints were found; recomputing strict rolling diagnostics.")
        rolling_window = 21 if pipeline.panel.D >= 21 else max(5, pipeline.panel.D // 3)
        workers = int(cfg.rolling_workers or cfg.workers or 1)
        rolling_gc, rolling_ev = rolling_gc_and_explained_variation(
            R=pipeline.R_cont,
            day_ids=pipeline.panel.day_ids,
            window_days=max(rolling_window, 2),
            K=pipeline.K_cont_hat,
            global_Lambda=pipeline.pca_cont.Lambda,
            step_days=1,
            workers=max(1, workers),
        )
        _persist_rolling_outputs(diagnostics_dir, rolling_gc, rolling_ev)
        return rolling_gc, rolling_ev, "recomputed"

    window_records: List[Tuple[int, np.ndarray, float]] = []
    for path in sorted(rolling_dir.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            window_index = np.asarray(payload.get("window_index", np.zeros(0, dtype=np.int32)), dtype=np.int32)
            gc = np.asarray(payload.get("gc", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
            explained = np.asarray(payload.get("explained_variation", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        for idx, win in enumerate(window_index.tolist()):
            window_records.append((int(win), np.asarray(gc[idx], dtype=np.float64), float(explained[idx])))

    window_records.sort(key=lambda item: int(item[0]))
    if not window_records:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64), "empty"

    rolling_gc = np.vstack([item[1] for item in window_records])
    rolling_ev = np.asarray([item[2] for item in window_records], dtype=np.float64)
    _persist_rolling_outputs(diagnostics_dir, rolling_gc, rolling_ev)
    return rolling_gc, rolling_ev, "checkpoint"


def load_legacy_paper_tail_assets(
    paper_tail_root: str | Path,
    *,
    weighting: str,
) -> Dict[str, pd.DataFrame]:
    root = Path(paper_tail_root).resolve()
    data = {
        "figure12_data": _load_csv(root / "figures" / "figure12_data.csv"),
        "figure13_data_legacy": _load_csv(root / "figures" / "figure13_data.csv", parse_dates=["date"]),
        "industry_assets": _load_csv(root / "assets" / "industry_portfolios.csv", parse_dates=["date"]),
        "size_value_assets": _load_csv(root / "assets" / "size_value_portfolios.csv", parse_dates=["date"]),
        "ffc_segmented": _load_csv(root / "factors" / "ffc_segmented_returns.csv", parse_dates=["date"]),
        "industry_factors": _load_csv(root / "factors" / "industry_factor_returns.csv", parse_dates=["date"]),
        "ffc_external_daily": _load_csv(root / "factors" / "ffc_external_daily.csv", parse_dates=["date"]),
    }
    for key in ("industry_assets", "size_value_assets"):
        df = data[key]
        if "weighting" in df.columns:
            data[key] = df.loc[df["weighting"].eq(weighting)].copy()
    return data


def _load_cached_figure1_table_i_rows(cfg: RunConfig) -> pd.DataFrame:
    """Reuse the already-computed yearwise balanced Table I block when available.

    The historical Table I `Balanced panel` block is exactly the Figure 1 sample
    semantics used for submission figures: cross-year changing universe, but each
    yearly panel is internally balanced.
    """
    path = _table_i_output_path(cfg)
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if raw.empty or "panel_block" not in raw.columns:
        return pd.DataFrame()

    if raw["panel_block"].eq("Figure 1 panel: yearwise balanced changing universe").any():
        subset = raw.loc[raw["panel_block"].eq("Figure 1 panel: yearwise balanced changing universe")].copy()
    else:
        subset = raw.loc[raw["panel_block"].eq("Balanced panel")].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["source_panel_block"] = subset.get("source_panel_block", subset["panel_block"])
    subset["panel_block"] = "Figure 1 panel: yearwise balanced changing universe"
    subset["figure_alignment"] = "Figure 1"
    subset["sample_semantics"] = "cross-year unbalanced / within-year balanced; N may vary by year"
    return subset


def _compute_figure1_table_i_rows(cfg: RunConfig, thresholds: Sequence[float]) -> pd.DataFrame:
    log_warn(
        "submission_fast",
        "Cached yearwise Table I rows were not found; recomputing Figure 1 Table I rows. "
        "This can be slower for large yearly universes.",
    )
    probe_panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        years=cfg.years,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    years = sorted({date.year for date in probe_panel.dates})
    rows: List[Dict[str, Any]] = []
    for year in years:
        year_panel = load_proc_hf_panel(
            proc_root=cfg.proc_root,
            sample_mode=STRICT_BALANCED_SAMPLE,
            years=[int(year)],
            return_mode=cfg.return_mode,
            max_stocks=cfg.max_stocks,
        )
        rows.extend(
            _table_i_rows_for_panel(
                year_panel,
                panel_block="Figure 1 panel: yearwise balanced changing universe",
                thresholds=thresholds,
            )
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["source_panel_block"] = "computed_yearwise_balanced"
        df["figure_alignment"] = "Figure 1"
        df["sample_semantics"] = "cross-year unbalanced / within-year balanced; N may vary by year"
    return df


def _compute_figure2_table_i_rows(cfg: RunConfig, thresholds: Sequence[float]) -> pd.DataFrame:
    full_panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        years=None,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    years = sorted({date.year for date in full_panel.dates})
    rows: List[Dict[str, Any]] = []
    for year in years:
        year_panel = subset_panel_by_years(full_panel, [int(year)])
        rows.extend(
            _table_i_rows_for_panel(
                year_panel,
                panel_block="Figure 2 panel: fixed-intersection yearly slices",
                thresholds=thresholds,
            )
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["source_panel_block"] = "strict_balanced_full_year_slice"
        df["figure_alignment"] = "Figure 2"
        df["sample_semantics"] = "fixed full-sample stock intersection; N is constant across years"
    return df


def build_submission_table_i_aligned(
    cfg: RunConfig,
    *,
    thresholds: Sequence[float] = (3.0, 4.0, 4.5, 5.0),
) -> pd.DataFrame:
    """Build Table I using exactly the Figure 1 and Figure 2 panel semantics."""
    fig1 = _compute_figure1_table_i_rows(cfg, thresholds)
    fig2 = _compute_figure2_table_i_rows(cfg, thresholds)
    table_i = pd.concat([fig1, fig2], ignore_index=True)
    if table_i.empty:
        return table_i

    panel_order = {
        "Figure 1 panel: yearwise balanced changing universe": 0,
        "Figure 2 panel: fixed-intersection yearly slices": 1,
    }
    table_i["_panel_order"] = table_i["panel_block"].map(panel_order).fillna(99)
    table_i = table_i.sort_values(["_panel_order", "year", "threshold_a"]).drop(columns=["_panel_order"]).reset_index(drop=True)
    return table_i


def export_table_i_internal(cfg: RunConfig) -> Path:
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)
    table_i = build_submission_table_i_aligned(cfg)
    output_path = _table_i_output_path(cfg)
    diagnostics_path = diagnostics_dir / "table_i_aligned_with_figures.csv"
    _write_csv(diagnostics_path, table_i)
    try:
        _write_csv(output_path, table_i)
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}_aligned.csv")
        _write_csv(fallback_path, table_i)
        log_warn(
            "submission_fast",
            f"Canonical Table I appears to be locked; wrote aligned Table I to {fallback_path} instead.",
        )
        output_path = fallback_path
    summary = {
        "table": "Table I",
        "mode": "submission_fast_aligned",
        "output_path": str(output_path),
        "diagnostics_path": str(diagnostics_path),
        "panels": sorted(table_i["panel_block"].dropna().unique().tolist()) if "panel_block" in table_i else [],
        "rows": int(table_i.shape[0]),
    }
    _write_json(diagnostics_dir / "table_i_aligned_summary.json", summary)
    log_info("submission_fast", f"Generated aligned Table I with {table_i.shape[0]} rows.")
    return output_path


def _year_columns(years: Sequence[int]) -> List[str]:
    return [str(int(year)) for year in years]


def _figure_factor_count_path(cfg: RunConfig, figure_number: int) -> Path:
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)
    if int(figure_number) == 1:
        return diagnostics_dir / "figure1_yearwise_balanced_changing_universe_diagnostics.csv"
    if int(figure_number) == 2:
        return diagnostics_dir / "figure2_fixed_intersection_yearly_diagnostics.csv"
    raise ValueError(f"Unsupported figure number for factor-count diagnostics: {figure_number}")


def _load_hf_factor_count_lookup(cfg: RunConfig, figure_number: int) -> Dict[int, Dict[str, float]]:
    path = _figure_factor_count_path(cfg, figure_number)
    if not path.exists():
        log_warn("submission_fast", f"Factor-count diagnostics not found for Figure {figure_number}: {path}")
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    hf = df.loc[df["return_component"].eq("hf")].copy()
    lookup: Dict[int, Dict[str, float]] = {}
    for _, row in hf.iterrows():
        lookup[int(row["year"])] = {
            "K_hat": float(row.get("K_hat", np.nan)),
            "n_symbols": float(row.get("n_symbols", np.nan)),
        }
    return lookup


def _gc_row(block: str, metric: str, years: Sequence[int], values: Mapping[int, float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"block": block, "metric": metric}
    for year in years:
        value = values.get(int(year), np.nan)
        row[str(int(year))] = float(value) if pd.notna(value) else np.nan
    return row


def _safe_generalized_correlations(F_ref: np.ndarray, F_alt: np.ndarray, K: int) -> np.ndarray:
    K_eff = min(int(K), F_ref.shape[1], F_alt.shape[1])
    if K_eff <= 0:
        return np.full(int(K), np.nan)
    gc = generalized_correlations(F_ref[:, :K_eff], F_alt[:, :K_eff])
    out = np.full(int(K), np.nan, dtype=float)
    out[: len(gc)] = gc[: int(K)]
    return out


def build_submission_table_ii_paper_style(
    cfg: RunConfig,
    *,
    jump_a: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixed-K Table II aligned with submission Figure 1/2 semantics.

    The reference side is Figure 2: fixed full-sample intersection yearly slices.
    The comparison side is Figure 1: yearwise balanced changing-universe panels.
    """
    full_panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        years=None,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    years = sorted({date.year for date in full_panel.dates})
    fig1_lookup = _load_hf_factor_count_lookup(cfg, 1)
    fig2_lookup = _load_hf_factor_count_lookup(cfg, 2)

    gc_values: Dict[Tuple[str, int], Dict[int, float]] = {}
    diagnostics: List[Dict[str, Any]] = []
    n_fig1: Dict[int, float] = {}
    n_fig2: Dict[int, float] = {}
    k_hat_fig1: Dict[int, float] = {}

    for year in years:
        fixed_year = subset_panel_by_years(full_panel, [int(year)])
        changing_year = load_proc_hf_panel(
            proc_root=cfg.proc_root,
            sample_mode=STRICT_BALANCED_SAMPLE,
            years=[int(year)],
            return_mode=cfg.return_mode,
            max_stocks=cfg.max_stocks,
        )
        if fixed_year.R_5min_full.shape[0] != changing_year.R_5min_full.shape[0]:
            raise ValueError(
                f"Year {year}: fixed-intersection and changing-universe panels have different row counts "
                f"({fixed_year.R_5min_full.shape[0]} vs {changing_year.R_5min_full.shape[0]})."
            )

        n_fig1[int(year)] = float(changing_year.N)
        n_fig2[int(year)] = float(fixed_year.N)
        if int(year) in fig1_lookup:
            k_hat_fig1[int(year)] = fig1_lookup[int(year)]["K_hat"]

        fixed_cont, fixed_jump = detect_jumps(fixed_year, a=float(jump_a))
        changing_cont, changing_jump = detect_jumps(changing_year, a=float(jump_a))
        component_payload = {
            "continuous": (fixed_cont, changing_cont),
            "jump": (fixed_jump, changing_jump),
        }
        for component, (fixed_matrix, changing_matrix) in component_payload.items():
            fixed_pca = _panel_pca(fixed_matrix, K=3, use_corr=True)
            changing_pca = _panel_pca(changing_matrix, K=3, use_corr=True)
            for K in (2, 3):
                block = f"First {K} {component} PCA factors"
                gc = _safe_generalized_correlations(fixed_pca.F, changing_pca.F, K)
                for idx, value in enumerate(gc, start=1):
                    gc_values.setdefault((block, idx), {})[int(year)] = float(value)
                    diagnostics.append(
                        {
                            "year": int(year),
                            "component": component,
                            "K_fixed": int(K),
                            "gc_index": int(idx),
                            "gc": float(value),
                            "fixed_intersection_n_symbols": int(fixed_year.N),
                            "changing_universe_n_symbols": int(changing_year.N),
                            "jump_a": float(jump_a),
                        }
                    )

    for year in years:
        if int(year) not in k_hat_fig1:
            k_hat_fig1[int(year)] = np.nan
        if int(year) in fig2_lookup:
            n_fig2[int(year)] = fig2_lookup[int(year)]["n_symbols"]

    rows: List[Dict[str, Any]] = [
        _gc_row("Panel metadata", "K_hat (Figure 1 HF)", years, k_hat_fig1),
        _gc_row("Panel metadata", "N (Figure 1 panel)", years, n_fig1),
        _gc_row("Panel metadata", "N (Figure 2 panel)", years, n_fig2),
    ]
    for block, count in [
        ("First 2 continuous PCA factors", 2),
        ("First 3 continuous PCA factors", 3),
        ("First 2 jump PCA factors", 2),
        ("First 3 jump PCA factors", 3),
    ]:
        for idx in range(1, count + 1):
            rows.append(_gc_row(block, f"{idx}. GC", years, gc_values.get((block, idx), {})))

    wide = pd.DataFrame(rows, columns=["block", "metric", *_year_columns(years)])
    long = pd.DataFrame(diagnostics).sort_values(["year", "component", "K_fixed", "gc_index"]).reset_index(drop=True)
    return wide, long


def export_table_ii_internal(cfg: RunConfig) -> Path:
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)
    wide, long = build_submission_table_ii_paper_style(cfg, jump_a=float(cfg.jump_a))
    output_path = _table_ii_output_path(cfg)
    diagnostics_wide_path = diagnostics_dir / "table_ii_paper_style_fixed_k.csv"
    diagnostics_long_path = diagnostics_dir / "table_ii_paper_style_fixed_k_long.csv"
    _write_csv(diagnostics_wide_path, wide)
    _write_csv(diagnostics_long_path, long)
    try:
        _write_csv(output_path, wide)
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}_paper_style_fixed_k.csv")
        _write_csv(fallback_path, wide)
        log_warn(
            "submission_fast",
            f"Canonical Table II appears to be locked; wrote paper-style Table II to {fallback_path} instead.",
        )
        output_path = fallback_path
    summary = {
        "table": "Table II",
        "mode": "submission_fast_paper_style_fixed_k",
        "output_path": str(output_path),
        "diagnostics_wide_path": str(diagnostics_wide_path),
        "diagnostics_long_path": str(diagnostics_long_path),
        "rows": int(wide.shape[0]),
        "long_rows": int(long.shape[0]),
        "blocks": wide["block"].dropna().unique().tolist() if "block" in wide else [],
    }
    _write_json(diagnostics_dir / "table_ii_paper_style_fixed_k_summary.json", summary)
    log_info("submission_fast", f"Generated paper-style fixed-K Table II with {wide.shape[0]} rows.")
    return output_path


def _submission_gc_row(
    comparison: str,
    panel: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, Any]:
    lhs, rhs = _drop_invalid_rows(np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float))
    gc = generalized_correlations(lhs, rhs) if lhs.size and rhs.size else np.array([], dtype=float)
    row: Dict[str, Any] = {
        "comparison": comparison,
        "panel": panel,
        "n_factors": int(rhs.shape[1]) if rhs.ndim == 2 else 1,
        "sample_days": int(lhs.shape[0]) if lhs.ndim == 2 else 0,
        "gc_mean": float(np.nanmean(gc)) if len(gc) else np.nan,
    }
    for idx in range(1, 5):
        row[f"gc_{idx}"] = float(gc[idx - 1]) if idx <= len(gc) else np.nan
    return row


def build_submission_table_iii(cfg: RunConfig) -> pd.DataFrame:
    figure13_data = _load_submission_figure13_data(cfg)
    sample_dates = pd.DatetimeIndex(sorted(pd.to_datetime(figure13_data["date"].dropna().unique())))
    pca_factor_names = [f"Factor {idx}" for idx in range(1, 5)]
    continuous = _standardized_return_mats_from_figure13(
        figure13_data,
        factor_set="Continuous PCA",
        factor_names=pca_factor_names,
        sample_dates=sample_dates,
    )
    baseline = continuous["daily"]

    legacy = load_legacy_paper_tail_assets(cfg.paper_tail_root, weighting=cfg.paper_tail_weighting)
    industry_daily = legacy["industry_factors"].loc[
        legacy["industry_factors"]["segment_kind"].eq("daily"),
        ["date", "factor", "ret"],
    ].copy()
    selected_industries = list(cfg.industry_factors_frozen or ())
    industry_label_map = {
        "大金融": "Finance industry factor",
        "医药生物": "Healthcare industry factor",
        "周期资源": "Cyclical resources factor",
    }

    rows: List[Dict[str, Any]] = []
    for factor_name, label in [("Market", "Market factor"), *[(name, industry_label_map.get(name, f"{name} factor")) for name in selected_industries]]:
        if factor_name not in set(industry_daily["factor"].dropna().astype(str)):
            continue
        matrix = _align_matrix(industry_daily, dates=sample_dates, factor_names=[factor_name])
        rows.append(_submission_gc_row(label, "Panel A: economic interpretation", baseline, matrix))

    selected_factor_order = ["Market", *[name for name in selected_industries if name in set(industry_daily["factor"].dropna().astype(str))]]
    if selected_factor_order:
        matrix = _align_matrix(industry_daily, dates=sample_dates, factor_names=selected_factor_order)
        rows.append(_submission_gc_row("Selected industry factors", "Panel A: economic interpretation", baseline, matrix))

    ffc = _long_to_factor_mats(sample_dates, legacy["ffc_segmented"])
    rows.append(_submission_gc_row("FFC 4-factor", "Panel B: traditional factor comparison", baseline, ffc["daily"]))

    continuous_unbalanced = _standardized_return_mats_from_figure13(
        figure13_data,
        factor_set="Continuous PCA (unbalanced, yearly aligned)",
        factor_names=pca_factor_names,
        sample_dates=sample_dates,
    )
    rows.append(
        _submission_gc_row(
            "Continuous PCA (unbalanced, yearly aligned)",
            "Panel C: sample-construction robustness",
            baseline,
            continuous_unbalanced["daily"],
        )
    )
    return pd.DataFrame(rows)


def export_table_iii_internal(cfg: RunConfig) -> Path:
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)
    table_iii = build_submission_table_iii(cfg)
    output_path = _table_iii_output_path(cfg)
    diagnostics_path = diagnostics_dir / "table_iii_submission_gc.csv"
    _write_csv(diagnostics_path, table_iii)
    try:
        _write_csv(output_path, table_iii)
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}_submission.csv")
        _write_csv(fallback_path, table_iii)
        log_warn(
            "submission_fast",
            f"Canonical Table III appears to be locked; wrote submission Table III to {fallback_path} instead.",
        )
        output_path = fallback_path
    summary = {
        "table": "Table III",
        "mode": "submission_fast_gc",
        "output_path": str(output_path),
        "diagnostics_path": str(diagnostics_path),
        "rows": int(table_iii.shape[0]),
        "comparisons": table_iii["comparison"].tolist() if "comparison" in table_iii else [],
    }
    _write_json(diagnostics_dir / "table_iii_submission_summary.json", summary)
    log_info("submission_fast", f"Generated submission Table III with {table_iii.shape[0]} rows.")
    return output_path


def _load_submission_figure13_data(cfg: RunConfig) -> pd.DataFrame:
    diagnostics_path = _submission_fast_diagnostics_dir(cfg) / "figure13_data.csv"
    if diagnostics_path.exists():
        return _load_csv(diagnostics_path, parse_dates=["date"])
    return _load_csv(Path(cfg.paper_tail_root) / "figures" / "figure13_data.csv", parse_dates=["date"])


def _standardized_return_mats_from_figure13(
    figure13_data: pd.DataFrame,
    *,
    factor_set: str,
    factor_names: Sequence[str],
    sample_dates: pd.DatetimeIndex,
) -> Dict[str, np.ndarray]:
    mats: Dict[str, np.ndarray] = {}
    subset = figure13_data.loc[figure13_data["factor_set"].eq(factor_set)].copy()
    if subset.empty:
        raise ValueError(f"Figure 13 data does not contain factor_set={factor_set!r}.")
    for segment in SEGMENT_ORDER:
        seg = subset.loc[subset["segment_kind"].eq(segment), ["date", "factor", "normalized_cumulative_return"]].copy()
        pivot = (
            seg.pivot_table(index="date", columns="factor", values="normalized_cumulative_return", aggfunc="last")
            .reindex(sample_dates)
            .reindex(columns=list(factor_names))
        )
        returns = pivot.diff()
        for factor in factor_names:
            first_valid = pivot[factor].first_valid_index()
            if first_valid is not None:
                returns.loc[first_valid, factor] = pivot.loc[first_valid, factor]
        mats[segment] = returns.to_numpy(dtype=float)
    return mats


def _single_factor_sr(series: np.ndarray) -> float:
    s = np.asarray(series, dtype=float)
    sd = np.nanstd(s, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float(np.nanmean(s) / sd * np.sqrt(float(_annualization_days())))


def _sharpe_row(section: str, portfolio: str, sharpes: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "section": section,
        "portfolio": portfolio,
        "SR_intraday": float(sharpes.get("SR_intraday", np.nan)) if pd.notna(sharpes.get("SR_intraday", np.nan)) else np.nan,
        "SR_overnight": float(sharpes.get("SR_overnight", np.nan)) if pd.notna(sharpes.get("SR_overnight", np.nan)) else np.nan,
        "SR_daily": float(sharpes.get("SR_daily", np.nan)) if pd.notna(sharpes.get("SR_daily", np.nan)) else np.nan,
    }


def _single_segment_tangency_sr(matrix: np.ndarray) -> float:
    import core.engine as eng

    x = np.asarray(matrix, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mask = np.isfinite(x).all(axis=1)
    if not mask.any():
        return np.nan
    _, sr = eng.tangency_portfolio(x[mask], np.zeros(int(mask.sum()), dtype=float))
    return float(sr * np.sqrt(float(_annualization_days()))) if np.isfinite(sr) else np.nan


def _component_weighted_sharpes(
    matrix_by_segment: Mapping[str, np.ndarray],
    *,
    weight_segment: str,
) -> Dict[str, float]:
    """Estimate tangency weights on one segment and evaluate them on all segments."""
    import core.engine as eng

    base = np.asarray(matrix_by_segment[weight_segment], dtype=float)
    if base.ndim == 1:
        base = base[:, None]
    mask = np.isfinite(base).all(axis=1)
    if not mask.any():
        return {"SR_intraday": np.nan, "SR_overnight": np.nan, "SR_daily": np.nan}
    weights, _ = eng.tangency_portfolio(base[mask], np.zeros(int(mask.sum()), dtype=float))
    weights = np.asarray(weights, dtype=float).reshape(-1)

    sharpes: Dict[str, float] = {}
    for segment in SEGMENT_ORDER:
        matrix = np.asarray(matrix_by_segment[segment], dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[1] != weights.shape[0]:
            sharpes[f"SR_{segment}"] = np.nan
            continue
        portfolio_return = matrix @ weights
        sharpes[f"SR_{segment}"] = _single_factor_sr(portfolio_return)
    return sharpes


def build_submission_table_v(
    cfg: RunConfig,
) -> pd.DataFrame:
    figure13_data = _load_submission_figure13_data(cfg)
    sample_dates = pd.DatetimeIndex(sorted(pd.to_datetime(figure13_data["date"].dropna().unique())))
    pca_factor_names = [f"Factor {idx}" for idx in range(1, 5)]

    continuous = _standardized_return_mats_from_figure13(
        figure13_data,
        factor_set="Continuous PCA",
        factor_names=pca_factor_names,
        sample_dates=sample_dates,
    )
    continuous_unbalanced = _standardized_return_mats_from_figure13(
        figure13_data,
        factor_set="Continuous PCA (unbalanced, yearly aligned)",
        factor_names=pca_factor_names,
        sample_dates=sample_dates,
    )

    legacy = load_legacy_paper_tail_assets(cfg.paper_tail_root, weighting=cfg.paper_tail_weighting)
    rf_daily = _load_rf_series(_rf_file(Path(cfg.external_data_root)))
    rf_daily_sample = rf_daily.reindex(sample_dates, fill_value=0.0).to_numpy(dtype=float)

    industry_names = sorted(
        legacy["industry_factors"]["factor"].dropna().unique().tolist(),
        key=lambda name: (0 if name == "Market" else 1, str(name)),
    )
    industry_raw = {
        segment: _align_matrix(
            legacy["industry_factors"].loc[
                legacy["industry_factors"]["segment_kind"].eq(segment),
                ["date", "factor", "ret"],
            ],
            dates=sample_dates,
            factor_names=industry_names,
        )
        for segment in SEGMENT_ORDER
    }
    industry = _to_excess_matrices(industry_raw, rf_daily_sample=rf_daily_sample)
    ffc = _long_to_factor_mats(sample_dates, legacy["ffc_segmented"])

    rows: List[Dict[str, Any]] = []
    for label, mats in [
        ("Continuous PCA", continuous),
        ("Continuous PCA (unbalanced, yearly aligned)", continuous_unbalanced),
        ("Industry factors", industry),
        ("FFC 4-factor", ffc),
    ]:
        rows.append(_sharpe_row("factor_set_tangency", label, _portfolio_sharpes_from_excess_matrices(mats)))

    rows.append(
        _sharpe_row(
            "factor_set_tangency",
            "PCA overnight",
            _component_weighted_sharpes(continuous, weight_segment="overnight"),
        )
    )
    rows.append(
        _sharpe_row(
            "factor_set_tangency",
            "PCA daily",
            _component_weighted_sharpes(continuous, weight_segment="daily"),
        )
    )

    for idx, factor in enumerate(pca_factor_names, start=1):
        rows.append(
            {
                "section": "continuous_individual_factors",
                "portfolio": f"Continuous PCA Factor {idx}",
                "SR_intraday": _single_factor_sr(continuous["intraday"][:, idx - 1]),
                "SR_overnight": _single_factor_sr(continuous["overnight"][:, idx - 1]),
                "SR_daily": _single_factor_sr(continuous["daily"][:, idx - 1]),
            }
        )

    for col, label in enumerate(["Market", "Size", "Value", "Momentum"]):
        rows.append(
            {
                "section": "characteristic_individual_factors",
                "portfolio": label,
                "SR_intraday": _single_factor_sr(ffc["intraday"][:, col]),
                "SR_overnight": _single_factor_sr(ffc["overnight"][:, col]),
                "SR_daily": _single_factor_sr(ffc["daily"][:, col]),
            }
        )

    for idx, factor in enumerate(pca_factor_names, start=1):
        rows.append(
            {
                "section": "unbalanced_aligned_individual_factors",
                "portfolio": f"Continuous PCA Factor {idx} (unbalanced, yearly aligned)",
                "SR_intraday": _single_factor_sr(continuous_unbalanced["intraday"][:, idx - 1]),
                "SR_overnight": _single_factor_sr(continuous_unbalanced["overnight"][:, idx - 1]),
                "SR_daily": _single_factor_sr(continuous_unbalanced["daily"][:, idx - 1]),
            }
        )

    table = pd.DataFrame(rows)
    return table


def export_table_v_internal(cfg: RunConfig) -> Path:
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)
    table_v = build_submission_table_v(cfg)
    output_path = _table_v_output_path(cfg)
    diagnostics_path = diagnostics_dir / "table_v_submission_sharpe_ratios.csv"
    _write_csv(diagnostics_path, table_v)
    try:
        _write_csv(output_path, table_v)
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}_submission.csv")
        _write_csv(fallback_path, table_v)
        log_warn(
            "submission_fast",
            f"Canonical Table V appears to be locked; wrote submission Table V to {fallback_path} instead.",
        )
        output_path = fallback_path
    summary = {
        "table": "Table V",
        "mode": "submission_fast_sharpe_ratios",
        "output_path": str(output_path),
        "diagnostics_path": str(diagnostics_path),
        "rows": int(table_v.shape[0]),
        "portfolios": table_v["portfolio"].tolist() if "portfolio" in table_v else [],
        "note": "PCA rows use standardized return increments recovered from Figure 13 normalized cumulative returns.",
    }
    _write_json(diagnostics_dir / "table_v_submission_summary.json", summary)
    log_info("submission_fast", f"Generated submission Table V with {table_v.shape[0]} rows.")
    return output_path


def build_submission_fast_tail_payload(
    result: ReplicationResult,
    cfg: RunConfig,
    *,
    diagnostics_dir: Path,
    strict_fail: bool = True,
) -> Dict[str, Any]:
    sample_dates = pd.DatetimeIndex(result.panel.dates)
    payload: Dict[str, Any] = load_legacy_paper_tail_assets(cfg.paper_tail_root, weighting=cfg.paper_tail_weighting)

    figure13_alignment: Dict[str, Any] = {
        "factor_set_label": "Continuous PCA (unbalanced, yearly aligned)",
        "status": "not_built",
        "years": [],
    }
    try:
        pca_unbalanced, figure13_alignment = _build_yearly_aligned_unbalanced_pca_segments(
            result,
            proc_root=Path(cfg.proc_root),
            sample_dates=sample_dates,
            k=4,
        )
        figure13_alignment["status"] = "built"
        figure13_data = _build_figure13_data(
            result,
            sample_dates=sample_dates,
            ffc_segmented=payload["ffc_segmented"],
            pca_unbalanced=pca_unbalanced,
        )
    except Exception as exc:
        if strict_fail:
            raise
        log_warn("submission_fast", f"Figure 13 strict yearly alignment failed, reusing legacy Figure 13 data: {exc!r}")
        figure13_alignment = {
            "factor_set_label": "Continuous PCA (unbalanced, yearly aligned)",
            "status": "legacy_fallback",
            "error": repr(exc),
            "years": [],
        }
        figure13_data = payload["figure13_data_legacy"].copy()

    rf_daily = _load_rf_series(_rf_file(Path(cfg.external_data_root)))
    rf_daily_sample = rf_daily.reindex(sample_dates, fill_value=0.0).to_numpy(dtype=float)
    continuous = {
        "intraday": np.asarray(result.pipeline.F_cont_display_daily_intra, dtype=float),
        "overnight": np.asarray(result.pipeline.F_cont_display_daily_night, dtype=float),
        "daily": np.asarray(result.pipeline.F_cont_display_daily_total, dtype=float),
    }
    ffc_mats = _long_to_factor_mats(sample_dates, payload["ffc_segmented"])
    factor_sets = {
        "Continuous PCA": continuous,
        "FFC 4-factor": ffc_mats,
    }
    pricing_industry = _build_pricing_frame(
        asset_df=payload["industry_assets"],
        sample_dates=sample_dates,
        factor_sets=factor_sets,
        rf_daily_sample=rf_daily_sample,
    )
    pricing_size_value = _build_pricing_frame(
        asset_df=payload["size_value_assets"],
        sample_dates=sample_dates,
        factor_sets=factor_sets,
        rf_daily_sample=rf_daily_sample,
    )

    payload["figure13_data"] = figure13_data
    payload["pricing_industry"] = pricing_industry
    payload["pricing_size_value"] = pricing_size_value
    payload["diagnostics"] = {
        "figure13_yearly_alignment": figure13_alignment,
        "source_mode": "legacy_tail_assets_plus_strict_continuous_pca",
        "paper_tail_root": str(Path(cfg.paper_tail_root).resolve()),
        "paper_tail_weighting": str(cfg.paper_tail_weighting),
    }
    payload["manifest"] = {
        "mode": "submission_fast",
        "strict_panel_mode": str(result.panel.sample_mode),
        "paper_tail_weighting": str(cfg.paper_tail_weighting),
    }

    _write_csv(diagnostics_dir / "figure12_data.csv", payload["figure12_data"])
    _write_csv(diagnostics_dir / "figure13_data.csv", figure13_data)
    _write_csv(diagnostics_dir / "pricing_industry.csv", pricing_industry)
    _write_csv(diagnostics_dir / "pricing_size_value.csv", pricing_size_value)
    _write_json(diagnostics_dir / "figure13_yearly_alignment.json", figure13_alignment)
    return payload


def build_submission_fast_result(
    cfg: RunConfig,
    *,
    strict_fail: bool = True,
    build_factor_count_diagnostics: bool = True,
) -> ReplicationResult:
    cfg.export_fidelity_env()
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)

    if str(cfg.balanced_mode) != STRICT_BALANCED_SAMPLE:
        raise ValueError("submission_fast requires strict_balanced as the main panel mode.")

    log_step("submission_fast", "Loading strict full-sample panel.")
    panel = load_proc_hf_panel(
        proc_root=cfg.proc_root,
        sample_mode=STRICT_BALANCED_SAMPLE,
        years=cfg.years,
        return_mode=cfg.return_mode,
        max_stocks=cfg.max_stocks,
    )
    universe_summary = _read_json(Path(cfg.proc_root) / "metadata" / "universe_summary.json")

    t0 = time.perf_counter()
    log_step("submission_fast", "Running minimal strict continuous-PCA pipeline.")
    pipeline = PelgerPipeline(
        panel=panel,
        jump_a=cfg.jump_a,
        K_max=cfg.k_max,
        gamma=cfg.gamma,
        g_fn=cfg.g_fn,
    ).run_full()
    pipeline_sec = time.perf_counter() - t0

    rolling_gc, rolling_ev, rolling_source = _load_or_build_rolling_outputs(
        cfg,
        diagnostics_dir=diagnostics_dir,
        pipeline=pipeline,
    )
    result = ReplicationResult(
        universe=pd.DataFrame(),
        universe_summary=universe_summary,
        panel=panel,
        pipeline=pipeline,
        rolling_gc=rolling_gc,
        rolling_explained_variation=rolling_ev,
        robustness=pd.DataFrame(),
        output_root=Path(cfg.final_result_root),
        runtime_root=submission_fast_runtime_root(cfg),
        stage_timings={"pipeline_core_sec": float(pipeline_sec)},
        resource_plan={"submission_fast": True, "rolling_source": rolling_source},
    )
    result = refresh_replication_result_views(
        result,
        proc_root=cfg.proc_root,
        external_data_root=cfg.external_data_root,
        paper_tail_root=cfg.paper_tail_root,
        paper_tail_weighting=cfg.paper_tail_weighting,
        refresh_paper_tail=bool(cfg.refresh_paper_tail),
        strict_final_export=cfg.strict_final_export,
    )

    if build_factor_count_diagnostics:
        fig1_df = build_submission_figure1_factor_counts(result)
        fig2_df = build_submission_figure2_factor_counts(result)
        _write_csv(diagnostics_dir / "figure1_yearwise_balanced_changing_universe_diagnostics.csv", fig1_df)
        _write_csv(diagnostics_dir / "figure2_fixed_intersection_yearly_diagnostics.csv", fig2_df)
    else:
        log_info("submission_fast", "Skipping Figure 1/2 factor-count diagnostics for RF-only tail refresh.")

    paper_tail_payload = build_submission_fast_tail_payload(
        result,
        cfg,
        diagnostics_dir=diagnostics_dir,
        strict_fail=strict_fail,
    )
    result.paper_tail = paper_tail_payload

    manifest = {
        "mode": "submission_fast",
        "proc_root": str(Path(cfg.proc_root).resolve()),
        "runtime_root": str(Path(cfg.runtime_root).resolve()),
        "submission_runtime_root": str(submission_fast_runtime_root(cfg)),
        "final_result_root": str(Path(cfg.final_result_root).resolve()),
        "balanced_mode": str(cfg.balanced_mode),
        "paper_tail_weighting": str(cfg.paper_tail_weighting),
        "rolling_source": rolling_source,
        "strict_panel_symbols": int(result.panel.N),
        "sample_days": int(result.panel.D),
        "pipeline_core_sec": float(pipeline_sec),
    }
    _write_json(diagnostics_dir / "submission_fast_manifest.json", manifest)
    return result


def export_core_internal(
    cfg: RunConfig,
    *,
    strict_fail: bool = True,
) -> Dict[str, str]:
    result = build_submission_fast_result(cfg, strict_fail=strict_fail)
    outputs: Dict[str, str] = {}
    failures: List[Tuple[str, str]] = []

    try:
        from tableCode.table_i import export_fast as export_table_i

        table_i_path = export_table_i(cfg)
        outputs["table_i"] = str(table_i_path)
    except Exception as exc:
        failures.append(("table_i", repr(exc)))
        if strict_fail:
            raise
        log_warn("submission_fast", f"table_i failed: {exc!r}")

    for task_key, module_name in SUBMISSION_FAST_FIGURES:
        module = __import__(module_name, fromlist=["generate"])
        generate = getattr(module, "generate")
        try:
            output_path = run_generator(task_key, generate, result=result, cfg=cfg)
            outputs[task_key] = str(output_path)
        except Exception as exc:
            failures.append((task_key, repr(exc)))
            if strict_fail:
                raise
            log_warn("submission_fast", f"{task_key} failed: {exc!r}")

    summary = {
        "generated": outputs,
        "failures": [{"task": key, "error": err} for key, err in failures],
    }
    _write_json(_submission_fast_diagnostics_dir(cfg) / "export_summary.json", summary)
    if failures and strict_fail:
        raise RuntimeError(f"submission_fast export failed: {failures!r}")
    log_info("submission_fast", f"Generated {len(outputs)} submission-core figures via the lightweight path.")
    return outputs


__all__ = [
    "SUBMISSION_FAST_FIGURES",
    "build_submission_fast_result",
    "build_submission_fast_tail_payload",
    "build_submission_table_i_aligned",
    "build_submission_table_ii_paper_style",
    "build_submission_table_iii",
    "build_submission_table_v",
    "load_legacy_paper_tail_assets",
    "submission_fast_runtime_root",
]
