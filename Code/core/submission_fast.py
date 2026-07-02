from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from core.config import RunConfig
from core.engine import (
    STRICT_BALANCED_SAMPLE,
    PelgerPipeline,
    ReplicationResult,
    _atomic_to_csv,
    _rolling_output_frames,
    build_submission_figure1_factor_counts,
    build_submission_figure2_factor_counts,
    build_weight_tables,
    load_proc_hf_panel,
    refresh_replication_result_views,
    rolling_gc_and_explained_variation,
)
from core.logging_utils import log_info, log_step, log_warn
from core.paper_tail import (
    OFFICIAL_FFC_FACTORS,
    SEGMENT_ORDER,
    _align_matrix,
    _build_figure13_data,
    _build_pricing_frame,
    _build_yearly_aligned_unbalanced_pca_segments,
    _load_rf_series,
)
from core.runner import run_generator


SUBMISSION_FAST_FIGURES: Tuple[Tuple[str, str], ...] = (
    ("fig1", "figcode.figure_01"),
    ("fig2", "figcode.figure_02"),
    ("fig4", "figcode.figure_04"),
    ("fig7", "figcode.figure_07"),
    ("fig10", "figcode.figure_10"),
    ("fig12", "figcode.figure_12"),
    ("fig13", "figcode.figure_13"),
    ("fig14", "figcode.figure_14"),
    ("fig15", "figcode.figure_15"),
)


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
    if not path.exists():
        raise FileNotFoundError(f"Required submission asset is missing: {path}")
    return pd.read_csv(path, parse_dates=list(parse_dates or ()))


def _rf_file(external_root: Path) -> Path:
    candidates = sorted((external_root / "factors" / "rf").glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No RF csv files found under {external_root / 'factors' / 'rf'}")
    return candidates[0]


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
    rolling_gc_csv = diagnostics_dir / "rolling_gc.csv"
    rolling_ev_csv = diagnostics_dir / "rolling_explained_variation.csv"
    if rolling_gc_csv.exists() and rolling_ev_csv.exists():
        try:
            rolling_gc_df = pd.read_csv(rolling_gc_csv)
            rolling_ev_df = pd.read_csv(rolling_ev_csv)
            gc_cols = [col for col in rolling_gc_df.columns if col.startswith("gc_")]
            if gc_cols and "explained_variation" in rolling_ev_df.columns:
                return (
                    rolling_gc_df[gc_cols].to_numpy(dtype=float),
                    rolling_ev_df["explained_variation"].to_numpy(dtype=float),
                    "submission_fast_csv",
                )
        except Exception as exc:
            log_warn("submission_fast", f"Existing rolling CSV diagnostics could not be reused: {exc!r}")

    rolling_dir = Path(cfg.runtime_root) / "checkpoints" / "rolling"
    if _rolling_signature_matches(cfg) and rolling_dir.exists():
        window_records: List[Tuple[int, np.ndarray, float]] = []
        for path in sorted(rolling_dir.glob("chunk_*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                window_index = np.asarray(payload.get("window_index", np.zeros(0, dtype=np.int32)), dtype=np.int32)
                gc = np.asarray(payload.get("gc", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
                explained = np.asarray(payload.get("explained_variation", np.zeros(0, dtype=np.float64)), dtype=np.float64)
            for idx, win in enumerate(window_index.tolist()):
                window_records.append((int(win), np.asarray(gc[idx], dtype=np.float64), float(explained[idx])))
        window_records.sort(key=lambda item: int(item[0]))
        if window_records:
            rolling_gc = np.vstack([item[1] for item in window_records])
            rolling_ev = np.asarray([item[2] for item in window_records], dtype=np.float64)
            _persist_rolling_outputs(diagnostics_dir, rolling_gc, rolling_ev)
            return rolling_gc, rolling_ev, "strict_checkpoint"

    log_warn("submission_fast", "No compatible strict rolling checkpoint found; recomputing strict rolling diagnostics.")
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
    return rolling_gc, rolling_ev, "strict_recomputed"


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


def build_submission_fast_tail_payload(
    result: ReplicationResult,
    cfg: RunConfig,
    *,
    diagnostics_dir: Path,
    strict_fail: bool = True,
) -> Dict[str, Any]:
    sample_dates = pd.DatetimeIndex(result.panel.dates)
    payload: Dict[str, Any] = load_legacy_paper_tail_assets(cfg.paper_tail_root, weighting=cfg.paper_tail_weighting)

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
        log_warn("submission_fast", f"Figure 13 yearly alignment failed; reusing legacy Figure 13 data: {exc!r}")
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
) -> ReplicationResult:
    cfg.export_fidelity_env()
    diagnostics_dir = _submission_fast_diagnostics_dir(cfg)

    if str(cfg.balanced_mode) != STRICT_BALANCED_SAMPLE:
        raise ValueError("submission_fast requires strict_balanced as the main panel mode.")

    log_step("submission_fast", "Loading strict full-sample panel from proc_Data.")
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
        refresh_paper_tail=False,
        strict_final_export=cfg.strict_final_export,
    )
    result.pca_weights, result.proxy_weights = build_weight_tables(pipeline, panel)

    fig1_df = build_submission_figure1_factor_counts(result)
    fig2_df = build_submission_figure2_factor_counts(result)
    _write_csv(diagnostics_dir / "figure1_yearwise_balanced_changing_universe_diagnostics.csv", fig1_df)
    _write_csv(diagnostics_dir / "figure2_fixed_intersection_yearly_diagnostics.csv", fig2_df)

    result.paper_tail = build_submission_fast_tail_payload(
        result,
        cfg,
        diagnostics_dir=diagnostics_dir,
        strict_fail=strict_fail,
    )

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


def export_submission_core_fast(
    cfg: RunConfig,
    *,
    strict_fail: bool = True,
) -> Dict[str, str]:
    result = build_submission_fast_result(cfg, strict_fail=strict_fail)
    outputs: Dict[str, str] = {}
    failures: List[Tuple[str, str]] = []

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
    "export_submission_core_fast",
    "load_legacy_paper_tail_assets",
    "submission_fast_runtime_root",
]
