"""
Pelger (2020) "Understanding Systematic Risk: A High-Frequency Approach"
China A-share 5-minute replication core.

This file does not read raw K-line `data.bz2` files. Raw data cleaning,
backward adjustment, and panel construction are owned by `preprocess_cn_data.py`.
The replication core consumes processed panels from `Data/proc_Data`, runs the
paper's jump decomposition, PCA factor extraction, Sharpe calculations, rolling
stability analysis, and exports results to `Result/`.
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from scipy.linalg import eigh


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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROC_ROOT = REPO_ROOT / "Data" / "proc_Data" / "pelger_cn_adjusted"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Result" / "pelger_cn_adjusted"


def _default_worker_count() -> int:
    """复现阶段默认保守并行度，避免和 NumPy/BLAS 内部线程过度竞争。"""
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def _normalize_worker_count(workers: Optional[int]) -> int:
    if workers is None:
        return _default_worker_count()
    return max(1, int(workers))


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


def load_proc_universe(proc_root: str | Path = DEFAULT_PROC_ROOT) -> pd.DataFrame:
    """读取预处理阶段生成的样本清单，并恢复 summary 元数据。"""
    proc_root = _ensure_path(proc_root)
    universe_path = proc_root / "metadata" / "universe.pkl"
    summary_path = proc_root / "metadata" / "universe_summary.json"
    if not universe_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"未找到预处理样本文件，请先运行 Code/preprocess_cn_data.py: {proc_root}"
        )

    universe = pd.read_pickle(universe_path)
    summary = _load_json(summary_path)
    if "calendar_days_by_year" in summary:
        summary["calendar_days_by_year"] = {int(k): int(v) for k, v in summary["calendar_days_by_year"].items()}
    if "global_dates_by_year" in summary:
        summary["global_dates_by_year"] = {int(k): list(v) for k, v in summary["global_dates_by_year"].items()}
    universe.attrs["summary"] = summary
    return universe



def summarize_cn_universe(universe: pd.DataFrame) -> Dict[str, Any]:
    """Summarize the already-preprocessed universe table."""
    if "summary" not in universe.attrs:
        raise ValueError("Processed universe is missing attrs['summary']; reload it with load_proc_universe().")
    summary = dict(universe.attrs["summary"])
    stats = dict(summary)
    if "coverage_ratio" in universe.columns and not universe.empty:
        stats["coverage_ratio_quantiles"] = {
            "p0": float(universe["coverage_ratio"].min()),
            "p25": float(universe["coverage_ratio"].quantile(0.25)),
            "p50": float(universe["coverage_ratio"].quantile(0.50)),
            "p75": float(universe["coverage_ratio"].quantile(0.75)),
            "p95": float(universe["coverage_ratio"].quantile(0.95)),
            "p100": float(universe["coverage_ratio"].max()),
        }
    stats["invalid_day_symbols"] = int((universe.get("n_invalid_days", pd.Series(dtype=float)) > 0).sum())
    if "suspicious_overnight_count_012" in universe.columns:
        stats["corp_action_risk_summary"] = {
            "symbols_with_suspicious_overnight_012": int((universe["suspicious_overnight_count_012"] > 0).sum()),
            "symbols_with_suspicious_overnight_020": int((universe["suspicious_overnight_count_020"] > 0).sum()),
            "max_abs_overnight_overall": float(universe["max_abs_overnight"].max()) if "max_abs_overnight" in universe.columns and not universe.empty else 0.0,
        }
    return stats

def _proc_panel_paths(proc_root: Path, sample_mode: str, panel_name: str) -> Tuple[Path, Path]:
    panel_dir = proc_root / "panels" / sample_mode
    return panel_dir / f"{panel_name}.npz", panel_dir / f"{panel_name}.json"


def _proc_panel_name(years: Optional[Sequence[int]]) -> Optional[str]:
    years = _normalize_years(years)
    if years is None:
        return "full"
    if len(years) == 1:
        return f"year_{years[0]}"
    return None


def _subset_panel_columns(panel: HFPanel, max_stocks: Optional[int]) -> HFPanel:
    if max_stocks is None or panel.N <= int(max_stocks):
        return panel

    keep = slice(0, int(max_stocks))
    sample_report = dict(panel.sample_report or {})
    sample_report["n_symbols_selected"] = int(max_stocks)
    sample_report["target_symbols"] = list(panel.tickers[keep])
    return HFPanel(
        R_intra=panel.R_intra[:, keep],
        R_night=panel.R_night[:, keep],
        day_ids=panel.day_ids.copy(),
        tickers=list(panel.tickers[keep]),
        dates=list(panel.dates),
        R_daily=panel.R_daily[:, keep],
        rf_intra=panel.rf_intra.copy(),
        rf_night=panel.rf_night.copy(),
        bar_times=list(panel.bar_times or []),
        sample_report=sample_report,
        sample_mode=panel.sample_mode,
        return_mode=panel.return_mode,
    )


def _load_proc_panel_file(proc_root: Path, sample_mode: str, panel_name: str) -> HFPanel:
    npz_path, meta_path = _proc_panel_paths(proc_root, sample_mode, panel_name)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Processed panel metadata not found: {meta_path}. Run Code/preprocess_cn_data.py first."
        )

    meta = _load_json(meta_path)
    if "array_files" in meta:
        arrays = {name: np.load(proc_root / rel_path, allow_pickle=False) for name, rel_path in meta["array_files"].items()}
    else:
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Processed panel arrays not found: {npz_path}. Run Code/preprocess_cn_data.py first."
            )
        arrays_raw = np.load(npz_path, allow_pickle=False)
        arrays = {name: arrays_raw[name] for name in arrays_raw.files}

    return HFPanel(
        R_intra=arrays["R_intra"],
        R_night=arrays["R_night"],
        day_ids=arrays["day_ids"],
        tickers=list(meta["tickers"]),
        dates=[pd.Timestamp(date) for date in meta["dates"]],
        R_daily=arrays["R_daily"],
        bar_times=list(meta["bar_times"]),
        sample_report=dict(meta.get("sample_report", {})),
        sample_mode=meta["sample_mode"],
        return_mode=meta.get("return_mode", "open_close"),
    )

def load_proc_hf_panel(
    proc_root: str | Path = DEFAULT_PROC_ROOT,
    sample_mode: str = STRICT_BALANCED_SAMPLE,
    years: Optional[Sequence[int]] = None,
    return_mode: str = "open_close",
    max_stocks: Optional[int] = None,
) -> HFPanel:
    """从 Data/proc_Data 中读取预处理好的 HFPanel。"""
    proc_root = _ensure_path(proc_root)
    panel_name = _proc_panel_name(years)
    if panel_name is not None:
        try:
            panel = _load_proc_panel_file(proc_root, sample_mode, panel_name)
        except FileNotFoundError:
            panel = subset_panel_by_years(_load_proc_panel_file(proc_root, sample_mode, "full"), _normalize_years(years) or [])
    else:
        panel = subset_panel_by_years(_load_proc_panel_file(proc_root, sample_mode, "full"), _normalize_years(years) or [])

    panel.return_mode = return_mode
    panel.sample_report = dict(panel.sample_report or {})
    panel.sample_report["proc_root"] = str(proc_root)
    panel.sample_report["years"] = _normalize_years(years)
    return _subset_panel_columns(panel, max_stocks)



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


def _bipower_variation_cube(R_cube: np.ndarray) -> np.ndarray:
    """向量化 BNS bipower variation，shape: (D, M, N) -> (D, N)。"""
    abs_r = np.abs(R_cube)
    prod = abs_r[:, 1:, :] * abs_r[:, :-1, :]
    return (np.pi / 2.0) * np.nansum(prod, axis=1)


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

    R_cube = R.reshape(D, M, N)
    BV = _bipower_variation_cube(R_cube)
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
    g_fn: str = "median_N",
    N: Optional[int] = None,
    gamma: float = 0.08,
    K_max: Optional[int] = None,
) -> Tuple[int, np.ndarray]:
    """Pelger (2019) 扰动特征值比率."""
    lam = np.asarray(eigvals, dtype=float).copy()
    if g_fn == "median_N":
        assert N is not None
        g = N * np.median(lam)
    elif g_fn == "median_sqrtN":
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
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """按滚动交易日窗口做局部 PCA."""
    D = int(day_ids.max()) + 1

    def _one_window(start: int) -> Optional[Dict[str, Any]]:
        end = start + window_days
        mask = (day_ids >= start) & (day_ids < end)
        R_win = R[mask]
        if R_win.shape[0] < 2 * K:
            return None
        res = pca_factors(R_win, K=K, use_corr=use_corr)
        return {
            "start_day": start,
            "end_day": end,
            "Lambda": res.Lambda,
            "F": res.F,
            "eigvals": res.eigvals,
        }

    starts = list(range(0, D - window_days + 1, step_days))
    workers = max(1, int(workers))
    if workers == 1 or len(starts) <= 1:
        results = [_one_window(start) for start in starts]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_one_window, start): start for start in starts}
            for future in as_completed(futures):
                results.append(future.result())
    results = [result for result in results if result is not None]
    results.sort(key=lambda item: int(item["start_day"]))
    return results


def rolling_gc_vs_global(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    global_Lambda: np.ndarray,
    step_days: int = 1,
    workers: int = 1,
) -> np.ndarray:
    """局部载荷与全局载荷的 generalized correlation."""
    loc = rolling_local_pca(R, day_ids, window_days, K=K, step_days=step_days, workers=workers)
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
    workers: int = 1,
) -> np.ndarray:
    """滚动窗口下前 K 个因子解释的变异占比."""
    loc = rolling_local_pca(R, day_ids, window_days, K=K, step_days=step_days, workers=workers)
    ratios = []
    for result in loc:
        eigvals = result["eigvals"]
        total = float(eigvals.sum())
        ratios.append(float(eigvals[:K].sum() / total) if total > 0 else 0.0)
    return np.asarray(ratios, dtype=float)


def rolling_gc_and_explained_variation(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    global_Lambda: np.ndarray,
    step_days: int = 1,
    workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """一次滚动 PCA 同时生成 GC 和解释度，避免重复计算窗口 PCA。"""
    loc = rolling_local_pca(R, day_ids, window_days, K=K, step_days=step_days, workers=workers)
    if not loc:
        return np.zeros((0, K)), np.zeros(0, dtype=float)

    GC = np.zeros((len(loc), K))
    ratios: List[float] = []
    for i, result in enumerate(loc):
        gc = generalized_correlations(global_Lambda, result["Lambda"])
        GC[i, : len(gc)] = gc
        eigvals = result["eigvals"]
        total = float(eigvals.sum())
        ratios.append(float(eigvals[:K].sum() / total) if total > 0 else 0.0)
    return GC, np.asarray(ratios, dtype=float)


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
    g_fn: str = "median_N"

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
        self.K_hf_hat, _ = perturbed_eigenvalue_ratio(r_hf.eigvals, g_fn=self.g_fn, N=N, gamma=self.gamma, K_max=self.K_max)
        self.K_cont_hat, _ = perturbed_eigenvalue_ratio(r_cont.eigvals, g_fn=self.g_fn, N=N, gamma=self.gamma, K_max=self.K_max)
        self.K_jump_hat, _ = perturbed_eigenvalue_ratio(r_jump.eigvals, g_fn=self.g_fn, N=N, gamma=self.gamma, K_max=self.K_max)

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


def _near_balanced_year_task(
    proc_root: str,
    year: int,
    K_compare: int,
    return_mode: str,
    jump_a: float,
    max_near_symbols: Optional[int],
) -> Optional[Dict[str, Any]]:
    """年度稳健性 worker：读取当年 strict/near 面板并返回 GC 行。"""
    try:
        strict_panel = load_proc_hf_panel(
            proc_root=proc_root,
            sample_mode=STRICT_BALANCED_SAMPLE,
            years=[year],
            return_mode=return_mode,
        )
        near_panel = load_proc_hf_panel(
            proc_root=proc_root,
            sample_mode=NEAR_BALANCED_99_SAMPLE,
            years=[year],
            return_mode=return_mode,
            max_stocks=max_near_symbols,
        )
    except FileNotFoundError:
        return None

    R_cont_strict, _ = detect_jumps(strict_panel, a=jump_a)
    R_cont_near, _ = detect_jumps(near_panel, a=jump_a)
    pca_strict = pca_factors(R_cont_strict, K=K_compare, use_corr=True)
    pca_near = pca_factors_pairwise(R_cont_near, K=K_compare, use_corr=True, psd_fix=True)
    gc = generalized_correlations(pca_strict.F, pca_near.F)

    row = {
        "year": int(year),
        "n_strict_symbols": int(strict_panel.N),
        "n_near_symbols": int(near_panel.N),
        "n_days": int(near_panel.D),
    }
    for idx, value in enumerate(gc, start=1):
        row[f"gc_{idx}"] = float(value)
    return row


def run_near_balanced_robustness(
    proc_root: str | Path,
    universe: pd.DataFrame,
    years: Optional[Sequence[int]],
    K_compare: int,
    return_mode: str = "open_close",
    jump_a: float = 3.0,
    max_near_symbols: Optional[int] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Compare yearly 99%-coverage panels with the strict-balanced factor space."""
    proc_root = _ensure_path(proc_root)
    if "summary" not in universe.attrs:
        universe = load_proc_universe(proc_root)
    summary = universe.attrs["summary"]
    years = _normalize_years(years) or sorted(int(year) for year in summary["calendar_days_by_year"])

    rows: List[Dict[str, Any]] = []
    workers = max(1, int(workers))
    if workers == 1 or len(years) <= 1:
        for year in years:
            row = _near_balanced_year_task(str(proc_root), int(year), K_compare, return_mode, jump_a, max_near_symbols)
            if row is not None:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_near_balanced_year_task, str(proc_root), int(year), K_compare, return_mode, jump_a, max_near_symbols): int(year)
                for year in years
            }
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
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
    paper_jump_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_factor_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_factor_sharpes: pd.DataFrame = field(default_factory=pd.DataFrame)
    replication_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    proxy_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_pca_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_weight_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_return_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    cumulative_factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    plot_status: pd.DataFrame = field(default_factory=pd.DataFrame)
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


def _explained_variation_from_pca(res: PCAResult, K: Optional[int] = None) -> float:
    """计算前 K 个 PCA 特征值解释的变异占比。"""
    eigvals = np.asarray(res.eigvals, dtype=float)
    if K is None:
        K = eigvals.shape[0]
    total = float(np.nansum(eigvals))
    return float(np.nansum(eigvals[:K]) / total) if total > 0 else 0.0


def build_paper_jump_stats(panel: HFPanel, thresholds: Sequence[float] = (3.0, 4.0, 4.5, 5.0)) -> pd.DataFrame:
    """生成更接近论文 Table I 的年度多阈值跳跃统计。"""
    rows: List[Dict[str, Any]] = []
    years = sorted({date.year for date in panel.dates})
    for year in years:
        year_panel = subset_panel_by_years(panel, [year])
        for a in thresholds:
            R_cont, R_jump = detect_jumps(year_panel, a=float(a))
            stats = jump_summary_stats(R_cont, R_jump)
            jump_res = pca_factors(R_jump, K=1, use_corr=True)
            cont_res = pca_factors(R_cont, K=min(4, year_panel.N), use_corr=True)
            rows.append({
                "year": int(year),
                "threshold_a": float(a),
                "n_symbols": int(year_panel.N),
                "n_days": int(year_panel.D),
                "frac_jump_increments": stats["frac_jump_increments"],
                "frac_qv_explained_by_jumps": stats["frac_qv_explained_by_jumps"],
                "jump_corr_explained_by_first_factor": _explained_variation_from_pca(jump_res, K=1),
                "continuous_corr_explained_by_first_four": _explained_variation_from_pca(cont_res, K=min(4, cont_res.eigvals.shape[0])),
            })
    return pd.DataFrame(rows)


def build_paper_factor_counts(
    panel: HFPanel,
    k_max: int,
    gamma: float,
    g_fn: str,
    jump_a: float,
) -> pd.DataFrame:
    """生成年度扰动特征值比率诊断表，对应论文 Figures 1-2 的数据基础。"""
    rows: List[Dict[str, Any]] = []
    years = sorted({date.year for date in panel.dates})
    for year in years:
        year_panel = subset_panel_by_years(panel, [year])
        R_cont, R_jump = detect_jumps(year_panel, a=jump_a)
        series = {
            "hf": pca_factors(year_panel.R_intra, K=1, use_corr=True).eigvals,
            "continuous": pca_factors(R_cont, K=1, use_corr=True).eigvals,
            "jump": pca_factors(R_jump, K=1, use_corr=True).eigvals,
        }
        for return_component, eigvals in series.items():
            K_hat, er = perturbed_eigenvalue_ratio(eigvals, g_fn=g_fn, N=year_panel.N, gamma=gamma, K_max=k_max)
            row = {
                "year": int(year),
                "return_component": return_component,
                "K_hat": int(max(1, K_hat)),
                "g_fn": g_fn,
                "gamma": float(gamma),
            }
            for idx, value in enumerate(er[:k_max], start=1):
                row[f"er_{idx}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def build_factor_sharpe_table(pipe: PelgerPipeline) -> pd.DataFrame:
    """输出连续 PCA 因子的切点组合和单因子 Sharpe，贴近论文 Table V 可由 K 线支持的部分。"""
    rows = [
        {
            "portfolio": "Continuous PCA tangency",
            "SR_intraday": pipe.sharpes.get("SR_intraday", np.nan),
            "SR_overnight": pipe.sharpes.get("SR_overnight", np.nan),
            "SR_daily": pipe.sharpes.get("SR_daily", np.nan),
        }
    ]
    if pipe.F_cont_daily_intra is not None and pipe.F_cont_daily_night is not None:
        scale = np.sqrt(252)
        F_daily = pipe.F_cont_daily_intra + pipe.F_cont_daily_night
        for idx in range(pipe.F_cont_daily_intra.shape[1]):
            rows.append({
                "portfolio": f"{idx + 1}. Continuous PCA Factor",
                "SR_intraday": float(np.nanmean(pipe.F_cont_daily_intra[:, idx]) / (np.nanstd(pipe.F_cont_daily_intra[:, idx], ddof=1) or np.nan) * scale),
                "SR_overnight": float(np.nanmean(pipe.F_cont_daily_night[:, idx]) / (np.nanstd(pipe.F_cont_daily_night[:, idx], ddof=1) or np.nan) * scale),
                "SR_daily": float(np.nanmean(F_daily[:, idx]) / (np.nanstd(F_daily[:, idx], ddof=1) or np.nan) * scale),
            })
    return pd.DataFrame(rows)


def build_replication_coverage_report() -> pd.DataFrame:
    """列出论文表图在当前 A 股 K 线数据条件下的复现状态。"""
    rows = [
        ("Table I", "Yearly jump and explained-variation statistics", "china_adapted", "已用 A 股 48 根 5 分钟后复权收益输出年度多阈值统计。"),
        ("Table II", "Balanced/unbalanced factor-space comparison", "china_adapted", "已用 strict_balanced 与 near_balanced_99 年度 GC 做稳健性对照。"),
        ("Table III", "Generalized correlations with industry and FFC factors", "external_data_required", "需要行业因子、Fama-French-Carhart 因子及相同频率收益。"),
        ("Table IV", "Time-variation decomposition across frequencies", "china_adapted", "已输出滚动 GC、解释度和可支持版结构分解。"),
        ("Table V", "Intraday/overnight/daily Sharpe ratios", "china_adapted", "已输出连续 PCA 因子可支持的 Sharpe；FFC 对照需要外部因子。"),
        ("Figure 1", "Perturbed eigenvalue ratio diagnostics for balanced panel", "china_adapted", "严格平衡 A 股样本，48 bars/day。"),
        ("Figure 2", "Perturbed eigenvalue ratio diagnostics for unbalanced panel", "china_adapted", "用年度 99% 样本稳健性结果近似展示。"),
        ("Figure 3", "Proxy factor portfolio weights", "china_adapted", "用连续 PCA 权重构造 proxy factors。"),
        ("Figure 4", "Continuous PCA factor portfolio weights", "china_adapted", "输出前 4 个连续 PCA 因子权重。"),
        ("Figure 5", "Monthly PCA factor portfolio weights", "china_adapted", "由日度收益聚合到月频后做 PCA。"),
        ("Figure 6", "Time variation in loadings", "china_adapted", "用滚动 GC 展示局部载荷稳定性。"),
        ("Figure 7", "Locally estimated continuous factors", "china_adapted", "用滚动局部连续因子与全局因子 GC 展示。"),
        ("Figure 8", "Time-varying portfolio weights", "china_adapted", "展示滚动窗口下 top 权重股票的权重变化。"),
        ("Figure 9", "Time-varying explained variation", "china_adapted", "输出滚动解释度。"),
        ("Figure 10", "Factor-structure time variation decomposition", "china_adapted", "用 GC 和解释度构造可支持版分解图。"),
        ("Figure 11", "Continuous factor-structure decomposition", "china_adapted", "连续因子滚动稳定性和解释度分解。"),
        ("Figure 12", "Expected intraday and overnight returns", "china_adapted", "用连续 PCA 因子日内/隔夜/日度均值。"),
        ("Figure 13", "Cumulative factor returns", "china_adapted", "用连续 PCA 因子累计收益。"),
        ("Figure 14", "Asset pricing of industry portfolios", "external_data_required", "需要行业组合收益；当前输出占位图。"),
        ("Figure 15", "Asset pricing of size/value portfolios", "external_data_required", "需要 size/value 测试资产组合；当前输出占位图。"),
    ]
    return pd.DataFrame(rows, columns=["paper_item", "paper_content", "status", "notes"])


def build_weight_tables(pipe: PelgerPipeline, panel: HFPanel, top_n: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """导出连续 PCA 权重和 proxy 权重，支持论文 Figure 3-4。"""
    W = factor_portfolio_weights(pipe.pca_cont)
    W_proxy, _ = build_proxy_factors(W, pipe.R_cont)
    k_count = min(W.shape[1], 4)
    rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    for k in range(k_count):
        order = np.argsort(-np.abs(W[:, k]))[: min(top_n, len(panel.tickers))]
        for rank, idx in enumerate(order, start=1):
            rows.append({"factor": k + 1, "rank": rank, "symbol": panel.tickers[idx], "weight": float(W[idx, k])})
            proxy_rows.append({"factor": k + 1, "rank": rank, "symbol": panel.tickers[idx], "weight": float(W_proxy[idx, k])})
    return pd.DataFrame(rows), pd.DataFrame(proxy_rows)


def build_monthly_pca_weights(panel: HFPanel, K: int = 4, top_n: int = 30) -> pd.DataFrame:
    """把日收益聚合到月频后做 PCA，支持论文 Figure 5 的 A 股适配版。"""
    months = pd.Index([date.to_period("M").to_timestamp() for date in panel.dates])
    month_labels = sorted(months.unique())
    monthly = np.zeros((len(month_labels), panel.N), dtype=float)
    for month_idx, month in enumerate(month_labels):
        day_mask = months == month
        monthly[month_idx] = np.nansum(panel.R_daily[day_mask], axis=0)
    if monthly.shape[0] < 2:
        return pd.DataFrame(columns=["factor", "rank", "symbol", "weight"])
    res = pca_factors(monthly, K=min(K, panel.N), use_corr=True)
    W = factor_portfolio_weights(res)
    rows: List[Dict[str, Any]] = []
    for k in range(min(W.shape[1], K)):
        order = np.argsort(-np.abs(W[:, k]))[: min(top_n, len(panel.tickers))]
        for rank, idx in enumerate(order, start=1):
            rows.append({"factor": k + 1, "rank": rank, "symbol": panel.tickers[idx], "weight": float(W[idx, k])})
    return pd.DataFrame(rows)


def build_rolling_weight_summary(panel: HFPanel, R_cont: np.ndarray, K: int, window_days: int, step_days: int = 21, top_n: int = 8) -> pd.DataFrame:
    """生成滚动 PCA 权重摘要，支持论文 Figure 8。"""
    W_global = factor_portfolio_weights(pca_factors(R_cont, K=K, use_corr=True))
    selected_idx = np.argsort(-np.abs(W_global[:, 0]))[: min(top_n, panel.N)]
    rows: List[Dict[str, Any]] = []
    D = int(panel.day_ids.max()) + 1
    for start in range(0, D - window_days + 1, step_days):
        end = start + window_days
        mask = (panel.day_ids >= start) & (panel.day_ids < end)
        if mask.sum() < 2 * K:
            continue
        W_local = factor_portfolio_weights(pca_factors(R_cont[mask], K=K, use_corr=True))
        for idx in selected_idx:
            rows.append({
                "start_day": int(start),
                "end_day": int(end),
                "symbol": panel.tickers[idx],
                "weight_factor_1": float(W_local[idx, 0]),
            })
    return pd.DataFrame(rows)


def build_factor_return_tables(pipe: PelgerPipeline, panel: HFPanel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成连续 PCA 因子的均值和累计收益，支持论文 Figure 12-13。"""
    F_intra = pipe.F_cont_daily_intra
    F_night = pipe.F_cont_daily_night
    F_daily = F_intra + F_night
    rows: List[Dict[str, Any]] = []
    cum_rows: List[Dict[str, Any]] = []
    for k in range(F_intra.shape[1]):
        rows.append({
            "factor": k + 1,
            "mean_intraday": float(np.nanmean(F_intra[:, k])),
            "mean_overnight": float(np.nanmean(F_night[:, k])),
            "mean_daily": float(np.nanmean(F_daily[:, k])),
        })
        cum_intra = np.cumsum(np.nan_to_num(F_intra[:, k], nan=0.0))
        cum_night = np.cumsum(np.nan_to_num(F_night[:, k], nan=0.0))
        cum_daily = np.cumsum(np.nan_to_num(F_daily[:, k], nan=0.0))
        for day_idx, date in enumerate(panel.dates):
            cum_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "factor": k + 1,
                "cum_intraday": float(cum_intra[day_idx]),
                "cum_overnight": float(cum_night[day_idx]),
                "cum_daily": float(cum_daily[day_idx]),
            })
    return pd.DataFrame(rows), pd.DataFrame(cum_rows)


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


def _plot_status_row(figure_id: str, title: str, output_path: Path, status: str, data_source: str, notes: str) -> Dict[str, str]:
    return {
        "figure_id": figure_id,
        "title": title,
        "file_path": str(output_path),
        "status": status,
        "data_source": data_source,
        "notes": notes,
    }


def _save_placeholder_figure(output_path: Path, title: str, message: str) -> None:
    """生成明确的占位图，不让缺外部数据的论文图静默缺失。"""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    ax.text(0.5, 0.22, "External data required", ha="center", va="center", fontsize=18, alpha=0.25, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_line_plot(df: pd.DataFrame, x_col: str, y_cols: Sequence[str], title: str, output_path: Path, ylabel: str = "") -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x_col], df[col], label=col, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    if ylabel:
        ax.set_ylabel(ylabel)
    if len(y_cols) > 1:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, group_col: Optional[str] = None, ylabel: str = "") -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    if group_col and group_col in df.columns:
        pivot = df.pivot_table(index=x_col, columns=group_col, values=y_col, aggfunc="first").fillna(0.0)
        pivot.plot(kind="bar", ax=ax, width=0.82)
    else:
        ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_heatmap(matrix: np.ndarray, x_labels: Sequence[str], y_labels: Sequence[str], title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(list(x_labels), rotation=75, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(list(y_labels))
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_all_paper_figures(
    result: ReplicationResult,
    figures_dir: Path,
    rolling_gc_df: pd.DataFrame,
    rolling_explained_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """按论文 Figure 1-15 生成真实图或明确占位图。"""
    exported: Dict[str, str] = {}
    status_rows: List[Dict[str, str]] = []
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        for idx in range(1, 16):
            figure_id = f"Figure_{idx:02d}"
            output_path = figures_dir / f"{figure_id}_matplotlib_missing.png"
            status_rows.append(_plot_status_row(figure_id, figure_id, output_path, "error", "matplotlib", repr(exc)))
        return pd.DataFrame(status_rows), exported

    def run_plot(figure_id: str, file_name: str, title: str, data_source: str, plot_func, notes: str = "") -> None:
        output_path = figures_dir / file_name
        try:
            plot_func(output_path)
            status = "placeholder" if "External data required" in notes else "generated"
            status_rows.append(_plot_status_row(figure_id, title, output_path, status, data_source, notes))
            exported[figure_id] = str(output_path)
        except Exception as exc:
            try:
                _save_placeholder_figure(output_path, title, f"Plot generation failed: {exc!r}")
                status_rows.append(_plot_status_row(figure_id, title, output_path, "error", data_source, repr(exc)))
                exported[figure_id] = str(output_path)
            except Exception:
                status_rows.append(_plot_status_row(figure_id, title, output_path, "error", data_source, repr(exc)))

    def er_plot(component: str, title: str):
        def _plot(output_path: Path) -> None:
            df = result.paper_factor_counts.loc[result.paper_factor_counts["return_component"].eq(component)].copy()
            er_cols = [col for col in df.columns if col.startswith("er_")]
            if df.empty or not er_cols:
                _save_placeholder_figure(output_path, title, "No perturbed eigenvalue-ratio data available.")
                return
            long_df = df[["year", *er_cols]].melt(id_vars="year", var_name="ratio_index", value_name="perturbed_er")
            long_df["x_label"] = long_df["year"].astype(str) + ":" + long_df["ratio_index"].str.replace("er_", "k", regex=False)
            _save_line_plot(long_df, "x_label", ["perturbed_er"], title, output_path, ylabel="Perturbed ER")
        return _plot

    run_plot("Figure_01", "Figure_01_perturbed_er_balanced.png", "Figure 1. Perturbed Eigenvalue Ratios, Balanced Panel", "Table_09_paper_style_factor_count_diagnostics", er_plot("continuous", "Figure 1. Perturbed Eigenvalue Ratios, Balanced Panel"))

    def fig02(output_path: Path) -> None:
        if result.robustness.empty:
            _save_placeholder_figure(output_path, "Figure 2. Perturbed ER / Near-Balanced Robustness", "Near-balanced yearly robustness was not run or no near-balanced panel is available.")
            return
        gc_cols = [col for col in result.robustness.columns if col.startswith("gc_")]
        _save_line_plot(result.robustness, "year", gc_cols, "Figure 2. Near-Balanced Factor-Space Robustness", output_path, ylabel="Generalized correlation")
    run_plot("Figure_02", "Figure_02_perturbed_er_near_balanced.png", "Figure 2. Near-Balanced Diagnostics", "Table_07_robustness_yearly_gc", fig02)

    def weight_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
        if df.empty:
            _save_placeholder_figure(output_path, title, "No PCA weight data available for this sample.")
            return
        pivot = df.pivot_table(index="factor", columns="symbol", values="weight", aggfunc="first").fillna(0.0)
        _save_heatmap(pivot.to_numpy(), pivot.columns.tolist(), [f"Factor {idx}" for idx in pivot.index], title, output_path)

    run_plot("Figure_03", "Figure_03_proxy_factor_weights.png", "Figure 3. Proxy Factor Portfolio Weights", "proxy_weights", lambda path: weight_heatmap(result.proxy_weights, "Figure 3. Proxy Factor Portfolio Weights", path))
    run_plot("Figure_04", "Figure_04_continuous_pca_weights.png", "Figure 4. Continuous PCA Factor Weights", "pca_weights", lambda path: weight_heatmap(result.pca_weights, "Figure 4. Continuous PCA Factor Weights", path))
    run_plot("Figure_05", "Figure_05_monthly_pca_weights.png", "Figure 5. Monthly PCA Factor Weights", "monthly_pca_weights", lambda path: weight_heatmap(result.monthly_pca_weights, "Figure 5. Monthly PCA Factor Weights", path))

    gc_cols = [col for col in rolling_gc_df.columns if col.startswith("gc_")]
    run_plot("Figure_06", "Figure_06_time_variation_loadings_gc.png", "Figure 6. Time Variation in Loadings", "Table_05_rolling_gc", lambda path: _save_line_plot(rolling_gc_df, "window_index", gc_cols, "Figure 6. Time Variation in Loadings", path, ylabel="Generalized correlation"))
    run_plot("Figure_07", "Figure_07_local_continuous_factor_gc.png", "Figure 7. Locally Estimated Continuous Factors", "Table_05_rolling_gc", lambda path: _save_line_plot(rolling_gc_df, "window_index", gc_cols[: min(4, len(gc_cols))], "Figure 7. Locally Estimated Continuous Factors", path, ylabel="Generalized correlation"))

    def fig08(output_path: Path) -> None:
        if result.rolling_weight_summary.empty:
            _save_placeholder_figure(output_path, "Figure 8. Time-Varying Portfolio Weights", "No rolling weight summary available.")
            return
        pivot = result.rolling_weight_summary.pivot_table(index="start_day", columns="symbol", values="weight_factor_1", aggfunc="first").reset_index()
        y_cols = [col for col in pivot.columns if col != "start_day"]
        _save_line_plot(pivot, "start_day", y_cols, "Figure 8. Time-Varying Portfolio Weights", output_path, ylabel="Factor 1 weight")
    run_plot("Figure_08", "Figure_08_time_varying_portfolio_weights.png", "Figure 8. Time-Varying Portfolio Weights", "rolling_weight_summary", fig08)

    run_plot("Figure_09", "Figure_09_time_varying_explained_variation.png", "Figure 9. Time-Varying Explained Variation", "Table_06_rolling_explained_variation", lambda path: _save_line_plot(rolling_explained_df, "window_index", ["explained_variation"], "Figure 9. Time-Varying Explained Variation", path, ylabel="Explained variation"))

    def fig10(output_path: Path) -> None:
        df = rolling_gc_df.copy()
        if df.empty:
            _save_placeholder_figure(output_path, "Figure 10. Factor-Structure Decomposition", "No rolling GC data available.")
            return
        df["avg_gc"] = df[gc_cols].mean(axis=1) if gc_cols else np.nan
        if not rolling_explained_df.empty:
            df = df.merge(rolling_explained_df, on="window_index", how="left")
        _save_line_plot(df, "window_index", [col for col in ["avg_gc", "explained_variation"] if col in df.columns], "Figure 10. Factor-Structure Decomposition", output_path)
    run_plot("Figure_10", "Figure_10_factor_structure_decomposition.png", "Figure 10. Factor-Structure Decomposition", "rolling_gc_and_explained_variation", fig10)

    def fig11(output_path: Path) -> None:
        df = rolling_gc_df.copy()
        if df.empty:
            _save_placeholder_figure(output_path, "Figure 11. Continuous Factor-Structure Decomposition", "No rolling GC data available.")
            return
        df["min_gc"] = df[gc_cols].min(axis=1) if gc_cols else np.nan
        df["mean_gc"] = df[gc_cols].mean(axis=1) if gc_cols else np.nan
        _save_line_plot(df, "window_index", ["min_gc", "mean_gc"], "Figure 11. Continuous Factor-Structure Decomposition", output_path, ylabel="Generalized correlation")
    run_plot("Figure_11", "Figure_11_continuous_factor_structure_decomposition.png", "Figure 11. Continuous Factor-Structure Decomposition", "rolling_gc", fig11)

    def fig12(output_path: Path) -> None:
        if result.factor_return_summary.empty:
            _save_placeholder_figure(output_path, "Figure 12. Expected Intraday and Overnight Returns", "No factor return summary available.")
            return
        long_df = result.factor_return_summary.melt(id_vars="factor", var_name="segment", value_name="mean_return")
        _save_bar_plot(long_df, "factor", "mean_return", "Figure 12. Expected Intraday and Overnight Returns", output_path, group_col="segment", ylabel="Mean return")
    run_plot("Figure_12", "Figure_12_expected_intraday_overnight_returns.png", "Figure 12. Expected Intraday and Overnight Returns", "factor_return_summary", fig12)

    def fig13(output_path: Path) -> None:
        if result.cumulative_factor_returns.empty:
            _save_placeholder_figure(output_path, "Figure 13. Cumulative Factor Returns", "No cumulative factor return data available.")
            return
        df = result.cumulative_factor_returns.loc[result.cumulative_factor_returns["factor"].eq(1)].copy()
        _save_line_plot(df, "date", ["cum_intraday", "cum_overnight", "cum_daily"], "Figure 13. Cumulative Factor Returns, Factor 1", output_path, ylabel="Cumulative log return")
    run_plot("Figure_13", "Figure_13_cumulative_factor_returns.png", "Figure 13. Cumulative Factor Returns", "cumulative_factor_returns", fig13)

    run_plot("Figure_14", "Figure_14_asset_pricing_industry_portfolios.png", "Figure 14. Asset Pricing of Industry Portfolios", "external_test_assets", lambda path: _save_placeholder_figure(path, "Figure 14. Asset Pricing of Industry Portfolios", "Industry portfolio returns are not available in the current repository. Provide external test-asset CSV files to replace this placeholder."), notes="External data required: industry portfolio returns")
    run_plot("Figure_15", "Figure_15_asset_pricing_size_value_portfolios.png", "Figure 15. Asset Pricing of Size- and Value-Sorted Portfolios", "external_test_assets", lambda path: _save_placeholder_figure(path, "Figure 15. Asset Pricing of Size- and Value-Sorted Portfolios", "Size/value sorted portfolio returns are not available in the current repository. Provide external test-asset CSV files to replace this placeholder."), notes="External data required: size/value portfolio returns")

    return pd.DataFrame(status_rows).sort_values("figure_id").reset_index(drop=True), exported


def export_replication_outputs(
    result: ReplicationResult,
    save_plots: bool = True,
) -> Dict[str, str]:
    """导出核心表格、摘要和可选图形."""
    output_root = result.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    diagnostics_dir = output_root / "diagnostics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    universe_summary = summarize_cn_universe(result.universe)
    sample_report = result.panel.sample_report or {}
    main_summary = {
        "sample_report": sample_report,
        "jump_stats": result.pipeline.jump_stats,
        "factor_counts": {
            "K_hf_hat": result.pipeline.K_hf_hat,
            "K_cont_hat": result.pipeline.K_cont_hat,
            "K_jump_hat": result.pipeline.K_jump_hat,
            "g_fn": result.pipeline.g_fn,
            "gamma": result.pipeline.gamma,
        },
        "sharpes": result.pipeline.sharpes,
    }

    universe_csv = diagnostics_dir / "universe_scan.csv"
    result.universe.to_csv(universe_csv, index=False, encoding="utf-8-sig")
    _write_json(diagnostics_dir / "universe_summary.json", universe_summary)
    _write_json(diagnostics_dir / "main_summary.json", main_summary)

    sample_summary_path = tables_dir / "Table_01_sample_summary.csv"
    pd.DataFrame([
        {
            "adjustment": sample_report.get("adjustment", "raw_or_unknown"),
            "sample_mode": result.panel.sample_mode,
            "return_mode": result.panel.return_mode,
            "years": sample_report.get("years"),
            "n_symbols_selected": result.panel.N,
            "n_days_selected": result.panel.D,
            "bars_per_day": result.panel.M_per_day,
            "selected_calendar_start": sample_report.get("selected_calendar_start"),
            "selected_calendar_end": sample_report.get("selected_calendar_end"),
            "universe_total_symbols": universe_summary.get("total_symbols"),
            "strict_balanced_symbols": universe_summary.get("strict_balanced_symbols"),
            "near_balanced_99_symbols": universe_summary.get("near_balanced_99_symbols"),
            "global_start": universe_summary.get("global_start"),
            "global_end": universe_summary.get("global_end"),
        }
    ]).to_csv(sample_summary_path, index=False, encoding="utf-8-sig")

    jump_stats_path = tables_dir / "Table_02_jump_stats.csv"
    pd.DataFrame([result.pipeline.jump_stats]).to_csv(
        jump_stats_path, index=False, encoding="utf-8-sig"
    )
    factor_counts_path = tables_dir / "Table_03_factor_counts.csv"
    pd.DataFrame(
        [{
            "K_hf_hat": result.pipeline.K_hf_hat,
            "K_cont_hat": result.pipeline.K_cont_hat,
            "K_jump_hat": result.pipeline.K_jump_hat,
            "g_fn": result.pipeline.g_fn,
            "gamma": result.pipeline.gamma,
        }]
    ).to_csv(factor_counts_path, index=False, encoding="utf-8-sig")
    sharpes_path = tables_dir / "Table_04_sharpes.csv"
    pd.DataFrame([result.pipeline.sharpes]).to_csv(
        sharpes_path, index=False, encoding="utf-8-sig"
    )

    sample_symbols_path = diagnostics_dir / "main_sample_symbols.csv"
    pd.DataFrame({"symbol": result.panel.tickers}).to_csv(
        sample_symbols_path,
        index=False,
        encoding="utf-8-sig",
    )

    rolling_gc_df, rolling_explained_df = _rolling_output_frames(
        result.rolling_gc,
        result.rolling_explained_variation,
    )
    rolling_gc_path = tables_dir / "Table_05_rolling_gc.csv"
    rolling_ev_path = tables_dir / "Table_06_rolling_explained_variation.csv"
    rolling_gc_df.to_csv(rolling_gc_path, index=False, encoding="utf-8-sig")
    rolling_explained_df.to_csv(rolling_ev_path, index=False, encoding="utf-8-sig")

    robustness_path = tables_dir / "Table_07_robustness_yearly_gc.csv"
    result.robustness.to_csv(robustness_path, index=False, encoding="utf-8-sig")

    paper_jump_path = tables_dir / "Table_08_paper_style_yearly_jump_stats.csv"
    result.paper_jump_stats.to_csv(paper_jump_path, index=False, encoding="utf-8-sig")
    paper_factor_counts_path = tables_dir / "Table_09_paper_style_factor_count_diagnostics.csv"
    result.paper_factor_counts.to_csv(paper_factor_counts_path, index=False, encoding="utf-8-sig")
    paper_sharpes_path = tables_dir / "Table_10_paper_style_factor_sharpes.csv"
    result.paper_factor_sharpes.to_csv(paper_sharpes_path, index=False, encoding="utf-8-sig")

    pca_weights_path = tables_dir / "Table_11_continuous_pca_weights.csv"
    result.pca_weights.to_csv(pca_weights_path, index=False, encoding="utf-8-sig")
    proxy_weights_path = tables_dir / "Table_12_proxy_factor_weights.csv"
    result.proxy_weights.to_csv(proxy_weights_path, index=False, encoding="utf-8-sig")
    monthly_weights_path = tables_dir / "Table_13_monthly_pca_weights.csv"
    result.monthly_pca_weights.to_csv(monthly_weights_path, index=False, encoding="utf-8-sig")
    factor_return_summary_path = tables_dir / "Table_14_factor_return_summary.csv"
    result.factor_return_summary.to_csv(factor_return_summary_path, index=False, encoding="utf-8-sig")

    corp_action_path = diagnostics_dir / "corp_action_risk_after_adjustment.csv"
    result.corp_action_risk.to_csv(corp_action_path, index=False, encoding="utf-8-sig")
    coverage_path = diagnostics_dir / "replication_coverage_report.csv"
    result.replication_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    rolling_weight_path = diagnostics_dir / "rolling_weight_summary.csv"
    result.rolling_weight_summary.to_csv(rolling_weight_path, index=False, encoding="utf-8-sig")
    cumulative_returns_path = diagnostics_dir / "cumulative_factor_returns.csv"
    result.cumulative_factor_returns.to_csv(cumulative_returns_path, index=False, encoding="utf-8-sig")

    exported_files = {
        "universe_scan": str(universe_csv),
        "universe_summary": str(diagnostics_dir / "universe_summary.json"),
        "main_summary": str(diagnostics_dir / "main_summary.json"),
        "sample_summary": str(sample_summary_path),
        "jump_stats": str(jump_stats_path),
        "factor_counts": str(factor_counts_path),
        "sharpes": str(sharpes_path),
        "main_sample_symbols": str(sample_symbols_path),
        "rolling_gc": str(rolling_gc_path),
        "rolling_explained_variation": str(rolling_ev_path),
        "robustness_yearly_gc": str(robustness_path),
        "paper_style_yearly_jump_stats": str(paper_jump_path),
        "paper_style_factor_count_diagnostics": str(paper_factor_counts_path),
        "paper_style_factor_sharpes": str(paper_sharpes_path),
        "continuous_pca_weights": str(pca_weights_path),
        "proxy_factor_weights": str(proxy_weights_path),
        "monthly_pca_weights": str(monthly_weights_path),
        "factor_return_summary": str(factor_return_summary_path),
        "rolling_weight_summary": str(rolling_weight_path),
        "cumulative_factor_returns": str(cumulative_returns_path),
        "corp_action_risk": str(corp_action_path),
        "replication_coverage_report": str(coverage_path),
    }

    if save_plots:
        plot_status, figure_files = export_all_paper_figures(result, figures_dir, rolling_gc_df, rolling_explained_df)
    else:
        plot_status = pd.DataFrame([
            {
                "figure_id": f"Figure_{idx:02d}",
                "title": f"Figure {idx}",
                "file_path": "",
                "status": "skipped",
                "data_source": "--no-plots",
                "notes": "Plot export skipped because --no-plots was used.",
            }
            for idx in range(1, 16)
        ])
        figure_files = {}
    result.plot_status = plot_status
    for figure_id, path in figure_files.items():
        exported_files[figure_id] = path
    plot_status_path = diagnostics_dir / "plot_export_status.csv"
    plot_status.to_csv(plot_status_path, index=False, encoding="utf-8-sig")
    exported_files["plot_export_status"] = str(plot_status_path)

    counts = plot_status["status"].value_counts().to_dict() if not plot_status.empty else {}
    main_summary["plots_generated_count"] = int(counts.get("generated", 0))
    main_summary["plots_placeholder_count"] = int(counts.get("placeholder", 0))
    main_summary["plots_failed_count"] = int(counts.get("error", 0))
    main_summary["plots_skipped_count"] = int(counts.get("skipped", 0))
    _write_json(diagnostics_dir / "main_summary.json", main_summary)

    result.exported_files = exported_files
    return exported_files


def run_cn_replication(
    proc_root: str | Path = DEFAULT_PROC_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    years: Optional[Sequence[int]] = None,
    run_robustness: bool = True,
    return_mode: str = "open_close",
    sample_mode: str = STRICT_BALANCED_SAMPLE,
    max_stocks: Optional[int] = None,
    jump_a: float = 3.0,
    k_max: int = 10,
    gamma: float = 0.08,
    g_fn: str = "median_N",
    save_plots: bool = True,
    universe: Optional[pd.DataFrame] = None,
    workers: Optional[int] = None,
) -> ReplicationResult:
    """Run the China A-share replication from preprocessed proc_Data panels only."""
    proc_root = _ensure_path(proc_root)
    output_root = _ensure_path(output_root)
    workers = _normalize_worker_count(workers)
    manifest_path = proc_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed data manifest not found: {manifest_path}. Run Code/preprocess_cn_data.py first."
        )

    universe = load_proc_universe(proc_root) if universe is None else universe
    panel = load_proc_hf_panel(
        proc_root=proc_root,
        sample_mode=sample_mode,
        years=years,
        return_mode=return_mode,
        max_stocks=max_stocks,
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

    pipeline = PelgerPipeline(panel=panel, jump_a=jump_a, K_max=k_max, gamma=gamma, g_fn=g_fn).run_full()
    rolling_window = 21 if panel.D >= 21 else max(5, panel.D // 3)
    rolling_gc, rolling_ev = rolling_gc_and_explained_variation(
        R=pipeline.R_cont,
        day_ids=panel.day_ids,
        window_days=max(rolling_window, 2),
        K=pipeline.K_cont_hat,
        global_Lambda=pipeline.pca_cont.Lambda,
        step_days=1,
        workers=workers,
    )

    robustness = pd.DataFrame()
    if run_robustness:
        robustness = run_near_balanced_robustness(
            proc_root=proc_root,
            universe=universe,
            years=years,
            K_compare=pipeline.K_cont_hat,
            return_mode=return_mode,
            jump_a=jump_a,
            workers=workers,
        )

    paper_jump_stats = build_paper_jump_stats(panel)
    paper_factor_counts = build_paper_factor_counts(panel, k_max=k_max, gamma=gamma, g_fn=g_fn, jump_a=jump_a)
    paper_factor_sharpes = build_factor_sharpe_table(pipeline)
    replication_coverage = build_replication_coverage_report()
    pca_weights, proxy_weights = build_weight_tables(pipeline, panel)
    monthly_pca_weights = build_monthly_pca_weights(panel)
    rolling_weight_summary = build_rolling_weight_summary(
        panel=panel,
        R_cont=pipeline.R_cont,
        K=pipeline.K_cont_hat,
        window_days=max(rolling_window, 2),
    )
    factor_return_summary, cumulative_factor_returns = build_factor_return_tables(pipeline, panel)

    result = ReplicationResult(
        universe=universe,
        panel=panel,
        pipeline=pipeline,
        rolling_gc=rolling_gc,
        rolling_explained_variation=rolling_ev,
        robustness=robustness,
        output_root=output_root,
        corp_action_risk=corp_action_risk,
        paper_jump_stats=paper_jump_stats,
        paper_factor_counts=paper_factor_counts,
        paper_factor_sharpes=paper_factor_sharpes,
        replication_coverage=replication_coverage,
        pca_weights=pca_weights,
        proxy_weights=proxy_weights,
        monthly_pca_weights=monthly_pca_weights,
        rolling_weight_summary=rolling_weight_summary,
        factor_return_summary=factor_return_summary,
        cumulative_factor_returns=cumulative_factor_returns,
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
        description="Run Pelger (2020) replication on preprocessed China A-share 5-minute data."
    )
    parser.add_argument("--proc-root", default=str(DEFAULT_PROC_ROOT), help="Preprocessed data directory")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Result output directory")
    parser.add_argument("--sample-mode", default=STRICT_BALANCED_SAMPLE, choices=[STRICT_BALANCED_SAMPLE, NEAR_BALANCED_99_SAMPLE])
    parser.add_argument("--return-mode", default="open_close", choices=["open_close", "close_close"])
    parser.add_argument("--years", nargs="+", type=int, help="Run selected years, e.g. --years 2015 2016")
    parser.add_argument("--max-stocks", type=int, help="Use first N symbols for smoke tests")
    parser.add_argument("--no-robustness", action="store_true", help="Skip yearly 99%%-coverage robustness")
    parser.add_argument("--no-plots", action="store_true", help="Do not export PNG figures")
    parser.add_argument("--jump-a", type=float, default=3.0, help="Jump threshold multiplier")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum factor count search bound")
    parser.add_argument("--gamma", type=float, default=0.08, help="Eigenvalue-ratio perturbation threshold")
    parser.add_argument("--g-fn", default="median_N", choices=["median_N", "median_sqrtN", "logN", "none"], help="Perturbed eigenvalue shift; median_N matches the paper default")
    parser.add_argument("--workers", type=int, help="Parallel workers for rolling PCA and robustness; default min(cpu_count - 1, 8)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    proc_root = _ensure_path(args.proc_root)
    output_root = _ensure_path(args.output_root)
    universe = load_proc_universe(proc_root)
    _print_scan_summary(universe)

    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    universe.to_csv(diagnostics_dir / "universe_scan.csv", index=False, encoding="utf-8-sig")
    _write_json(diagnostics_dir / "universe_summary.json", summarize_cn_universe(universe))

    result = run_cn_replication(
        proc_root=proc_root,
        output_root=output_root,
        years=args.years,
        run_robustness=not args.no_robustness,
        return_mode=args.return_mode,
        sample_mode=args.sample_mode,
        max_stocks=args.max_stocks,
        jump_a=args.jump_a,
        k_max=args.k_max,
        gamma=args.gamma,
        g_fn=args.g_fn,
        save_plots=not args.no_plots,
        universe=universe,
        workers=args.workers,
    )
    print_pipeline_summary(result.pipeline)
    print("Exported files:")
    for name, path in sorted(result.exported_files.items()):
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
