from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prepareCore.engine import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROC_ROOT,
    PAPER_LENIENT_SAMPLE,
    STRICT_BALANCED_SAMPLE,
)


CACHE_SCHEMA_VERSION = 4
DEFAULT_EXTERNAL_DATA_ROOT = Path(DEFAULT_PROC_ROOT).parents[1] / "external_Data" / "pelger_tail"
DEFAULT_RUNTIME_ROOT = Path(DEFAULT_PROC_ROOT) / "runtime"
DEFAULT_FINAL_RESULT_ROOT = Path(DEFAULT_OUTPUT_ROOT)
DEFAULT_PAPER_TAIL_ROOT = Path(DEFAULT_PROC_ROOT) / "paper_tail"
DEFAULT_INDUSTRY_INFO_FILENAME = "stock_full_info_std_industry_final.csv"
DEFAULT_INDUSTRY_MAPPING_FILENAME = "\u884c\u4e1a\u6620\u5c04\u8868_\u7ec8\u7248.csv"
VALID_BALANCED_MODES = {STRICT_BALANCED_SAMPLE, PAPER_LENIENT_SAMPLE}
VALID_PAPER_TAIL_WEIGHTINGS = {"value_weighted", "equal_weighted"}
SUBMISSION_FROZEN_INDUSTRIES = (
    "\u5927\u91d1\u878d",
    "\u533b\u836f\u751f\u7269",
    "\u5468\u671f\u8d44\u6e90",
)


def _ensure_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


@dataclass
class RunConfig:
    proc_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_PROC_ROOT))
    runtime_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_RUNTIME_ROOT))
    final_result_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_FINAL_RESULT_ROOT))
    output_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_FINAL_RESULT_ROOT))

    years: Optional[List[int]] = None
    max_stocks: Optional[int] = None
    balanced_mode: str = STRICT_BALANCED_SAMPLE

    return_mode: str = "open_close"
    jump_a: float = 3.0
    k_max: int = 10
    gamma: float = 0.08
    g_fn: str = "median_sqrtN"

    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None
    progress_interval_sec: float = 10.0

    external_data_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_EXTERNAL_DATA_ROOT))
    paper_tail_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_PAPER_TAIL_ROOT))
    paper_tail_weighting: str = "value_weighted"
    refresh_paper_tail: bool = True
    strict_final_export: bool = False

    paper_faithful_signs: bool = True
    industry_factors_frozen: Optional[List[str]] = None
    ffc_mom_mode: str = "carhart_daily"
    size_value_full_market: bool = True
    annualization_days: int = 252
    size_value_start: str = "2014-07-01"
    industry_info_filename: str = DEFAULT_INDUSTRY_INFO_FILENAME
    industry_mapping_filename: str = DEFAULT_INDUSTRY_MAPPING_FILENAME

    save_plots: bool = True
    restart: bool = False

    def __post_init__(self) -> None:
        self.proc_root = _ensure_path(self.proc_root)
        self.runtime_root = _ensure_path(self.runtime_root)
        self.final_result_root = _ensure_path(self.final_result_root)
        self.output_root = _ensure_path(self.output_root)
        self.external_data_root = _ensure_path(self.external_data_root)
        self.paper_tail_root = _ensure_path(self.paper_tail_root)
        if self.years is not None:
            self.years = [int(year) for year in self.years]
        if self.output_root != self.final_result_root:
            self.output_root = self.final_result_root

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "proc_root": str(self.proc_root),
            "runtime_root": str(self.runtime_root),
            "final_result_root": str(self.final_result_root),
            "output_root": str(self.output_root),
            "years": self.years,
            "balanced_mode": self.balanced_mode,
            "return_mode": self.return_mode,
            "max_stocks": self.max_stocks,
            "jump_a": self.jump_a,
            "k_max": self.k_max,
            "gamma": self.gamma,
            "g_fn": self.g_fn,
            "save_plots": self.save_plots,
            "workers": self.workers,
            "paper_workers": self.paper_workers,
            "rolling_workers": self.rolling_workers,
            "memory_budget_gb": self.memory_budget_gb,
            "progress_interval_sec": self.progress_interval_sec,
            "external_data_root": str(self.external_data_root),
            "paper_tail_root": str(self.paper_tail_root),
            "paper_tail_weighting": self.paper_tail_weighting,
            "refresh_paper_tail": self.refresh_paper_tail,
            "strict_final_export": self.strict_final_export,
            "restart": self.restart,
        }

    def export_fidelity_env(self) -> None:
        import os as _os

        _os.environ["PELGER_PAPER_FAITHFUL_SIGNS"] = "1" if self.paper_faithful_signs else "0"
        _os.environ["PELGER_FFC_MOM_MODE"] = str(self.ffc_mom_mode)
        _os.environ["PELGER_SIZE_VALUE_FULL_MARKET"] = "1" if self.size_value_full_market else "0"
        _os.environ["PELGER_ANNUALIZATION_DAYS"] = str(int(self.annualization_days))
        _os.environ["PELGER_SIZE_VALUE_START"] = str(self.size_value_start or "")
        _os.environ["PELGER_INDUSTRY_INFO_FILENAME"] = str(self.industry_info_filename)
        _os.environ["PELGER_INDUSTRY_MAPPING_FILENAME"] = str(self.industry_mapping_filename)
        _os.environ["PELGER_BALANCED_MODE"] = str(self.balanced_mode)
        _os.environ["PELGER_STRICT_FINAL_EXPORT"] = "1" if self.strict_final_export else "0"
        if self.industry_factors_frozen:
            _os.environ["PELGER_INDUSTRY_FROZEN"] = ",".join(self.industry_factors_frozen)
        else:
            _os.environ.pop("PELGER_INDUSTRY_FROZEN", None)

    def cache_signature(self) -> Dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "proc_root": str(self.proc_root.resolve()),
            "years": tuple(self.years) if self.years is not None else None,
            "max_stocks": self.max_stocks,
            "balanced_mode": self.balanced_mode,
            "return_mode": self.return_mode,
            "jump_a": self.jump_a,
            "k_max": self.k_max,
            "gamma": self.gamma,
            "g_fn": self.g_fn,
        }

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in (
            "proc_root",
            "runtime_root",
            "final_result_root",
            "output_root",
            "external_data_root",
            "paper_tail_root",
        ):
            data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class MainLaunchProfile:
    task_selectors: Tuple[str, ...] = ("all",)
    list_tasks_only: bool = False
    rebuild_result: bool = False
    restart: bool = False
    fail_fast: bool = False
    enable_heartbeat: bool = True
    heartbeat_sec: float = 10.0

    proc_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_PROC_ROOT))
    runtime_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_RUNTIME_ROOT))
    final_result_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_FINAL_RESULT_ROOT))
    output_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_FINAL_RESULT_ROOT))
    years: Optional[Tuple[int, ...]] = None
    max_stocks: Optional[int] = None
    balanced_mode: str = STRICT_BALANCED_SAMPLE
    return_mode: str = "open_close"
    jump_a: float = 3.0
    k_max: int = 10
    gamma: float = 0.08
    g_fn: str = "median_sqrtN"
    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None
    external_data_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_EXTERNAL_DATA_ROOT))
    paper_tail_root: Path = field(default_factory=lambda: _ensure_path(DEFAULT_PAPER_TAIL_ROOT))
    paper_tail_weighting: str = "value_weighted"
    refresh_paper_tail: bool = True
    strict_final_export: bool = False
    industry_factors_frozen: Optional[Tuple[str, ...]] = None
    industry_info_filename: str = DEFAULT_INDUSTRY_INFO_FILENAME
    industry_mapping_filename: str = DEFAULT_INDUSTRY_MAPPING_FILENAME


MAIN_RUN_PROFILES: Dict[str, MainLaunchProfile] = {
    "reuse_export_smoke": MainLaunchProfile(
        task_selectors=("all",),
        rebuild_result=False,
        restart=False,
        balanced_mode=STRICT_BALANCED_SAMPLE,
        strict_final_export=False,
    ),
    "rebuild_proc_and_result": MainLaunchProfile(
        task_selectors=("all",),
        rebuild_result=True,
        restart=True,
        balanced_mode=PAPER_LENIENT_SAMPLE,
        strict_final_export=False,
    ),
    "final_paper_export": MainLaunchProfile(
        task_selectors=("all",),
        rebuild_result=True,
        restart=True,
        balanced_mode=PAPER_LENIENT_SAMPLE,
        strict_final_export=True,
        fail_fast=True,
    ),
    "diagnostics_only": MainLaunchProfile(
        task_selectors=("diagnostics",),
        rebuild_result=False,
        restart=False,
        balanced_mode=STRICT_BALANCED_SAMPLE,
    ),
    "figures_only": MainLaunchProfile(
        task_selectors=("figures",),
        rebuild_result=False,
        restart=False,
        balanced_mode=STRICT_BALANCED_SAMPLE,
    ),
    "tables_only": MainLaunchProfile(
        task_selectors=("tables",),
        rebuild_result=False,
        restart=False,
        balanced_mode=STRICT_BALANCED_SAMPLE,
    ),
    "fig13_only": MainLaunchProfile(
        task_selectors=("fig13",),
        rebuild_result=False,
        restart=False,
        fail_fast=True,
        heartbeat_sec=5.0,
        balanced_mode=STRICT_BALANCED_SAMPLE,
    ),
    "submission_core": MainLaunchProfile(
        task_selectors=("fig1", "fig2", "fig4", "fig7", "fig10", "fig12", "fig13", "fig14", "fig15"),
        rebuild_result=False,
        restart=False,
        fail_fast=True,
        heartbeat_sec=5.0,
        balanced_mode=STRICT_BALANCED_SAMPLE,
        strict_final_export=True,
        industry_factors_frozen=SUBMISSION_FROZEN_INDUSTRIES,
    ),
    "submission_core_rebuild": MainLaunchProfile(
        task_selectors=("fig1", "fig2", "fig4", "fig7", "fig10", "fig12", "fig13", "fig14", "fig15"),
        rebuild_result=True,
        restart=False,
        fail_fast=True,
        heartbeat_sec=5.0,
        balanced_mode=STRICT_BALANCED_SAMPLE,
        strict_final_export=True,
        industry_factors_frozen=SUBMISSION_FROZEN_INDUSTRIES,
    ),
}

MAIN_RUN_PROFILES["export_all"] = MAIN_RUN_PROFILES["reuse_export_smoke"]
MAIN_RUN_PROFILES["rebuild_all"] = MAIN_RUN_PROFILES["rebuild_proc_and_result"]

MAIN_RUN_PROFILES["final_paper_export_resume"] = replace(
    MAIN_RUN_PROFILES["final_paper_export"],
    restart=False,
)
MAIN_RUN_PROFILES["final_paper_export_hpc_48g"] = replace(
    MAIN_RUN_PROFILES["final_paper_export_resume"],
    paper_workers=8,
    rolling_workers=16,
    memory_budget_gb=48.0,
)
MAIN_RUN_PROFILES["final_paper_export_hpc_64g"] = replace(
    MAIN_RUN_PROFILES["final_paper_export_resume"],
    paper_workers=10,
    rolling_workers=16,
    memory_budget_gb=64.0,
)
MAIN_RUN_PROFILES["final_paper_export_hpc_96g"] = replace(
    MAIN_RUN_PROFILES["final_paper_export_resume"],
    paper_workers=13,
    rolling_workers=24,
    memory_budget_gb=96.0,
)
MAIN_RUN_PROFILES["final_paper_export_hpc"] = MAIN_RUN_PROFILES["final_paper_export_hpc_64g"]

ACTIVE_MAIN_PROFILE = "submission_core"


def available_main_profile_names() -> List[str]:
    return sorted(MAIN_RUN_PROFILES.keys())


def get_main_profile(profile_name: str) -> MainLaunchProfile:
    try:
        return MAIN_RUN_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"Main profile {profile_name!r} not found. Available: {', '.join(available_main_profile_names())}"
        ) from exc


def _validate_positive_optional_int(name: str, value: Optional[int], profile_name: str) -> None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"profile {profile_name!r}: {name} must be greater than 0.")


def validate_main_profile(profile_name: str, profile: MainLaunchProfile) -> None:
    if not isinstance(profile, MainLaunchProfile):
        raise TypeError(f"profile {profile_name!r} is not a MainLaunchProfile instance.")
    if not profile.list_tasks_only and not profile.task_selectors:
        raise ValueError(f"profile {profile_name!r}: task_selectors cannot be empty.")
    if profile.restart and not profile.rebuild_result:
        raise ValueError(f"profile {profile_name!r}: restart=True requires rebuild_result=True.")
    if profile.heartbeat_sec <= 0:
        raise ValueError(f"profile {profile_name!r}: heartbeat_sec must be greater than 0.")
    for path_name in (
        "proc_root",
        "runtime_root",
        "final_result_root",
        "output_root",
        "external_data_root",
        "paper_tail_root",
    ):
        if not isinstance(getattr(profile, path_name), Path):
            raise TypeError(f"profile {profile_name!r}: {path_name} must be a pathlib.Path.")
    if profile.balanced_mode not in VALID_BALANCED_MODES:
        raise ValueError(
            f"profile {profile_name!r}: balanced_mode must be one of {', '.join(sorted(VALID_BALANCED_MODES))}."
        )
    if profile.paper_tail_weighting not in VALID_PAPER_TAIL_WEIGHTINGS:
        raise ValueError(
            f"profile {profile_name!r}: paper_tail_weighting must be value_weighted or equal_weighted."
        )
    if not isinstance(profile.refresh_paper_tail, bool):
        raise TypeError(f"profile {profile_name!r}: refresh_paper_tail must be bool.")
    if not isinstance(profile.strict_final_export, bool):
        raise TypeError(f"profile {profile_name!r}: strict_final_export must be bool.")
    if profile.years is not None:
        if len(profile.years) == 0:
            raise ValueError(f"profile {profile_name!r}: years cannot be empty.")
        if any(not isinstance(year, int) for year in profile.years):
            raise TypeError(f"profile {profile_name!r}: years must contain only ints.")
    _validate_positive_optional_int("max_stocks", profile.max_stocks, profile_name)
    _validate_positive_optional_int("workers", profile.workers, profile_name)
    _validate_positive_optional_int("paper_workers", profile.paper_workers, profile_name)
    _validate_positive_optional_int("rolling_workers", profile.rolling_workers, profile_name)
    if profile.memory_budget_gb is not None and float(profile.memory_budget_gb) <= 0:
        raise ValueError(f"profile {profile_name!r}: memory_budget_gb must be greater than 0.")


def get_active_main_profile() -> Tuple[str, MainLaunchProfile]:
    profile_name = ACTIVE_MAIN_PROFILE
    profile = get_main_profile(profile_name)
    validate_main_profile(profile_name, profile)
    return profile_name, profile


def profile_to_run_config(profile: MainLaunchProfile, *, save_plots: bool) -> RunConfig:
    return RunConfig(
        proc_root=profile.proc_root,
        runtime_root=profile.runtime_root,
        final_result_root=profile.final_result_root,
        output_root=profile.final_result_root,
        years=list(profile.years) if profile.years is not None else None,
        max_stocks=profile.max_stocks,
        balanced_mode=profile.balanced_mode,
        return_mode=profile.return_mode,
        jump_a=profile.jump_a,
        k_max=profile.k_max,
        gamma=profile.gamma,
        g_fn=profile.g_fn,
        workers=profile.workers,
        paper_workers=profile.paper_workers,
        rolling_workers=profile.rolling_workers,
        memory_budget_gb=profile.memory_budget_gb,
        progress_interval_sec=profile.heartbeat_sec,
        external_data_root=profile.external_data_root,
        paper_tail_root=profile.paper_tail_root,
        paper_tail_weighting=profile.paper_tail_weighting,
        refresh_paper_tail=profile.refresh_paper_tail,
        strict_final_export=profile.strict_final_export,
        industry_factors_frozen=list(profile.industry_factors_frozen) if profile.industry_factors_frozen else None,
        industry_info_filename=profile.industry_info_filename,
        industry_mapping_filename=profile.industry_mapping_filename,
        save_plots=save_plots,
        restart=bool(profile.restart and profile.rebuild_result),
    )


def clone_main_profile(profile_name: str, **updates: Any) -> MainLaunchProfile:
    profile = get_main_profile(profile_name)
    return replace(profile, **updates)
