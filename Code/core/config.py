from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.engine import DEFAULT_OUTPUT_ROOT, DEFAULT_PROC_ROOT


CACHE_SCHEMA_VERSION = 3


@dataclass
class RunConfig:
    proc_root: Path = field(default_factory=lambda: Path(DEFAULT_PROC_ROOT))
    output_root: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_ROOT))

    years: Optional[List[int]] = None
    max_stocks: Optional[int] = None

    return_mode: str = "open_close"
    jump_a: float = 3.0
    k_max: int = 10
    gamma: float = 0.08
    g_fn: str = "median_N"

    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None
    progress_interval_sec: float = 10.0

    save_plots: bool = True
    restart: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "proc_root": str(self.proc_root),
            "output_root": str(self.output_root),
            "years": self.years,
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
            "restart": self.restart,
        }

    def cache_signature(self) -> Dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "proc_root": str(Path(self.proc_root).resolve()),
            "years": tuple(self.years) if self.years is not None else None,
            "max_stocks": self.max_stocks,
            "return_mode": self.return_mode,
            "jump_a": self.jump_a,
            "k_max": self.k_max,
            "gamma": self.gamma,
            "g_fn": self.g_fn,
        }

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["proc_root"] = str(self.proc_root)
        data["output_root"] = str(self.output_root)
        return data


@dataclass(frozen=True)
class MainLaunchProfile:
    # 当前 main.py 要运行哪些任务。支持 all / figures / tables / fig13 / table_i 等选择器。
    task_selectors: Tuple[str, ...] = ("all",)
    # 若为 True，则只打印任务列表并退出，不执行任何图表/表格导出。
    list_tasks_only: bool = False
    # 是否显式重建上游 ReplicationResult。False 时优先复用已有结果。
    rebuild_result: bool = False
    # 是否在重建前清理兼容 checkpoint。仅当 rebuild_result=True 时允许开启。
    restart: bool = False
    # 任一任务失败后是否立刻中止整个 main 流程。
    fail_fast: bool = False
    # 是否启用控制台心跳，便于在 PyCharm 中观察长任务进度。
    enable_heartbeat: bool = True
    # 心跳输出间隔，单位秒。
    heartbeat_sec: float = 10.0

    # 以下字段会转换为 RunConfig，决定 pipeline / 导出层看到的运行参数。
    proc_root: Path = field(default_factory=lambda: Path(DEFAULT_PROC_ROOT))
    output_root: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_ROOT))
    years: Optional[Tuple[int, ...]] = None
    max_stocks: Optional[int] = None
    return_mode: str = "open_close"
    jump_a: float = 3.0
    k_max: int = 10
    gamma: float = 0.08
    g_fn: str = "median_N"
    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None


MAIN_RUN_PROFILES: Dict[str, MainLaunchProfile] = {
    "export_all": MainLaunchProfile(
        # 默认正式入口：优先复用已有 ReplicationResult，重导全部图表和表格。
        task_selectors=("all",),
        rebuild_result=False,
        restart=False,
        fail_fast=False,
        enable_heartbeat=True,
        heartbeat_sec=10.0,
    ),
    "figures_only": MainLaunchProfile(
        # 只重导全部图，适合检查图是否正常刷新。
        task_selectors=("figures",),
        rebuild_result=False,
        restart=False,
        fail_fast=False,
        enable_heartbeat=True,
        heartbeat_sec=10.0,
    ),
    "tables_only": MainLaunchProfile(
        # 只重导全部表，适合论文附表核对。
        task_selectors=("tables",),
        rebuild_result=False,
        restart=False,
        fail_fast=False,
        enable_heartbeat=True,
        heartbeat_sec=10.0,
    ),
    "fig13_only": MainLaunchProfile(
        # 单图调试预设：快速检查 Figure 13 是否正常。
        task_selectors=("fig13",),
        rebuild_result=False,
        restart=False,
        fail_fast=True,
        enable_heartbeat=True,
        heartbeat_sec=5.0,
    ),
    "rebuild_all": MainLaunchProfile(
        # 显式全量重建入口：重跑上游 pipeline 后再导出全部结果。
        task_selectors=("all",),
        rebuild_result=True,
        restart=True,
        fail_fast=False,
        enable_heartbeat=True,
        heartbeat_sec=10.0,
    ),
}


# 在 PyCharm 直接运行 main.py 时，这里就是唯一生效的入口开关。
ACTIVE_MAIN_PROFILE = "export_all"


def available_main_profile_names() -> List[str]:
    return sorted(MAIN_RUN_PROFILES.keys())


def get_main_profile(profile_name: str) -> MainLaunchProfile:
    try:
        return MAIN_RUN_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"未找到 main 启动配置 {profile_name!r}。可用 profile：{', '.join(available_main_profile_names())}"
        ) from exc


def validate_main_profile(profile_name: str, profile: MainLaunchProfile) -> None:
    if not isinstance(profile, MainLaunchProfile):
        raise TypeError(f"profile {profile_name!r} 不是 MainLaunchProfile 实例。")

    if not profile.list_tasks_only and not profile.task_selectors:
        raise ValueError(f"profile {profile_name!r} 的 task_selectors 不能为空。")

    if profile.restart and not profile.rebuild_result:
        raise ValueError(
            f"profile {profile_name!r} 配置非法：restart=True 只能与 rebuild_result=True 一起使用。"
        )

    if profile.heartbeat_sec <= 0:
        raise ValueError(f"profile {profile_name!r} 的 heartbeat_sec 必须大于 0。")

    if not isinstance(profile.proc_root, Path):
        raise TypeError(f"profile {profile_name!r} 的 proc_root 必须是 pathlib.Path。")

    if not isinstance(profile.output_root, Path):
        raise TypeError(f"profile {profile_name!r} 的 output_root 必须是 pathlib.Path。")

    if profile.years is not None:
        if len(profile.years) == 0:
            raise ValueError(f"profile {profile_name!r} 的 years 不能为空元组。")
        for year in profile.years:
            if not isinstance(year, int):
                raise TypeError(f"profile {profile_name!r} 的 years 必须全部是 int。")

    if profile.max_stocks is not None and profile.max_stocks <= 0:
        raise ValueError(f"profile {profile_name!r} 的 max_stocks 必须大于 0。")

    if profile.workers is not None and profile.workers <= 0:
        raise ValueError(f"profile {profile_name!r} 的 workers 必须大于 0。")

    if profile.paper_workers is not None and profile.paper_workers <= 0:
        raise ValueError(f"profile {profile_name!r} 的 paper_workers 必须大于 0。")

    if profile.rolling_workers is not None and profile.rolling_workers <= 0:
        raise ValueError(f"profile {profile_name!r} 的 rolling_workers 必须大于 0。")

    if profile.memory_budget_gb is not None and profile.memory_budget_gb <= 0:
        raise ValueError(f"profile {profile_name!r} 的 memory_budget_gb 必须大于 0。")


def get_active_main_profile() -> Tuple[str, MainLaunchProfile]:
    profile_name = ACTIVE_MAIN_PROFILE
    profile = get_main_profile(profile_name)
    validate_main_profile(profile_name, profile)
    return profile_name, profile


def profile_to_run_config(profile: MainLaunchProfile, *, save_plots: bool) -> RunConfig:
    return RunConfig(
        proc_root=Path(profile.proc_root),
        output_root=Path(profile.output_root),
        years=list(profile.years) if profile.years is not None else None,
        max_stocks=profile.max_stocks,
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
        save_plots=save_plots,
        restart=bool(profile.restart and profile.rebuild_result),
    )


def clone_main_profile(profile_name: str, **updates: Any) -> MainLaunchProfile:
    profile = get_main_profile(profile_name)
    return replace(profile, **updates)
