from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.engine import DEFAULT_OUTPUT_ROOT, DEFAULT_PROC_ROOT


CACHE_SCHEMA_VERSION = 3
DEFAULT_EXTERNAL_DATA_ROOT = Path(DEFAULT_PROC_ROOT).parents[1] / "external_Data" / "pelger_tail"
DEFAULT_PAPER_TAIL_ROOT = Path(DEFAULT_PROC_ROOT) / "paper_tail"


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
    # P1 修复：默认改为论文 footnote 17 的口径 g(N,M)=√N·median{λ}（"median_sqrtN"）。
    #  K̂(γ)=max{k: ER_k>1+γ}，临界值 1.08。
    # 注意：cache_signature 含 g_fn，故此默认值【不再命中】旧重结果缓存
    #  replication_result_2ae5dce6a23fd2a2.pkl —— 需重跑（你已计划补数据重跑）。
    #  若确需复用旧缓存做对照，可显式传 g_fn="median_N"（= N·median，非论文口径）。
    g_fn: str = "median_sqrtN"

    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None
    progress_interval_sec: float = 10.0
    external_data_root: Path = field(default_factory=lambda: Path(DEFAULT_EXTERNAL_DATA_ROOT))
    paper_tail_root: Path = field(default_factory=lambda: Path(DEFAULT_PAPER_TAIL_ROOT))
    paper_tail_weighting: str = "value_weighted"
    refresh_paper_tail: bool = True

    # ------------------------------------------------------------------
    # 论文保真开关（视图/尾部层，复用路径即可生效；不进 cache_signature，
    # 故改动这些不会让重结果缓存失效）。详见 docs/SPEC_PAPER_FAITHFUL.md。
    # P4：对 4 个连续 PCA display 因子做确定性符号定向（因子1=正等权市场）。
    paper_faithful_signs: bool = True
    # P5/D1：事先冻结的行业因子桶（std_industry 名称，市场恒为等权全市场）。
    #   None = 暂用“看修好后 CN Figure 4 自动挑 3 个最集中桶”的占位逻辑；
    #   跑完一次后把最终 3 个桶写死在这里即冻结（决策 D1）。
    industry_factors_frozen: Optional[List[str]] = None
    # P6/P7/D4：自建日频 Carhart 12-1 月 MOM（"carhart_daily"）vs 旧高频 1 日动量。
    ffc_mom_mode: str = "carhart_daily"
    # P9/D5：2×3 用全市场 symbol_returns 重建（True）vs 旧平衡子集（False）。
    size_value_full_market: bool = True
    # N6：年化交易日数（论文 252；A 股实测约 243，可按需切换）。
    annualization_days: int = 252
    # P10/D5：size/value（及分段 FFC）样本起点；2012 账面缺失 -> 固定 2014-07-01。
    size_value_start: str = "2014-07-01"
    # 新版 11 桶行业映射文件名（放在 external_data/.../industry/ 下）。
    industry_info_filename: str = "stock_full_info_with_std_industry.csv"

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
            "external_data_root": str(self.external_data_root),
            "paper_tail_root": str(self.paper_tail_root),
            "paper_tail_weighting": self.paper_tail_weighting,
            "refresh_paper_tail": self.refresh_paper_tail,
            "restart": self.restart,
        }

    def export_fidelity_env(self) -> None:
        """把视图层论文保真开关写入 env，供 engine / paper_tail 读取。
        在每次构建/刷新 ReplicationResult 前调用（见 pipeline_cache）。
        这些 env 都【不】进 cache_signature，故不会让重结果缓存失效。"""
        import os as _os

        _os.environ["PELGER_PAPER_FAITHFUL_SIGNS"] = "1" if self.paper_faithful_signs else "0"
        _os.environ["PELGER_FFC_MOM_MODE"] = str(self.ffc_mom_mode)
        _os.environ["PELGER_SIZE_VALUE_FULL_MARKET"] = "1" if self.size_value_full_market else "0"
        _os.environ["PELGER_ANNUALIZATION_DAYS"] = str(int(self.annualization_days))
        _os.environ["PELGER_SIZE_VALUE_START"] = str(self.size_value_start or "")
        _os.environ["PELGER_INDUSTRY_INFO_FILENAME"] = str(self.industry_info_filename)
        if self.industry_factors_frozen:
            _os.environ["PELGER_INDUSTRY_FROZEN"] = ",".join(self.industry_factors_frozen)
        else:
            _os.environ.pop("PELGER_INDUSTRY_FROZEN", None)

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
        data["external_data_root"] = str(self.external_data_root)
        data["paper_tail_root"] = str(self.paper_tail_root)
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
    g_fn: str = "median_sqrtN"  # P1：论文 footnote 17 口径（√N·median）
    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None
    external_data_root: Path = field(default_factory=lambda: Path(DEFAULT_EXTERNAL_DATA_ROOT))
    paper_tail_root: Path = field(default_factory=lambda: Path(DEFAULT_PAPER_TAIL_ROOT))
    paper_tail_weighting: str = "value_weighted"
    refresh_paper_tail: bool = True


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

    if not isinstance(profile.external_data_root, Path):
        raise TypeError(f"profile {profile_name!r} 的 external_data_root 必须是 pathlib.Path。")

    if not isinstance(profile.paper_tail_root, Path):
        raise TypeError(f"profile {profile_name!r} 的 paper_tail_root 必须是 pathlib.Path。")

    if profile.paper_tail_weighting not in {"value_weighted", "equal_weighted"}:
        raise ValueError(
            f"profile {profile_name!r} 的 paper_tail_weighting 必须是 value_weighted 或 equal_weighted。"
        )

    if not isinstance(profile.refresh_paper_tail, bool):
        raise TypeError(f"profile {profile_name!r} 的 refresh_paper_tail 必须是 bool。")

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
        external_data_root=Path(profile.external_data_root),
        paper_tail_root=Path(profile.paper_tail_root),
        paper_tail_weighting=profile.paper_tail_weighting,
        refresh_paper_tail=profile.refresh_paper_tail,
        save_plots=save_plots,
        restart=bool(profile.restart and profile.rebuild_result),
    )


def clone_main_profile(profile_name: str, **updates: Any) -> MainLaunchProfile:
    profile = get_main_profile(profile_name)
    return replace(profile, **updates)
