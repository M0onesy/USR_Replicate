"""
core/config.py
==============

集中存放"一次复现运行"的可调参数，供 main.py 与各图表脚本共享。

为什么需要它：
  Option A 的核心是"上游昂贵计算只跑一次"。无论是 main.py 一键全量，还是单独
  调试某一张图，都必须使用 **完全相同** 的参数去构建 ReplicationResult，否则缓存
  命中判断会失真、结果也会不一致。把参数收口到这里，避免散落在各处。

用法：
  from core.config import RunConfig
  cfg = RunConfig()                 # 默认全量
  cfg = RunConfig(years=[2016], max_stocks=10, save_plots=False)  # smoke run

  这些字段与 engine.run_cn_replication 的关键参数一一对应。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.engine import DEFAULT_PROC_ROOT, DEFAULT_OUTPUT_ROOT


@dataclass
class RunConfig:
    """单次复现运行的全部参数（与 run_cn_replication 对齐）。"""

    # ---- 路径 ----
    proc_root: Path = field(default_factory=lambda: Path(DEFAULT_PROC_ROOT))
    output_root: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_ROOT))

    # ---- 样本选择 ----
    years: Optional[List[int]] = None          # None = 全部年份
    max_stocks: Optional[int] = None           # None = 全部股票；小样本调试可设 10

    # ---- 论文方法参数（影响数值，必须在缓存键里区分）----
    return_mode: str = "open_close"
    jump_a: float = 3.0
    k_max: int = 10
    gamma: float = 0.08
    g_fn: str = "median_N"

    # ---- 并行 / 资源 ----
    workers: Optional[int] = None
    paper_workers: Optional[int] = None
    rolling_workers: Optional[int] = None
    memory_budget_gb: Optional[float] = None
    progress_interval_sec: float = 10.0

    # ---- 运行行为 ----
    save_plots: bool = True       # 全量导出时是否让 engine 顺带画图（拆分后通常关掉，交给 figcode）
    restart: bool = False         # 丢弃兼容 checkpoint 重跑

    def to_kwargs(self) -> Dict[str, Any]:
        """转换成 run_cn_replication 接受的关键字参数。"""
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
        """决定缓存是否可复用的"数值相关"参数子集。

        只包含会改变 ReplicationResult 内容的字段；并行 / 进度间隔等纯性能参数不计入。
        """
        return {
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
        d = asdict(self)
        d["proc_root"] = str(self.proc_root)
        d["output_root"] = str(self.output_root)
        return d
