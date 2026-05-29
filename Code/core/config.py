from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        d = asdict(self)
        d["proc_root"] = str(self.proc_root)
        d["output_root"] = str(self.output_root)
        return d
