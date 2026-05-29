"""
core/engine.py
==============

Pelger (2020) 高频系统性风险复现项目 —— 共享计算引擎（"重活"全部在这里）。

本模块由原始单文件 ``allcode_Need.py`` 拆分而来，函数 / 类的实现保持 **逐字节不变**，
因此数值结果与原脚本完全一致。这里只做"搬家"，不改任何数学。

职责：
  - 载入预处理面板（proc_Data）
  - 跳跃分解（continuous / jump）
  - PCA 因子提取、因子个数估计（扰动特征值比）
  - 滚动 PCA、广义相关性、解释度
  - 年度论文表（Table I / II 及因子计数诊断）
  - 把所有结果汇总进 ``ReplicationResult``（一次算清，供图表脚本复用）
  - 原始的整体导出函数（``export_replication_outputs`` / ``export_all_paper_figures``）
    保留下来，方便需要"一键全量复现"时直接调用。

设计要点（与拆分方案 Option A 对应）：
  ``run_cn_replication()`` 把昂贵的上游计算只跑一次，产出一个 ``ReplicationResult``
  对象；figcode / tablecode 下的每篇脚本都只是从这个对象里取字段、画图或落表，
  因此真正耗时的部分不会被重复执行。缓存层见 ``core/pipeline_cache.py``。

注意：
  - 本文件不读取原始 K 线 ``data.bz2``，只消费 ``Data/proc_Data``。
  - 不要在本文件里写 ``argparse`` / ``main``，命令行入口在 ``main.py``。
"""

from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import shutil
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
PANEL_RETURN_SCHEME = "daily_intra_night_total_plus_full_5min_v1"

# 拆分后本文件位于 Code/core/engine.py，比原始 Code/allcode_Need.py 深一层，
# 因此仓库根目录需要回退两级（core -> Code -> Reposit）。
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROC_ROOT = REPO_ROOT / "Data" / "proc_Data" / "pelger_cn_adjusted"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Result" / "pelger_cn_adjusted"
BLAS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
DEFAULT_MEMORY_BUDGET_RATIO = 0.65
DEFAULT_PROGRESS_INTERVAL_SEC = 10.0
PROGRESS_HEARTBEAT_EVENT = "heartbeat"
CHECKPOINT_FORMAT_VERSION = 1
ROLLING_CHECKPOINT_WINDOW_COUNT = 64
PAPER_PANEL_THREAD_CAP = 4
PAPER_MEMORY_SAFETY_MULTIPLIER = 1.35
DISPLAY_CONTINUOUS_FACTOR_COUNT = 4


def _default_worker_count() -> int:
    """复现阶段默认保守并行度，避免和 NumPy/BLAS 内部线程过度竞争。"""
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def _normalize_worker_count(workers: Optional[int]) -> int:
    if workers is None:
        return _default_worker_count()
    return max(1, int(workers))


def _resolve_stage_workers(
    workers: Optional[int],
    stage_workers: Optional[int],
) -> int:
    if stage_workers is not None:
        return _normalize_worker_count(stage_workers)
    return _normalize_worker_count(workers)


@contextlib.contextmanager
def _temporary_blas_thread_env(num_threads: int = 1):
    previous = {key: os.environ.get(key) for key in BLAS_THREAD_ENV_KEYS}
    value = str(max(1, int(num_threads)))
    try:
        for key in BLAS_THREAD_ENV_KEYS:
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _spawn_process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max(1, int(max_workers)),
        mp_context=get_context("spawn"),
    )


def _spawn_pool_launch_supported() -> Tuple[bool, str]:
    if os.name != "nt":
        return True, ""
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if not main_file:
        return False, "Windows spawn multiprocessing requires launching from a real .py file"
    main_file_text = str(main_file)
    if main_file_text in {"<stdin>", "<string>"}:
        return False, f"Windows spawn multiprocessing does not support __main__={main_file_text}"
    if getattr(sys, "ps1", None) is not None:
        return False, "interactive console detected"
    if "ipykernel" in sys.modules:
        return False, "IPython kernel detected"
    return True, ""


def _warn_parallel_fallback(stage_name: str, reason: str) -> None:
    print(f"[WARN] {stage_name}: falling back to single-process execution ({reason}).")


def _safe_process_pool_worker_count(stage_name: str, workers: int) -> int:
    workers = max(1, int(workers))
    if workers <= 1:
        return 1
    supported, reason = _spawn_pool_launch_supported()
    if not supported:
        _warn_parallel_fallback(stage_name, reason)
        return 1
    return workers


def _is_spawn_context_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return (
        isinstance(exc, BrokenProcessPool)
        or "<stdin>" in text
        or "<string>" in text
        or "freeze_support" in text
        or "bootstrapping phase" in text
        or "Can't get attribute" in text
    )


def _chunk_sequence(values: Sequence[int], n_chunks: int) -> List[List[int]]:
    if not values:
        return []
    chunk_count = max(1, min(int(n_chunks), len(values)))
    chunk_size = (len(values) + chunk_count - 1) // chunk_count
    return [list(values[i : i + chunk_size]) for i in range(0, len(values), chunk_size)]


def _physical_memory_bytes() -> int:
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            pass

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and page_count > 0:
                return page_size * page_count
        except Exception:
            pass
    return int(8 * 1024 ** 3)


def _format_bytes_gb(value: float) -> float:
    return float(value) / float(1024 ** 3)


def _append_progress_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_ready(record), ensure_ascii=False) + "\n")


def _console_line_for_progress_record(record: Dict[str, Any]) -> str:
    event = record.get("event", "")
    stage = record.get("stage", "")
    year = record.get("year")
    elapsed = float(record.get("elapsed_sec", 0.0))
    prefix = f"[{elapsed:8.1f}s]"
    if event == "rolling_chunk_finished":
        completed = record.get("completed_chunks")
        total = record.get("total_chunks")
        if completed is None or total is None:
            return ""
        completed_int = int(completed)
        total_int = max(1, int(total))
        if completed_int != total_int and completed_int % max(1, total_int // 10) != 0:
            return ""
        parts = [prefix, "[rolling_chunk_finished]"]
        if stage:
            parts.append(f"stage={stage}")
        parts.append(f"chunks={completed_int}/{total_int}")
        return " ".join(parts)
    if event == PROGRESS_HEARTBEAT_EVENT:
        parts = [prefix, "[HEARTBEAT]"]
        if stage:
            parts.append(f"stage={stage}")
        if record.get("completed_years") is not None and record.get("total_years") is not None:
            parts.append(f"years={record.get('completed_years')}/{record.get('total_years')}")
        if record.get("active_years") is not None:
            parts.append(f"active={record.get('active_years')}")
        if record.get("paper_workers_effective") is not None:
            parts.append(f"workers={record.get('paper_workers_effective')}")
        if record.get("memory_reserved_gb") is not None and record.get("memory_budget_gb") is not None:
            parts.append(
                f"mem={float(record.get('memory_reserved_gb')):.2f}/{float(record.get('memory_budget_gb')):.2f}GB"
            )
        return " ".join(parts)

    parts = [prefix, f"[{event}]"]
    if stage:
        parts.append(f"stage={stage}")
    if year is not None:
        parts.append(f"year={year}")
    if record.get("panel_block"):
        parts.append(f"panel={record['panel_block']}")
    if record.get("component"):
        parts.append(f"component={record['component']}")
    if record.get("threshold_a") is not None:
        parts.append(f"a={float(record['threshold_a']):.1f}")
    if record.get("completed_chunks") is not None and record.get("total_chunks") is not None:
        parts.append(f"chunks={record['completed_chunks']}/{record['total_chunks']}")
    if record.get("message"):
        parts.append(str(record["message"]))
    return " ".join(parts)


def _emit_progress_event_to_path(
    progress_path: Optional[str | Path],
    event_type: str,
    *,
    run_started_unix: Optional[float] = None,
    state: Optional[Dict[str, Any]] = None,
    **payload: Any,
) -> None:
    if not progress_path:
        return
    path = Path(progress_path)
    now_unix = time.time()
    elapsed_sec = max(0.0, now_unix - float(run_started_unix)) if run_started_unix is not None else 0.0
    record: Dict[str, Any] = {
        "event": str(event_type),
        "elapsed_sec": float(elapsed_sec),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    if state:
        record.update(state)
    record.update(payload)
    _append_progress_record(path, record)
    line = _console_line_for_progress_record(record)
    if line:
        print(line, flush=True)


def _estimate_unbalanced_year_peak_bytes(day_count: int, stock_count: int) -> int:
    rows_5min = int(day_count) * len(CN_5MIN_BAR_TIMES)
    panel_bytes = rows_5min * int(stock_count) * 8
    daily_bytes = int(day_count) * int(stock_count) * 8 * 3
    workspace_bytes = int(panel_bytes * 4.5)
    return int(panel_bytes + daily_bytes + workspace_bytes)


def _estimate_balanced_year_peak_bytes(day_count: int, stock_count: int) -> int:
    rows_5min = int(day_count) * len(CN_5MIN_BAR_TIMES)
    panel_bytes = rows_5min * int(stock_count) * 8
    daily_bytes = int(day_count) * int(stock_count) * 8 * 3
    return int(panel_bytes + daily_bytes)


@dataclass
class ProgressReporter:
    diagnostics_dir: Path
    interval_sec: float = DEFAULT_PROGRESS_INTERVAL_SEC
    reset_existing: bool = True
    _state: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _start_time: float = field(default_factory=time.perf_counter)
    _start_time_unix: float = field(default_factory=time.time)
    _last_heartbeat: float = field(default_factory=lambda: 0.0)

    def __post_init__(self) -> None:
        self.interval_sec = max(1.0, float(self.interval_sec))
        self.path = self.diagnostics_dir / "progress.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.reset_existing and self.path.exists():
            self.path.unlink()

    def update_state(self, **kwargs: Any) -> None:
        with self._lock:
            self._state.update(kwargs)

    def snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def event(self, event_type: str, **payload: Any) -> None:
        now = time.perf_counter()
        with self._lock:
            base = {
                "event": str(event_type),
                "elapsed_sec": float(now - self._start_time),
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
            merged = dict(self._state)
            merged.update(payload)
            record = {**base, **merged}
            _append_progress_record(self.path, record)

            if event_type == PROGRESS_HEARTBEAT_EVENT:
                self._last_heartbeat = now

            line = _console_line_for_progress_record(record)
            if line:
                print(line, flush=True)

    def heartbeat(self, force: bool = False, **payload: Any) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_heartbeat) < self.interval_sec:
            return
        self.event(PROGRESS_HEARTBEAT_EVENT, **payload)

@dataclass
class RuntimeConfig:
    memory_budget_bytes: int
    memory_budget_gb: float
    progress_interval_sec: float
    physical_memory_bytes: int
    logical_cpus: int
    scratch_root: Path


@dataclass
class ResourcePlan:
    physical_memory_bytes: int
    physical_memory_gb: float
    memory_budget_bytes: int
    memory_budget_gb: float
    logical_cpus: int
    requested_workers: int
    requested_paper_workers: int
    requested_rolling_workers: int
    paper_workers_effective: int
    rolling_workers_effective: int
    paper_blas_threads: int
    execution_mode: str
    paper_schedule_policy: str
    paper_memory_cap_workers: int
    paper_cpu_cap_workers: int
    paper_memory_safety_multiplier: float
    paper_year_estimates: List[Dict[str, Any]]


@dataclass
class CheckpointLayout:
    root: Path
    rolling_dir: Path
    paper_dir: Path
    run_state_path: Path


@dataclass
class CheckpointManager:
    output_root: Path
    signature: Dict[str, Any]
    restart: bool = False
    layout: CheckpointLayout = field(init=False)
    state: Dict[str, Any] = field(init=False)
    resumed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        checkpoint_root = self.output_root / "checkpoints"
        self.layout = CheckpointLayout(
            root=checkpoint_root,
            rolling_dir=checkpoint_root / "rolling",
            paper_dir=checkpoint_root / "paper",
            run_state_path=checkpoint_root / "run_state.json",
        )
        self.layout.root.mkdir(parents=True, exist_ok=True)
        self.layout.rolling_dir.mkdir(parents=True, exist_ok=True)
        self.layout.paper_dir.mkdir(parents=True, exist_ok=True)
        self.state = {}

    def prepare(self) -> Dict[str, Any]:
        cleaned_items: List[str] = []
        if self.restart:
            if self.layout.root.exists():
                shutil.rmtree(self.layout.root, ignore_errors=True)
            self.layout.root.mkdir(parents=True, exist_ok=True)
            self.layout.rolling_dir.mkdir(parents=True, exist_ok=True)
            self.layout.paper_dir.mkdir(parents=True, exist_ok=True)
            cleaned_items.extend(_cleanup_export_tmp_files(self.output_root))
            self.state = _new_run_state(self.signature)
            self.save()
            return {"resumed": False, "cleaned_items": cleaned_items, "reason": "restart"}

        loaded_state = _load_json_if_exists(self.layout.run_state_path)
        if loaded_state is not None:
            loaded_signature = loaded_state.get("signature", {})
            if loaded_signature != self.signature:
                raise ValueError(
                    "Existing checkpoint state is not compatible with the current semantic run signature. "
                    "Use a different --output-root or pass --restart."
                )
            self.state = dict(loaded_state)
            self.resumed = True
        else:
            self.state = _new_run_state(self.signature)

        cleaned_items.extend(_cleanup_export_tmp_files(self.output_root))
        cleaned_items.extend(self._cleanup_incomplete_checkpoint_units())
        self.state["status"] = "running"
        self.state["last_updated"] = pd.Timestamp.utcnow().isoformat()
        self.state["resumed_from_previous"] = bool(self.resumed)
        self.save()
        return {"resumed": bool(self.resumed), "cleaned_items": cleaned_items, "reason": "resume" if self.resumed else "new_run"}

    def save(self) -> None:
        self.state["checkpoint_format_version"] = int(CHECKPOINT_FORMAT_VERSION)
        self.state["last_updated"] = pd.Timestamp.utcnow().isoformat()
        _write_json(self.layout.run_state_path, self.state)

    def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        self.save()

    def rolling_chunk_path(self, chunk_index: int) -> Path:
        return self.layout.rolling_dir / f"chunk_{int(chunk_index):05d}.npz"

    def paper_year_dir(self, year: int) -> Path:
        return self.layout.paper_dir / f"year_{int(year)}"

    def completed_rolling_chunks(self) -> Set[int]:
        out: Set[int] = set()
        for path in self.layout.rolling_dir.glob("chunk_*.npz"):
            try:
                out.add(int(path.stem.split("_")[-1]))
            except Exception:
                continue
        return out

    def completed_paper_years(self) -> Set[int]:
        out: Set[int] = set()
        for path in self.layout.paper_dir.glob("year_*"):
            complete_path = path / "complete.json"
            if not complete_path.exists():
                continue
            try:
                out.add(int(path.name.split("_")[-1]))
            except Exception:
                continue
        return out

    def mark_rolling_plan(self, total_chunks: int) -> None:
        self.state["rolling_total_chunks"] = int(total_chunks)
        self.state["completed_rolling_chunks"] = sorted(self.completed_rolling_chunks())
        self.save()

    def mark_rolling_chunk_complete(self, chunk_index: int) -> None:
        completed = self.completed_rolling_chunks()
        completed.add(int(chunk_index))
        self.state["completed_rolling_chunks"] = sorted(completed)
        self.save()

    def mark_paper_plan(self, years: Sequence[int]) -> None:
        self.state["paper_total_years"] = int(len(years))
        self.state["completed_paper_years"] = sorted(self.completed_paper_years())
        self.save()

    def mark_paper_year_complete(self, year: int) -> None:
        completed = self.completed_paper_years()
        completed.add(int(year))
        self.state["completed_paper_years"] = sorted(completed)
        self.save()

    def mark_export_complete(self) -> None:
        self.state["export_completed"] = True
        self.state["status"] = "finished"
        self.state["stage"] = "done"
        self.save()

    def mark_failed(self, stage: str) -> None:
        self.state["status"] = "failed"
        self.state["stage"] = str(stage)
        self.save()

    def _cleanup_incomplete_checkpoint_units(self) -> List[str]:
        cleaned_items: List[str] = []
        for tmp_path in self.layout.root.rglob("*.tmp"):
            try:
                tmp_path.unlink()
                cleaned_items.append(str(tmp_path))
            except Exception:
                pass
        for tmp_dir in self.layout.root.rglob("tmp_*"):
            if not tmp_dir.is_dir():
                continue
            shutil.rmtree(tmp_dir, ignore_errors=True)
            cleaned_items.append(str(tmp_dir))
        for year_dir in self.layout.paper_dir.glob("year_*"):
            if not year_dir.is_dir():
                continue
            if (year_dir / "complete.json").exists():
                continue
            shutil.rmtree(year_dir, ignore_errors=True)
            cleaned_items.append(str(year_dir))
        return cleaned_items


@dataclass
class HF5MinPanel:
    R_5min_full: np.ndarray
    day_ids: np.ndarray
    tickers: List[str]
    dates: List[pd.Timestamp]
    bar_times: Optional[List[str]] = None
    sample_mode: str = "custom"
    panel_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.R_5min_full = np.asarray(self.R_5min_full, dtype=float)
        self.day_ids = np.asarray(self.day_ids, dtype=int)
        self.tickers = [str(ticker) for ticker in self.tickers]
        self.dates = [pd.Timestamp(date) for date in self.dates]
        if self.bar_times is not None:
            self.bar_times = list(self.bar_times)
        assert self.R_5min_full.ndim == 2, "R_5min_full must be 2D"
        assert self.day_ids.shape[0] == self.R_5min_full.shape[0], "day_ids length mismatch"
        assert self.R_5min_full.shape[1] == len(self.tickers), "ticker count mismatch"

    @property
    def N(self) -> int:
        return int(self.R_5min_full.shape[1])

    @property
    def D(self) -> int:
        return int(len(self.dates))

    @property
    def M_per_day(self) -> int:
        if self.bar_times:
            return len(self.bar_times)
        return int(self.R_5min_full.shape[0] / max(self.D, 1))


@dataclass
class MemmapContext:
    scratch_root: Optional[Path] = None
    prefix: str = "runtime"
    cleanup: bool = True
    _paths: List[Path] = field(default_factory=list)

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype | str = np.float64,
        fill_value: float = np.nan,
        stem: str = "array",
    ) -> np.ndarray:
        dtype_np = np.dtype(dtype)
        if self.scratch_root is None:
            arr = np.empty(shape, dtype=dtype_np)
            arr.fill(fill_value)
            return arr

        self.scratch_root.mkdir(parents=True, exist_ok=True)
        path = self.scratch_root / f"{self.prefix}_{stem}_{len(self._paths)}.dat"
        arr = np.memmap(path, mode="w+", dtype=dtype_np, shape=shape)
        arr.fill(fill_value)
        self._paths.append(path)
        return arr

    def cleanup_files(self) -> None:
        if not self.cleanup:
            return
        for path in self._paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        self._paths.clear()


def _universe_attr_summary(universe: pd.DataFrame) -> Dict[str, Any]:
    if "summary" not in universe.attrs:
        raise ValueError("Processed universe is missing attrs['summary']; reload it with load_proc_universe().")
    return dict(universe.attrs["summary"])


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
    text = json.dumps(_json_ready(payload), ensure_ascii=False, indent=2)
    _atomic_write_text(path, text)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return _load_json(path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _atomic_copy_file(source_path: Path, target_path: Path) -> None:
    if source_path.resolve() == target_path.resolve():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    shutil.copyfile(source_path, tmp_path)
    tmp_path.replace(target_path)


def _atomic_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=index, encoding=encoding)
    tmp_path.replace(path)


def _atomic_to_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    pd.to_pickle(obj, tmp_path)
    tmp_path.replace(path)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    np.savez_compressed(tmp_path, **arrays)
    actual_tmp = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    actual_tmp.replace(path)


def _atomic_save_figure(fig: Any, path: Path, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp_path, dpi=dpi)
    tmp_path.replace(path)


def _cleanup_export_tmp_files(output_root: Path) -> List[str]:
    cleaned_items: List[str] = []
    for subdir_name in ("tables", "figures", "diagnostics"):
        subdir = output_root / subdir_name
        if not subdir.exists():
            continue
        for tmp_path in subdir.rglob("*.tmp"):
            try:
                tmp_path.unlink()
                cleaned_items.append(str(tmp_path))
            except Exception:
                pass
    return cleaned_items


def _current_code_file_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _build_run_signature(
    proc_root: Path,
    years: Optional[Sequence[int]],
    max_stocks: Optional[int],
    jump_a: float,
    k_max: int,
    gamma: float,
    g_fn: str,
    return_mode: str,
) -> Dict[str, Any]:
    payload = {
        "proc_root": str(proc_root),
        "years": _normalize_years(years),
        "max_stocks": None if max_stocks is None else int(max_stocks),
        "jump_a": float(jump_a),
        "k_max": int(k_max),
        "gamma": float(gamma),
        "g_fn": str(g_fn),
        "return_mode": str(return_mode),
        "checkpoint_format_version": int(CHECKPOINT_FORMAT_VERSION),
        "code_file_sha256": _current_code_file_hash(),
    }
    payload_text = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True)
    return {
        "payload": payload,
        "sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
    }


def _new_run_state(signature: Dict[str, Any]) -> Dict[str, Any]:
    now_text = pd.Timestamp.utcnow().isoformat()
    return {
        "checkpoint_format_version": int(CHECKPOINT_FORMAT_VERSION),
        "signature": signature,
        "status": "initialized",
        "stage": "startup",
        "run_started_at": now_text,
        "last_updated": now_text,
        "resumed_from_previous": False,
        "rolling_total_chunks": 0,
        "completed_rolling_chunks": [],
        "paper_total_years": 0,
        "completed_paper_years": [],
        "export_completed": False,
    }


@dataclass
class HFPanel:
    """
    高频面板数据容器.

    字段:
        R_intra:
            shape (D, N), 日内总收益.
        R_night:
            shape (D, N), 日度隔夜收益.
        R_5min_full:
            shape (M_total, N), 以收盘价链构造的全量 5 分钟对数收益.
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
    R_5min_full: np.ndarray
    tickers: List[str]
    dates: List[pd.Timestamp]
    R_daily: Optional[np.ndarray] = None
    rf_intra: Optional[np.ndarray] = None
    rf_night: Optional[np.ndarray] = None
    bar_times: Optional[List[str]] = None
    sample_report: Optional[Dict[str, Any]] = None
    sample_mode: str = "custom"
    panel_return_scheme: str = PANEL_RETURN_SCHEME
    requested_return_mode: Optional[str] = None

    def __post_init__(self) -> None:
        self.R_intra = np.asarray(self.R_intra, dtype=float)
        self.R_night = np.asarray(self.R_night, dtype=float)
        self.day_ids = np.asarray(self.day_ids, dtype=int)
        self.R_5min_full = np.asarray(self.R_5min_full, dtype=float)
        self.dates = [pd.Timestamp(date) for date in self.dates]
        self.tickers = [str(ticker) for ticker in self.tickers]
        if self.bar_times is not None:
            self.bar_times = list(self.bar_times)

        assert self.R_intra.ndim == 2, "R_intra 必须是 2D"
        assert self.R_night.ndim == 2, "R_night 必须是 2D"
        assert self.R_5min_full.ndim == 2, "R_5min_full 必须是 2D"
        assert self.R_intra.shape[0] == self.R_night.shape[0] == len(self.dates), "日频行数必须等于交易日数"
        assert self.R_intra.shape[1] == self.R_night.shape[1] == len(self.tickers), "股票维度不一致"
        assert self.R_5min_full.shape[1] == len(self.tickers), "R_5min_full 股票维度不一致"
        assert self.day_ids.shape[0] == self.R_5min_full.shape[0], "day_ids 长度错误"

        if self.bar_times is not None and len(self.bar_times) > 0:
            inferred = self.R_5min_full.shape[0] / max(len(self.dates), 1)
            assert int(round(inferred)) == len(self.bar_times), "bar_times 与 R_5min_full 行数不一致"

        if self.R_daily is None:
            self.R_daily = self.R_intra + self.R_night
        else:
            self.R_daily = np.asarray(self.R_daily, dtype=float)
        assert self.R_daily.shape == self.R_intra.shape, "R_daily 形状必须是 (D, N)"
        if not np.allclose(self.R_daily, self.R_intra + self.R_night, equal_nan=True, atol=1e-12):
            raise ValueError("R_daily 必须等于 R_intra + R_night")

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
        return int(self.R_5min_full.shape[0] / max(self.D, 1))


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


def _symbol_npz_path(proc_root: Path, symbol: str) -> Path:
    """单只股票预处理收益缓存的 `.npz` 路径。"""
    return proc_root / "symbol_returns" / f"{symbol}.npz"


def _load_symbol_arrays(proc_root: Path, symbol: str) -> Dict[str, np.ndarray]:
    """读取单只股票的预处理收益缓存。"""
    arrays_raw = np.load(_symbol_npz_path(proc_root, symbol), allow_pickle=False)
    return {name: arrays_raw[name] for name in arrays_raw.files}


def _align_symbol_arrays_to_dates(
    arrays: Dict[str, np.ndarray],
    date_codes: np.ndarray,
    allow_missing: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把单股票收益对齐到目标交易日历。"""
    source_codes = arrays["date_codes"].astype(np.int32, copy=False)
    day_count = int(len(date_codes))
    bar_count = int(arrays["full_5min_returns"].shape[1]) if arrays.get("full_5min_returns") is not None else len(CN_5MIN_BAR_TIMES)
    fill_value = np.nan if allow_missing else 0.0

    daily_intra = np.full(day_count, fill_value, dtype=np.float64)
    daily_night = np.full(day_count, fill_value, dtype=np.float64)
    daily_total = np.full(day_count, fill_value, dtype=np.float64)
    full_5min = np.full(day_count * bar_count, fill_value, dtype=np.float64)

    if len(source_codes) == 0:
        if not allow_missing:
            raise ValueError("目标股票没有任何有效交易日。")
        return daily_intra, daily_night, daily_total, full_5min

    source_pos = np.searchsorted(source_codes, date_codes)
    clipped_pos = source_pos.clip(max=max(len(source_codes) - 1, 0))
    matched = (source_pos < len(source_codes)) & (source_codes[clipped_pos] == date_codes)

    if not bool(matched.all()) and not allow_missing:
        missing_code = int(date_codes[np.where(~matched)[0][0]])
        raise ValueError(f"缺少 {missing_code} 对应的完整收益缓存。")

    if matched.any():
        target_idx = np.where(matched)[0]
        source_idx = source_pos[matched]
        daily_intra[target_idx] = arrays["intraday_returns"][source_idx]
        daily_night[target_idx] = arrays["overnight_returns"][source_idx]
        daily_total[target_idx] = arrays["daily_returns"][source_idx]
        if "full_5min_returns" not in arrays:
            raise KeyError("symbol_returns 缓存缺少 full_5min_returns。")
        full_5min.reshape(day_count, bar_count)[target_idx, :] = arrays["full_5min_returns"][source_idx]
    return daily_intra, daily_night, daily_total, full_5min


def _load_and_align_symbol_task(task: Tuple[Path, str, np.ndarray]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task_proc_root, symbol, codes = task
    arrays = _load_symbol_arrays(task_proc_root, symbol)
    intra, night, daily, full_5min = _align_symbol_arrays_to_dates(arrays, codes, allow_missing=True)
    return symbol, intra, night, daily, full_5min


def _align_symbol_5min_only_to_dates(
    arrays: Dict[str, np.ndarray],
    date_codes: np.ndarray,
) -> np.ndarray:
    source_codes = arrays["date_codes"].astype(np.int32, copy=False)
    day_count = int(len(date_codes))
    bar_count = int(arrays["full_5min_returns"].shape[1]) if arrays.get("full_5min_returns") is not None else len(CN_5MIN_BAR_TIMES)
    full_5min = np.full(day_count * bar_count, np.nan, dtype=np.float64)
    if len(source_codes) == 0:
        return full_5min

    source_pos = np.searchsorted(source_codes, date_codes)
    clipped_pos = source_pos.clip(max=max(len(source_codes) - 1, 0))
    matched = (source_pos < len(source_codes)) & (source_codes[clipped_pos] == date_codes)
    if matched.any():
        target_idx = np.where(matched)[0]
        source_idx = source_pos[matched]
        full_5min.reshape(day_count, bar_count)[target_idx, :] = arrays["full_5min_returns"][source_idx]
    return full_5min


def _load_and_align_symbol_5min_only_task(task: Tuple[Path, str, np.ndarray]) -> Tuple[str, np.ndarray]:
    task_proc_root, symbol, codes = task
    arrays = _load_symbol_arrays(task_proc_root, symbol)
    full_5min = _align_symbol_5min_only_to_dates(arrays, codes)
    return symbol, full_5min


def _year_dates_from_universe_summary(summary: Dict[str, Any], year: int) -> List[str]:
    global_dates_by_year = summary.get("global_dates_by_year", {})
    if isinstance(global_dates_by_year, dict):
        return list(global_dates_by_year.get(int(year), global_dates_by_year.get(str(int(year)), [])))
    return []


def _select_unbalanced_symbols_for_year(
    universe: pd.DataFrame,
    year: int,
    max_stocks: Optional[int] = None,
) -> List[str]:
    year_col = f"valid_days_{int(year)}"
    if year_col not in universe.columns:
        raise ValueError(f"universe 中缺少 {year_col}。请先重新预处理。")
    selected = universe.loc[universe[year_col] > 0, ["symbol"]].copy().sort_values("symbol")
    if max_stocks is not None:
        selected = selected.iloc[: int(max_stocks)].copy()
    return selected["symbol"].tolist()


def _balanced_symbol_count_for_year(proc_root: Path, year: int, max_stocks: Optional[int] = None) -> int:
    _, meta_path = _proc_panel_paths(proc_root, STRICT_BALANCED_SAMPLE, f"year_{int(year)}")
    meta = _load_json(meta_path)
    count = int(len(meta.get("tickers", [])))
    if max_stocks is not None:
        count = min(count, int(max_stocks))
    return count


def _paper_panel_thread_count(logical_cpus: int, active_years: int) -> int:
    cpu_budget = max(1, (max(1, int(logical_cpus) - 2)) // max(1, int(active_years)) // 2)
    return int(max(1, min(PAPER_PANEL_THREAD_CAP, cpu_budget)))


def _estimate_paper_year_peak_bytes(
    day_count: int,
    balanced_stock_count: int,
    unbalanced_stock_count: int,
    safety_multiplier: float = PAPER_MEMORY_SAFETY_MULTIPLIER,
) -> Dict[str, float]:
    rows_5min = int(day_count) * len(CN_5MIN_BAR_TIMES)
    balanced_full_bytes = rows_5min * int(balanced_stock_count) * 8
    unbalanced_full_bytes = rows_5min * int(unbalanced_stock_count) * 8
    retained_threshold3_bytes = rows_5min * (int(balanced_stock_count) + int(unbalanced_stock_count)) * 8 * 2
    high_threshold_temp_bytes = rows_5min * max(int(balanced_stock_count), int(unbalanced_stock_count)) * 8 * 2
    pairwise_n = max(int(balanced_stock_count), int(unbalanced_stock_count))
    pairwise_cross_bytes = pairwise_n * pairwise_n * 8
    pairwise_count_bytes = pairwise_n * pairwise_n * 4
    mask_workspace_bytes = rows_5min * pairwise_n * 2
    raw_peak_bytes = int(
        balanced_full_bytes
        + unbalanced_full_bytes
        + retained_threshold3_bytes
        + high_threshold_temp_bytes
        + pairwise_cross_bytes
        + pairwise_count_bytes
        + mask_workspace_bytes
    )
    safe_peak_bytes = int(max(raw_peak_bytes, 1) * float(safety_multiplier))
    return {
        "peak_memory_raw_bytes": int(raw_peak_bytes),
        "peak_memory_raw_gb": _format_bytes_gb(raw_peak_bytes),
        "peak_memory_bytes": int(safe_peak_bytes),
        "peak_memory_gb": _format_bytes_gb(safe_peak_bytes),
        "safety_multiplier": float(safety_multiplier),
    }


def _build_unbalanced_year_panel_from_dates_and_tickers(
    proc_root: Path,
    year: int,
    year_dates: Sequence[str],
    tickers: Sequence[str],
    panel_workers: Optional[int] = None,
    memmap_ctx: Optional[MemmapContext] = None,
) -> HFPanel:
    tickers = [str(ticker) for ticker in tickers]
    if not year_dates:
        raise ValueError(f"找不到年份 {year} 的交易日历。")
    if not tickers:
        raise ValueError(f"{year} 年没有可用的非平衡样本。")

    date_codes = np.array([int(str(date).replace("-", "")) for date in year_dates], dtype=np.int32)
    fill_value = np.nan
    day_count = len(year_dates)
    bar_count = len(CN_5MIN_BAR_TIMES)
    stock_count = len(tickers)
    ctx = memmap_ctx or MemmapContext(scratch_root=None)
    R_intra = ctx.allocate((day_count, stock_count), dtype=np.float64, fill_value=fill_value, stem=f"{year}_intra")
    R_night = ctx.allocate((day_count, stock_count), dtype=np.float64, fill_value=fill_value, stem=f"{year}_night")
    R_daily = ctx.allocate((day_count, stock_count), dtype=np.float64, fill_value=fill_value, stem=f"{year}_daily")
    R_5min_full = ctx.allocate((day_count * bar_count, stock_count), dtype=np.float64, fill_value=fill_value, stem=f"{year}_full5min")

    panel_workers = _normalize_worker_count(panel_workers)
    tasks = [(proc_root, symbol, date_codes) for symbol in tickers]
    if panel_workers == 1 or len(tasks) <= 1:
        for col_idx, task in enumerate(tasks):
            _, intra, night, daily, full_5min = _load_and_align_symbol_task(task)
            R_intra[:, col_idx] = intra
            R_night[:, col_idx] = night
            R_daily[:, col_idx] = daily
            R_5min_full[:, col_idx] = full_5min
    else:
        with ThreadPoolExecutor(max_workers=panel_workers) as executor:
            futures = {executor.submit(_load_and_align_symbol_task, task): col_idx for col_idx, task in enumerate(tasks)}
            for future in as_completed(futures):
                col_idx = futures[future]
                _, intra, night, daily, full_5min = future.result()
                R_intra[:, col_idx] = intra
                R_night[:, col_idx] = night
                R_daily[:, col_idx] = daily
                R_5min_full[:, col_idx] = full_5min

    day_ids = np.repeat(np.arange(day_count), bar_count).astype(np.int32)
    sample_report = {
        "sample_mode": "unbalanced_yearly",
        "panel_return_scheme": PANEL_RETURN_SCHEME,
        "requested_return_mode": "open_close",
        "panel_name": f"unbalanced_year_{year}",
        "adjustment": "backward",
        "n_symbols_selected": int(stock_count),
        "n_days_selected": int(day_count),
        "bars_per_day": int(bar_count),
        "selected_calendar_start": str(year_dates[0]),
        "selected_calendar_end": str(year_dates[-1]),
        "target_symbols": tickers,
        "contains_nan": bool(np.isnan(R_intra).any() or np.isnan(R_5min_full).any()),
        "panel_workers": int(panel_workers),
    }

    return HFPanel(
        R_intra=R_intra,
        R_night=R_night,
        day_ids=day_ids,
        R_5min_full=R_5min_full,
        tickers=tickers,
        dates=[pd.Timestamp(date) for date in year_dates],
        R_daily=R_daily,
        bar_times=list(CN_5MIN_BAR_TIMES),
        sample_report=sample_report,
        sample_mode="unbalanced_yearly",
        panel_return_scheme=PANEL_RETURN_SCHEME,
        requested_return_mode="open_close",
    )


def _build_unbalanced_year_panel(
    proc_root: Path,
    universe: pd.DataFrame,
    year: int,
    max_stocks: Optional[int] = None,
    workers: Optional[int] = None,
) -> HFPanel:
    """用逐股票缓存重建某一年的非平衡面板。"""
    summary = _universe_attr_summary(universe)
    year_dates = _year_dates_from_universe_summary(summary, year)
    tickers = _select_unbalanced_symbols_for_year(universe, year, max_stocks=max_stocks)
    return _build_unbalanced_year_panel_from_dates_and_tickers(
        proc_root=proc_root,
        year=year,
        year_dates=year_dates,
        tickers=tickers,
        panel_workers=workers,
    )



def summarize_cn_universe(universe: pd.DataFrame) -> Dict[str, Any]:
    """Summarize the already-preprocessed universe table."""
    summary = _universe_attr_summary(universe)
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
    if sample_mode != STRICT_BALANCED_SAMPLE:
        raise ValueError("当前数据政策仅保留 strict_balanced 面板")
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
        R_5min_full=panel.R_5min_full[:, keep],
        tickers=list(panel.tickers[keep]),
        dates=list(panel.dates),
        R_daily=panel.R_daily[:, keep],
        rf_intra=panel.rf_intra.copy(),
        rf_night=panel.rf_night.copy(),
        bar_times=list(panel.bar_times or []),
        sample_report=sample_report,
        sample_mode=panel.sample_mode,
        panel_return_scheme=panel.panel_return_scheme,
        requested_return_mode=panel.requested_return_mode,
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

    tickers = list(meta["tickers"])
    dates = [pd.Timestamp(date) for date in meta["dates"]]
    bar_times = list(meta.get("bar_times", []))
    if "R_5min_full" not in arrays:
        raise ValueError(
            f"旧版面板缺少 R_5min_full: {meta_path}. 请先重新运行 Code/preprocess_cn_data.py --refresh 重建 proc_Data。"
        )
    if arrays["R_intra"].ndim != 2 or arrays["R_night"].ndim != 2 or arrays["R_daily"].ndim != 2:
        raise ValueError(f"检测到旧版面板结构: {meta_path}. 请重新运行 Code/preprocess_cn_data.py --refresh。")
    if arrays["R_intra"].shape != arrays["R_night"].shape or arrays["R_daily"].shape != arrays["R_intra"].shape:
        raise ValueError(f"面板日频收益形状不一致: {meta_path}. 请重新运行 Code/preprocess_cn_data.py --refresh。")
    if arrays["R_intra"].shape[0] != len(dates):
        raise ValueError(f"面板日频行数与交易日数不一致: {meta_path}. 请重新运行 Code/preprocess_cn_data.py --refresh。")
    if arrays["R_5min_full"].shape[1] != len(tickers):
        raise ValueError(f"R_5min_full 列数与股票数不一致: {meta_path}.")
    if bar_times and arrays["R_5min_full"].shape[0] != len(dates) * len(bar_times):
        raise ValueError(f"R_5min_full 行数与日历不一致: {meta_path}.")

    return HFPanel(
        R_intra=arrays["R_intra"],
        R_night=arrays["R_night"],
        day_ids=arrays["day_ids"],
        R_5min_full=arrays["R_5min_full"],
        tickers=tickers,
        dates=dates,
        R_daily=arrays["R_daily"],
        bar_times=bar_times,
        sample_report=dict(meta.get("sample_report", {})),
        sample_mode=meta["sample_mode"],
        panel_return_scheme=meta.get("panel_return_scheme", PANEL_RETURN_SCHEME),
        requested_return_mode=meta.get("requested_return_mode"),
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
    if sample_mode != STRICT_BALANCED_SAMPLE:
        raise ValueError("当前数据政策仅保留 strict_balanced 面板")
    panel_name = _proc_panel_name(years)
    if panel_name is not None:
        try:
            panel = _load_proc_panel_file(proc_root, sample_mode, panel_name)
        except FileNotFoundError:
            panel = subset_panel_by_years(_load_proc_panel_file(proc_root, sample_mode, "full"), _normalize_years(years) or [])
    else:
        panel = subset_panel_by_years(_load_proc_panel_file(proc_root, sample_mode, "full"), _normalize_years(years) or [])

    panel.requested_return_mode = return_mode
    panel.sample_report = dict(panel.sample_report or {})
    panel.sample_report["proc_root"] = str(proc_root)
    panel.sample_report["years"] = _normalize_years(years)
    panel.sample_report["panel_return_scheme"] = panel.panel_return_scheme
    panel.sample_report["requested_return_mode"] = return_mode
    return _subset_panel_columns(panel, max_stocks)


def load_proc_5min_panel(
    proc_root: str | Path = DEFAULT_PROC_ROOT,
    years: Optional[Sequence[int]] = None,
    max_stocks: Optional[int] = None,
) -> HF5MinPanel:
    proc_root = _ensure_path(proc_root)
    panel_name = _proc_panel_name(years)
    if panel_name is None:
        raise ValueError("load_proc_5min_panel only supports one calendar year or full panel.")
    _, meta_path = _proc_panel_paths(proc_root, STRICT_BALANCED_SAMPLE, panel_name)
    meta = _load_json(meta_path)
    array_files = meta.get("array_files", {})
    if "R_5min_full" not in array_files or "day_ids" not in array_files:
        raise ValueError(f"Processed panel metadata missing R_5min_full/day_ids array files: {meta_path}")
    r_path = proc_root / array_files["R_5min_full"]
    day_ids_path = proc_root / array_files["day_ids"]
    r_5min_full = np.load(r_path, mmap_mode="r", allow_pickle=False)
    day_ids = np.load(day_ids_path, mmap_mode="r", allow_pickle=False)
    tickers = list(meta["tickers"])
    if max_stocks is not None:
        keep = int(max_stocks)
        tickers = tickers[:keep]
        r_5min_full = r_5min_full[:, :keep]
    return HF5MinPanel(
        R_5min_full=r_5min_full,
        day_ids=day_ids,
        tickers=tickers,
        dates=[pd.Timestamp(date) for date in meta["dates"]],
        bar_times=list(meta.get("bar_times", [])),
        sample_mode=meta.get("sample_mode", STRICT_BALANCED_SAMPLE),
        panel_name=panel_name,
    )


def _build_unbalanced_year_5min_panel(
    proc_root: Path,
    year: int,
    year_dates: Sequence[str],
    tickers: Sequence[str],
    panel_workers: int,
    memmap_ctx: Optional[MemmapContext] = None,
) -> HF5MinPanel:
    tickers = [str(ticker) for ticker in tickers]
    if not year_dates:
        raise ValueError(f"找不到年份 {year} 的交易日历。")
    if not tickers:
        raise ValueError(f"{year} 年没有可用的非平衡样本。")
    date_codes = np.array([int(str(date).replace("-", "")) for date in year_dates], dtype=np.int32)
    day_count = len(year_dates)
    bar_count = len(CN_5MIN_BAR_TIMES)
    stock_count = len(tickers)
    ctx = memmap_ctx or MemmapContext(scratch_root=None)
    r_5min_full = ctx.allocate((day_count * bar_count, stock_count), dtype=np.float64, fill_value=np.nan, stem=f"{year}_full5min")
    tasks = [(proc_root, symbol, date_codes) for symbol in tickers]
    panel_workers = max(1, int(panel_workers))
    if panel_workers == 1 or len(tasks) <= 1:
        for col_idx, task in enumerate(tasks):
            _, full_5min = _load_and_align_symbol_5min_only_task(task)
            r_5min_full[:, col_idx] = full_5min
    else:
        with ThreadPoolExecutor(max_workers=panel_workers) as executor:
            futures = {executor.submit(_load_and_align_symbol_5min_only_task, task): col_idx for col_idx, task in enumerate(tasks)}
            for future in as_completed(futures):
                col_idx = futures[future]
                _, full_5min = future.result()
                r_5min_full[:, col_idx] = full_5min
    day_ids = np.repeat(np.arange(day_count), bar_count).astype(np.int32)
    return HF5MinPanel(
        R_5min_full=r_5min_full,
        day_ids=day_ids,
        tickers=tickers,
        dates=[pd.Timestamp(date) for date in year_dates],
        bar_times=list(CN_5MIN_BAR_TIMES),
        sample_mode="unbalanced_yearly",
        panel_name=f"unbalanced_year_{year}",
    )



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
        R_intra=panel.R_intra[selected_day_idx],
        R_night=panel.R_night[selected_day_idx],
        day_ids=new_day_ids,
        R_5min_full=panel.R_5min_full[row_mask],
        tickers=list(panel.tickers),
        dates=[panel.dates[idx] for idx in selected_day_idx],
        R_daily=panel.R_daily[selected_day_idx],
        rf_intra=panel.rf_intra[selected_day_idx],
        rf_night=panel.rf_night[selected_day_idx],
        bar_times=list(panel.bar_times or []),
        sample_report=sample_report,
        sample_mode=panel.sample_mode,
        panel_return_scheme=panel.panel_return_scheme,
        requested_return_mode=panel.requested_return_mode,
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
    memmap_ctx: Optional[MemmapContext] = None,
    stats_out: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TOD 阈值法识别跳跃.

    返回:
        R_cont, R_jump
    若输入包含 NaN, 输出会在缺失位置保留 NaN.
    """
    R = panel.R_5min_full
    D, M, N = panel.D, panel.M_per_day, panel.N
    delta_M = 1.0 / M

    R_cube = R.reshape(D, M, N)
    BV = _bipower_variation_cube(R_cube)
    BV = np.where(np.isfinite(BV) & (BV > 0.0), BV, np.nan)

    tod = _time_of_day_pattern(R, M)
    sqrt_BV = np.sqrt(BV)
    sqrt_TOD = np.sqrt(np.where(np.isfinite(tod) & (tod > 0.0), tod, 1.0))
    ctx = memmap_ctx or MemmapContext(scratch_root=None)
    R_jump = ctx.allocate(R.shape, dtype=np.float64, fill_value=np.nan, stem="jump")
    R_cont = ctx.allocate(R.shape, dtype=np.float64, fill_value=np.nan, stem="cont")
    jump_scale = float(a) * float(delta_M ** omega_bar)
    observed = 0
    n_jump = 0
    qv_total = 0.0
    qv_jump = 0.0

    for tod_idx in range(M):
        row_slice = slice(tod_idx, D * M, M)
        R_block = R[row_slice, :]
        thr_block = jump_scale * (sqrt_BV * float(sqrt_TOD[tod_idx]))
        finite_block = np.isfinite(R_block)
        jump_block = finite_block & (np.abs(R_block) > thr_block)
        observed += int(finite_block.sum())
        n_jump += int(jump_block.sum())
        if observed:
            abs_sq = np.square(R_block, where=finite_block, out=np.zeros_like(R_block))
            qv_total += float(abs_sq.sum())
            qv_jump += float(np.square(R_block, where=jump_block, out=np.zeros_like(R_block)).sum())

        jump_values = np.where(jump_block, R_block, 0.0)
        cont_values = np.where(jump_block, 0.0, R_block)
        if not np.all(finite_block):
            jump_values = np.where(finite_block, jump_values, np.nan)
            cont_values = np.where(finite_block, cont_values, np.nan)

        R_jump[row_slice, :] = jump_values
        R_cont[row_slice, :] = cont_values
    if stats_out is not None:
        stats_out.clear()
        stats_out.update(
            {
                "frac_jump_increments": float(n_jump / observed) if observed > 0 else 0.0,
                "frac_qv_explained_by_jumps": float(qv_jump / qv_total) if qv_total > 0 else 0.0,
            }
        )
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


def _truncate_pca_result(res: PCAResult, K: int) -> PCAResult:
    """Keep the first K factors while preserving the original eigen-spectrum metadata."""
    K_eff = min(max(int(K), 1), int(res.Lambda.shape[1]))
    return PCAResult(
        Lambda=res.Lambda[:, :K_eff],
        F=res.F[:, :K_eff],
        eigvals=res.eigvals,
        scales=res.scales,
        use_corr=res.use_corr,
        covariance=res.covariance,
        counts=res.counts,
    )


@dataclass
class YearJumpDecomposition:
    threshold: float
    R_cont: np.ndarray
    R_jump: np.ndarray
    stats: Dict[str, float]
    arrays_retained: bool = True


@dataclass
class YearPanelAnalysis:
    year: int
    balanced_year: HFPanel
    unbalanced_year: HFPanel
    decompositions: Dict[Tuple[str, float], YearJumpDecomposition] = field(default_factory=dict)
    pca_cache: Dict[Tuple[str, str, float], PCAResult] = field(default_factory=dict)
    explained_cache: Dict[Tuple[str, str, float, int], float] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    scratch_root: Optional[Path] = None


@dataclass
class YearlyPaperOutputs:
    table_i: pd.DataFrame
    table_ii: pd.DataFrame
    factor_counts: pd.DataFrame
    stage_timings: Dict[str, float] = field(default_factory=dict)


def _paper_year_checkpoint_paths(year_dir: Path) -> Dict[str, Path]:
    return {
        "table_i": year_dir / "table_i.pkl",
        "table_ii": year_dir / "table_ii.pkl",
        "factor_counts": year_dir / "factor_counts.pkl",
        "timings": year_dir / "timings.json",
        "meta": year_dir / "meta.json",
        "complete": year_dir / "complete.json",
    }


def _write_paper_year_checkpoint(year_dir: Path, item: Dict[str, Any]) -> None:
    year_dir.mkdir(parents=True, exist_ok=True)
    paths = _paper_year_checkpoint_paths(year_dir)
    _atomic_to_pickle(pd.DataFrame(item.get("table_i_rows", [])), paths["table_i"])
    _atomic_to_pickle(pd.DataFrame(item.get("table_ii_rows", [])), paths["table_ii"])
    _atomic_to_pickle(pd.DataFrame(item.get("factor_count_rows", [])), paths["factor_counts"])
    _write_json(paths["timings"], dict(item.get("stage_timings", {})))
    _write_json(
        paths["meta"],
        {
            "year": int(item.get("year")),
            "n_days": int(item.get("n_days", 0)),
            "n_symbols": int(item.get("n_symbols", 0)),
        },
    )
    _write_json(
        paths["complete"],
        {
            "year": int(item.get("year")),
            "completed_at": pd.Timestamp.utcnow().isoformat(),
        },
    )


def _load_paper_year_checkpoint(year_dir: Path) -> Dict[str, Any]:
    paths = _paper_year_checkpoint_paths(year_dir)
    if not paths["complete"].exists():
        raise FileNotFoundError(f"Incomplete paper checkpoint: {year_dir}")
    return {
        "table_i": pd.read_pickle(paths["table_i"]),
        "table_ii": pd.read_pickle(paths["table_ii"]),
        "factor_counts": pd.read_pickle(paths["factor_counts"]),
        "stage_timings": _load_json(paths["timings"]),
        "meta": _load_json(paths["meta"]),
        "complete": _load_json(paths["complete"]),
    }


def _assemble_yearly_paper_outputs_from_checkpoints(
    checkpoint_manager: CheckpointManager,
    years: Sequence[int],
) -> YearlyPaperOutputs:
    stage_timings = {
        "panel_build_sec": 0.0,
        "jump_decompose_sec": 0.0,
        "pca_sec": 0.0,
        "table_assemble_sec": 0.0,
    }
    table_i_parts: List[pd.DataFrame] = []
    table_ii_parts: List[pd.DataFrame] = []
    factor_count_parts: List[pd.DataFrame] = []
    for year in sorted(int(value) for value in years):
        payload = _load_paper_year_checkpoint(checkpoint_manager.paper_year_dir(int(year)))
        table_i_parts.append(payload["table_i"])
        table_ii_parts.append(payload["table_ii"])
        factor_count_parts.append(payload["factor_counts"])
        item_timings = payload.get("stage_timings", {})
        for key in stage_timings:
            stage_timings[key] += float(item_timings.get(key, 0.0))

    t_assemble = time.perf_counter()
    table_i_df = pd.concat(table_i_parts, ignore_index=True) if table_i_parts else pd.DataFrame()
    if not table_i_df.empty:
        table_i_df = table_i_df.sort_values(["panel_block", "year", "threshold_a"]).reset_index(drop=True)

    table_ii_df = pd.concat(table_ii_parts, ignore_index=True) if table_ii_parts else pd.DataFrame()
    if not table_ii_df.empty:
        component_order = {"hf": 0, "continuous": 1, "jump": 2}
        table_ii_df["_component_order"] = table_ii_df["return_component"].map(component_order).fillna(99)
        table_ii_df = table_ii_df.sort_values(["year", "_component_order"]).drop(columns=["_component_order"]).reset_index(drop=True)

    factor_count_df = pd.concat(factor_count_parts, ignore_index=True) if factor_count_parts else pd.DataFrame()
    if not factor_count_df.empty:
        component_order = {"hf": 0, "continuous": 1, "jump": 2}
        block_order = {"Balanced panel": 0, "Unbalanced panel": 1}
        factor_count_df["_block_order"] = factor_count_df["panel_block"].map(block_order).fillna(99)
        factor_count_df["_component_order"] = factor_count_df["return_component"].map(component_order).fillna(99)
        factor_count_df = factor_count_df.sort_values(["_block_order", "year", "_component_order"]).drop(columns=["_block_order", "_component_order"]).reset_index(drop=True)
    stage_timings["table_assemble_sec"] += time.perf_counter() - t_assemble

    return YearlyPaperOutputs(
        table_i=table_i_df,
        table_ii=table_ii_df,
        factor_counts=factor_count_df,
        stage_timings=stage_timings,
    )


def _assemble_yearly_paper_outputs_from_items(yearly_results: Sequence[Dict[str, Any]]) -> YearlyPaperOutputs:
    stage_timings = {
        "panel_build_sec": 0.0,
        "jump_decompose_sec": 0.0,
        "pca_sec": 0.0,
        "table_assemble_sec": 0.0,
    }
    for item in yearly_results:
        item_timings = item.get("stage_timings", {})
        for key in stage_timings:
            stage_timings[key] += float(item_timings.get(key, 0.0))

    t_assemble = time.perf_counter()
    table_i_rows = [row for item in yearly_results for row in item["table_i_rows"]]
    table_ii_rows = [row for item in yearly_results for row in item["table_ii_rows"]]
    factor_count_rows = [row for item in yearly_results for row in item["factor_count_rows"]]

    table_i_df = pd.DataFrame(table_i_rows)
    if not table_i_df.empty:
        table_i_df = table_i_df.sort_values(["panel_block", "year", "threshold_a"]).reset_index(drop=True)

    table_ii_df = pd.DataFrame(table_ii_rows)
    if not table_ii_df.empty:
        component_order = {"hf": 0, "continuous": 1, "jump": 2}
        table_ii_df["_component_order"] = table_ii_df["return_component"].map(component_order).fillna(99)
        table_ii_df = table_ii_df.sort_values(["year", "_component_order"]).drop(columns=["_component_order"]).reset_index(drop=True)

    factor_count_df = pd.DataFrame(factor_count_rows)
    if not factor_count_df.empty:
        component_order = {"hf": 0, "continuous": 1, "jump": 2}
        block_order = {"Balanced panel": 0, "Unbalanced panel": 1}
        factor_count_df["_block_order"] = factor_count_df["panel_block"].map(block_order).fillna(99)
        factor_count_df["_component_order"] = factor_count_df["return_component"].map(component_order).fillna(99)
        factor_count_df = factor_count_df.sort_values(["_block_order", "year", "_component_order"]).drop(columns=["_block_order", "_component_order"]).reset_index(drop=True)
    stage_timings["table_assemble_sec"] += time.perf_counter() - t_assemble

    return YearlyPaperOutputs(
        table_i=table_i_df,
        table_ii=table_ii_df,
        factor_counts=factor_count_df,
        stage_timings=stage_timings,
    )


def _build_runtime_config(
    output_root: Path,
    memory_budget_gb: Optional[float],
    progress_interval_sec: float,
) -> RuntimeConfig:
    physical_memory_bytes = _physical_memory_bytes()
    if memory_budget_gb is None:
        budget_bytes = int(max(1, physical_memory_bytes * DEFAULT_MEMORY_BUDGET_RATIO))
    else:
        requested_budget_gb = float(memory_budget_gb)
        if requested_budget_gb <= 0.0:
            raise ValueError("--memory-budget-gb must be positive")
        budget_bytes = int(max(1.0, requested_budget_gb * (1024 ** 3)))
    diagnostics_dir = output_root / "diagnostics"
    scratch_root = diagnostics_dir / "runtime_tmp"
    return RuntimeConfig(
        memory_budget_bytes=budget_bytes,
        memory_budget_gb=_format_bytes_gb(budget_bytes),
        progress_interval_sec=max(1.0, float(progress_interval_sec)),
        physical_memory_bytes=physical_memory_bytes,
        logical_cpus=int(os.cpu_count() or 1),
        scratch_root=scratch_root,
    )


def _standardize_returns_for_pca(
    R: np.ndarray,
    use_corr: bool,
    allow_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float)
    if not allow_nan and np.isnan(R).any():
        raise ValueError("PCA input contains NaN values")
    _, N = R.shape
    if use_corr:
        qv = np.nansum(R ** 2, axis=0) if allow_nan else np.sum(R ** 2, axis=0)
        scales = np.sqrt(np.where(qv > 0.0, qv, 1.0))
        Rs = R / scales
    else:
        Rs = R
        scales = np.ones(N, dtype=float)
    return Rs, scales


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
    Rs, scales = _standardize_returns_for_pca(R, use_corr=use_corr, allow_nan=False)

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
    R = np.asarray(R, dtype=float)
    _, N = R.shape
    block_size = max(64, min(512, N))
    counts = np.zeros((N, N), dtype=np.int32)
    cross = np.zeros((N, N), dtype=np.float64)
    mask = np.isfinite(R)
    mask_int = mask.astype(np.int32)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        mask_block = mask_int[:, start:end]
        data_block = np.where(mask[:, start:end], R[:, start:end], 0.0)
        for other_start in range(start, N, block_size):
            other_end = min(other_start + block_size, N)
            other_mask_block = mask_int[:, other_start:other_end]
            other_data = np.where(mask[:, other_start:other_end], R[:, other_start:other_end], 0.0)
            cross_block = data_block.T @ other_data
            count_block = mask_block.T @ other_mask_block
            cross[start:end, other_start:other_end] = cross_block
            counts[start:end, other_start:other_end] = count_block
            if other_start != start:
                cross[other_start:other_end, start:end] = cross_block.T
                counts[other_start:other_end, start:end] = count_block.T

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
    对含缺失值的矩阵做 pairwise-covariance PCA。

    这是备用实现，当前 strict-balanced 主流程不依赖它。
    """
    R = np.asarray(R, dtype=float)
    M, N = R.shape
    Rs, scales = _standardize_returns_for_pca(R, use_corr=use_corr, allow_nan=True)

    pairwise_cov, counts = pairwise_available_covariance(Rs)
    eigvals, V = _sym_eig_desc(pairwise_cov)
    if psd_fix:
        eigvals = np.clip(eigvals, 0.0, None)
    K_eff = min(K, V.shape[1])
    Lambda = np.sqrt(N) * V[:, :K_eff]
    F = np.nan_to_num(Rs, nan=0.0) @ Lambda / N

    return PCAResult(
        Lambda=Lambda,
        F=F,
        eigvals=eigvals,
        scales=scales,
        use_corr=use_corr,
        covariance=None,
        counts=counts,
    )


def _explained_variation_complete_top_k(
    R: np.ndarray,
    K: int,
    use_corr: bool = True,
) -> float:
    R = np.asarray(R, dtype=float)
    if np.isnan(R).any():
        raise ValueError("Complete-matrix explained variation does not accept NaN inputs")
    _, N = R.shape
    K_eff = min(max(int(K), 0), N)
    if K_eff <= 0:
        return 0.0
    Rs, _ = _standardize_returns_for_pca(R, use_corr=use_corr, allow_nan=False)
    S = (Rs.T @ Rs) / N
    total = float(np.trace(S))
    if total <= 0.0:
        return 0.0
    S = (S + S.T) / 2.0
    if K_eff >= N:
        top_eigvals = eigh(S, eigvals_only=True)
    else:
        top_eigvals = eigh(S, eigvals_only=True, subset_by_index=[N - K_eff, N - 1])
    return float(np.sum(top_eigvals) / total)


def _explained_variation_pairwise_top_k(
    R: np.ndarray,
    K: int,
    use_corr: bool = True,
    psd_fix: bool = True,
) -> float:
    R = np.asarray(R, dtype=float)
    _, N = R.shape
    K_eff = min(max(int(K), 0), N)
    if K_eff <= 0:
        return 0.0
    Rs, _ = _standardize_returns_for_pca(R, use_corr=use_corr, allow_nan=True)
    pairwise_cov, _ = pairwise_available_covariance(Rs)
    eigvals, _ = _sym_eig_desc(pairwise_cov)
    if psd_fix:
        eigvals = np.clip(eigvals, 0.0, None)
    total = float(np.sum(eigvals))
    if total <= 0.0:
        return 0.0
    return float(np.sum(eigvals[:K_eff]) / total)


def _panel_explained_variation_top_k(
    R: np.ndarray,
    K: int,
    use_corr: bool = True,
) -> float:
    if np.isnan(R).any():
        return _explained_variation_pairwise_top_k(R, K=K, use_corr=use_corr, psd_fix=True)
    return _explained_variation_complete_top_k(R, K=K, use_corr=use_corr)


def _build_paper_resource_plan(
    proc_root: Path,
    summary: Dict[str, Any],
    years: Sequence[int],
    requested_workers: int,
    requested_paper_workers: int,
    requested_rolling_workers: int,
    runtime: RuntimeConfig,
    universe: pd.DataFrame,
    max_stocks: Optional[int],
) -> ResourcePlan:
    year_estimates: List[Dict[str, Any]] = []
    requested_workers = max(1, int(requested_workers))
    requested_paper_workers = max(1, int(requested_paper_workers))
    requested_rolling_workers = max(1, int(requested_rolling_workers))
    memory_cap = requested_paper_workers
    for year in years:
        year_dates = _year_dates_from_universe_summary(summary, int(year))
        tickers = _select_unbalanced_symbols_for_year(universe, int(year), max_stocks=max_stocks)
        day_count = len(year_dates)
        unbalanced_n = len(tickers)
        balanced_n = _balanced_symbol_count_for_year(proc_root, int(year), max_stocks=max_stocks)
        peak_estimate = _estimate_paper_year_peak_bytes(
            day_count=day_count,
            balanced_stock_count=balanced_n,
            unbalanced_stock_count=unbalanced_n,
            safety_multiplier=PAPER_MEMORY_SAFETY_MULTIPLIER,
        )
        year_estimates.append(
            {
                "year": int(year),
                "day_count": int(day_count),
                "unbalanced_symbols": int(unbalanced_n),
                "balanced_symbols": int(balanced_n),
                **peak_estimate,
            }
        )
    year_estimates.sort(key=lambda item: int(item["peak_memory_bytes"]), reverse=True)
    if year_estimates:
        largest_year_bytes = max(int(item["peak_memory_bytes"]) for item in year_estimates)
        memory_cap = max(1, min(requested_paper_workers, runtime.memory_budget_bytes // max(largest_year_bytes, 1)))
    cpu_cap = max(1, min(requested_paper_workers, max(1, runtime.logical_cpus // 2)))
    paper_workers_effective = max(1, min(requested_paper_workers, memory_cap, cpu_cap))
    paper_blas_threads = max(1, min(8, (max(1, runtime.logical_cpus - 2)) // paper_workers_effective))
    rolling_workers_effective = max(1, min(requested_rolling_workers, max(1, runtime.logical_cpus - 1)))
    return ResourcePlan(
        physical_memory_bytes=runtime.physical_memory_bytes,
        physical_memory_gb=_format_bytes_gb(runtime.physical_memory_bytes),
        memory_budget_bytes=runtime.memory_budget_bytes,
        memory_budget_gb=runtime.memory_budget_gb,
        logical_cpus=runtime.logical_cpus,
        requested_workers=int(requested_workers),
        requested_paper_workers=int(requested_paper_workers),
        requested_rolling_workers=int(requested_rolling_workers),
        paper_workers_effective=int(paper_workers_effective),
        rolling_workers_effective=int(rolling_workers_effective),
        paper_blas_threads=int(max(1, paper_blas_threads)),
        execution_mode="adaptive_steady_state",
        paper_schedule_policy="memory_budget_driven_process_pool",
        paper_memory_cap_workers=int(memory_cap),
        paper_cpu_cap_workers=int(cpu_cap),
        paper_memory_safety_multiplier=float(PAPER_MEMORY_SAFETY_MULTIPLIER),
        paper_year_estimates=year_estimates,
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


def _day_offsets_from_day_ids(day_ids: np.ndarray) -> np.ndarray:
    day_ids = np.asarray(day_ids, dtype=np.int64)
    day_count = int(day_ids.max()) + 1 if day_ids.size else 0
    counts = np.bincount(day_ids, minlength=day_count).astype(np.int64, copy=False)
    offsets = np.zeros(day_count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets


def _rolling_local_pca_result(
    R: np.ndarray,
    day_offsets: np.ndarray,
    start_day: int,
    window_days: int,
    K: int,
    use_corr: bool,
) -> Optional[Dict[str, Any]]:
    end_day = int(start_day + window_days)
    row_start = int(day_offsets[start_day])
    row_end = int(day_offsets[end_day])
    R_win = R[row_start:row_end]
    if R_win.shape[0] < 2 * K:
        return None
    res = pca_factors(R_win, K=K, use_corr=use_corr)
    return {
        "start_day": int(start_day),
        "end_day": int(end_day),
        "Lambda": res.Lambda,
        "eigvals": res.eigvals,
        "scales": res.scales,
    }


def _save_array_for_memmap(path: Path, array: np.ndarray) -> Path:
    np.save(path, np.asarray(array), allow_pickle=False)
    return path


def rolling_window_chunk_worker(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    R = np.load(task["R_path"], mmap_mode="r", allow_pickle=False)
    day_offsets = np.asarray(task["day_offsets"], dtype=np.int64)
    window_days = int(task["window_days"])
    K = int(task["K"])
    use_corr = bool(task["use_corr"])
    out: List[Dict[str, Any]] = []
    for window_index, start_day in task["window_specs"]:
        result = _rolling_local_pca_result(
            R=R,
            day_offsets=day_offsets,
            start_day=int(start_day),
            window_days=window_days,
            K=K,
            use_corr=use_corr,
        )
        if result is None:
            continue
        result["window_index"] = int(window_index)
        out.append(result)
    return out


def _rolling_window_specs(day_ids: np.ndarray, window_days: int, step_days: int) -> List[Tuple[int, int]]:
    D = int(day_ids.max()) + 1 if len(day_ids) else 0
    starts = list(range(0, max(D - window_days + 1, 0), step_days))
    return [(int(window_index), int(start_day)) for window_index, start_day in enumerate(starts)]


def rolling_checkpoint_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    R = np.load(task["R_path"], mmap_mode="r", allow_pickle=False)
    global_Lambda = np.load(task["global_Lambda_path"], mmap_mode="r", allow_pickle=False)
    day_offsets = np.asarray(task["day_offsets"], dtype=np.int64)
    window_days = int(task["window_days"])
    K = int(task["K"])
    use_corr = bool(task["use_corr"])
    weight_step_days = int(task["weight_step_days"])
    selected_idx = np.asarray(task["selected_idx"], dtype=np.int64)

    window_indices: List[int] = []
    start_days: List[int] = []
    end_days: List[int] = []
    gc_rows: List[np.ndarray] = []
    explained_rows: List[float] = []
    weight_window_indices: List[int] = []
    weight_start_days: List[int] = []
    weight_end_days: List[int] = []
    weight_rows: List[np.ndarray] = []

    for window_index, start_day in task["window_specs"]:
        result = _rolling_local_pca_result(
            R=R,
            day_offsets=day_offsets,
            start_day=int(start_day),
            window_days=window_days,
            K=K,
            use_corr=use_corr,
        )
        if result is None:
            continue
        gc = generalized_correlations(global_Lambda, result["Lambda"])
        gc_padded = np.zeros(K, dtype=np.float64)
        gc_padded[: len(gc)] = gc
        eigvals = np.asarray(result["eigvals"], dtype=np.float64)
        total = float(eigvals.sum())
        explained = float(eigvals[:K].sum() / total) if total > 0 else 0.0

        window_indices.append(int(window_index))
        start_days.append(int(result["start_day"]))
        end_days.append(int(result["end_day"]))
        gc_rows.append(gc_padded)
        explained_rows.append(explained)

        if selected_idx.size and int(result["start_day"]) % max(1, weight_step_days) == 0:
            local_weight_factor_1 = result["Lambda"][selected_idx, 0] / (
                np.sqrt(result["Lambda"].shape[0]) * result["scales"][selected_idx]
            )
            weight_window_indices.append(int(window_index))
            weight_start_days.append(int(result["start_day"]))
            weight_end_days.append(int(result["end_day"]))
            weight_rows.append(np.asarray(local_weight_factor_1, dtype=np.float64))

    return {
        "chunk_index": int(task["chunk_index"]),
        "window_index": np.asarray(window_indices, dtype=np.int32),
        "start_day": np.asarray(start_days, dtype=np.int32),
        "end_day": np.asarray(end_days, dtype=np.int32),
        "gc": np.vstack(gc_rows) if gc_rows else np.zeros((0, K), dtype=np.float64),
        "explained_variation": np.asarray(explained_rows, dtype=np.float64),
        "weight_window_index": np.asarray(weight_window_indices, dtype=np.int32),
        "weight_start_day": np.asarray(weight_start_days, dtype=np.int32),
        "weight_end_day": np.asarray(weight_end_days, dtype=np.int32),
        "weight_factor_1": np.vstack(weight_rows) if weight_rows else np.zeros((0, len(selected_idx)), dtype=np.float64),
    }


def _load_rolling_checkpoint_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _assemble_rolling_outputs_from_checkpoints(
    checkpoint_manager: CheckpointManager,
    selected_symbols: Sequence[str],
    K: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    window_records: List[Tuple[int, np.ndarray, float]] = []
    weight_rows: List[Dict[str, Any]] = []
    for path in sorted(checkpoint_manager.layout.rolling_dir.glob("chunk_*.npz")):
        payload = _load_rolling_checkpoint_arrays(path)
        window_index = payload.get("window_index", np.zeros(0, dtype=np.int32))
        gc = payload.get("gc", np.zeros((0, K), dtype=np.float64))
        explained = payload.get("explained_variation", np.zeros(0, dtype=np.float64))
        for idx, win in enumerate(window_index.tolist()):
            window_records.append((int(win), np.asarray(gc[idx], dtype=np.float64), float(explained[idx])))

        weight_matrix = payload.get("weight_factor_1", np.zeros((0, len(selected_symbols)), dtype=np.float64))
        start_days = payload.get("weight_start_day", np.zeros(0, dtype=np.int32))
        end_days = payload.get("weight_end_day", np.zeros(0, dtype=np.int32))
        for row_idx in range(weight_matrix.shape[0]):
            for symbol_idx, symbol in enumerate(selected_symbols):
                weight_rows.append(
                    {
                        "start_day": int(start_days[row_idx]),
                        "end_day": int(end_days[row_idx]),
                        "symbol": str(symbol),
                        "weight_factor_1": float(weight_matrix[row_idx, symbol_idx]),
                    }
                )

    window_records.sort(key=lambda item: int(item[0]))
    rolling_gc = np.vstack([item[1] for item in window_records]) if window_records else np.zeros((0, K), dtype=np.float64)
    rolling_ev = np.asarray([item[2] for item in window_records], dtype=np.float64)
    rolling_weight_summary = pd.DataFrame(weight_rows)
    if not rolling_weight_summary.empty:
        rolling_weight_summary = rolling_weight_summary.sort_values(["start_day", "symbol"]).reset_index(drop=True)
    return rolling_gc, rolling_ev, rolling_weight_summary


def run_checkpointed_rolling_pca(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    global_Lambda: np.ndarray,
    checkpoint_manager: CheckpointManager,
    selected_symbols: Sequence[str],
    selected_idx: np.ndarray,
    step_days: int = 1,
    use_corr: bool = True,
    workers: int = 1,
    weight_step_days: int = 21,
    progress: Optional[ProgressReporter] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    window_specs = _rolling_window_specs(day_ids=day_ids, window_days=window_days, step_days=step_days)
    chunk_specs: List[Tuple[int, List[Tuple[int, int]]]] = []
    for start in range(0, len(window_specs), ROLLING_CHECKPOINT_WINDOW_COUNT):
        chunk_index = start // ROLLING_CHECKPOINT_WINDOW_COUNT
        chunk_specs.append((int(chunk_index), window_specs[start : start + ROLLING_CHECKPOINT_WINDOW_COUNT]))

    checkpoint_manager.mark_rolling_plan(len(chunk_specs))
    completed_chunks = checkpoint_manager.completed_rolling_chunks()
    skipped_chunks = 0
    if progress is not None:
        progress.update_state(stage="rolling", total_chunks=len(chunk_specs))

    pending_chunk_specs: List[Tuple[int, List[Tuple[int, int]]]] = []
    for chunk_index, chunk in chunk_specs:
        chunk_path = checkpoint_manager.rolling_chunk_path(chunk_index)
        if chunk_path.exists() and chunk_index in completed_chunks:
            skipped_chunks += 1
            if progress is not None:
                progress.event(
                    "rolling_chunk_skipped",
                    stage="rolling",
                    chunk_index=int(chunk_index),
                    completed_chunks=skipped_chunks,
                    total_chunks=len(chunk_specs),
                    message="reused rolling checkpoint",
                )
            continue
        pending_chunk_specs.append((chunk_index, chunk))

    workers = _safe_process_pool_worker_count("rolling PCA", max(1, int(workers)))
    day_offsets = _day_offsets_from_day_ids(day_ids)
    with tempfile.TemporaryDirectory(prefix="pelger_cn_roll_") as tmpdir, _temporary_blas_thread_env(1):
        tmpdir_path = Path(tmpdir)
        R_path = _save_array_for_memmap(tmpdir_path / "R.npy", np.asarray(R, dtype=np.float64))
        global_lambda_path = _save_array_for_memmap(tmpdir_path / "global_lambda.npy", np.asarray(global_Lambda, dtype=np.float64))
        tasks = [
            {
                "chunk_index": int(chunk_index),
                "R_path": str(R_path),
                "global_Lambda_path": str(global_lambda_path),
                "day_offsets": day_offsets.tolist(),
                "window_days": int(window_days),
                "K": int(K),
                "use_corr": bool(use_corr),
                "weight_step_days": int(weight_step_days),
                "selected_idx": np.asarray(selected_idx, dtype=np.int64).tolist(),
                "window_specs": [(int(window_index), int(start_day)) for window_index, start_day in chunk],
            }
            for chunk_index, chunk in pending_chunk_specs
        ]

        finished_chunks = skipped_chunks
        total_chunks = len(chunk_specs)
        if workers == 1 or len(tasks) <= 1:
            for task in tasks:
                payload = rolling_checkpoint_chunk_worker(task)
                _atomic_save_npz(checkpoint_manager.rolling_chunk_path(int(task["chunk_index"])), **payload)
                checkpoint_manager.mark_rolling_chunk_complete(int(task["chunk_index"]))
                finished_chunks += 1
                if progress is not None:
                    progress.event(
                        "rolling_chunk_finished",
                        stage="rolling",
                        chunk_index=int(task["chunk_index"]),
                        completed_chunks=finished_chunks,
                        total_chunks=total_chunks,
                    )
                    progress.heartbeat(force=True, completed_chunks=finished_chunks, total_chunks=total_chunks)
        else:
            try:
                with _spawn_process_pool(workers) as executor:
                    future_map = {executor.submit(rolling_checkpoint_chunk_worker, task): task for task in tasks}
                    while future_map:
                        progress and progress.heartbeat(completed_chunks=finished_chunks, total_chunks=total_chunks)
                        try:
                            future = next(
                                as_completed(
                                    list(future_map.keys()),
                                    timeout=progress.interval_sec if progress is not None else None,
                                )
                            )
                        except FuturesTimeoutError:
                            continue
                        task = future_map.pop(future)
                        payload = future.result()
                        _atomic_save_npz(checkpoint_manager.rolling_chunk_path(int(task["chunk_index"])), **payload)
                        checkpoint_manager.mark_rolling_chunk_complete(int(task["chunk_index"]))
                        finished_chunks += 1
                        if progress is not None:
                            progress.event(
                                "rolling_chunk_finished",
                                stage="rolling",
                                chunk_index=int(task["chunk_index"]),
                                completed_chunks=finished_chunks,
                                total_chunks=total_chunks,
                            )
            except Exception as exc:
                if not _is_spawn_context_failure(exc):
                    raise
                _warn_parallel_fallback("rolling PCA", str(exc))
                for task in tasks:
                    payload = rolling_checkpoint_chunk_worker(task)
                    _atomic_save_npz(checkpoint_manager.rolling_chunk_path(int(task["chunk_index"])), **payload)
                    checkpoint_manager.mark_rolling_chunk_complete(int(task["chunk_index"]))
                    finished_chunks += 1
                    if progress is not None:
                        progress.event(
                            "rolling_chunk_finished",
                            stage="rolling",
                            chunk_index=int(task["chunk_index"]),
                            completed_chunks=finished_chunks,
                            total_chunks=total_chunks,
                            message="fallback single-process chunk finished",
                        )

    return _assemble_rolling_outputs_from_checkpoints(
        checkpoint_manager=checkpoint_manager,
        selected_symbols=selected_symbols,
        K=int(K),
    )


def rolling_local_pca(
    R: np.ndarray,
    day_ids: np.ndarray,
    window_days: int,
    K: int,
    step_days: int = 1,
    use_corr: bool = True,
    workers: int = 1,
    progress: Optional[ProgressReporter] = None,
) -> List[Dict[str, Any]]:
    """按滚动交易日窗口做局部 PCA；多进程模式使用只读 memmap 共享输入矩阵。"""
    D = int(day_ids.max()) + 1 if len(day_ids) else 0
    starts = list(range(0, max(D - window_days + 1, 0), step_days))
    workers = _safe_process_pool_worker_count("rolling PCA", max(1, int(workers)))
    day_offsets = _day_offsets_from_day_ids(day_ids)

    if workers == 1 or len(starts) <= 1:
        results = []
        total_chunks = len(starts)
        for idx, start in enumerate(starts, start=1):
            result = _rolling_local_pca_result(
                R=R,
                day_offsets=day_offsets,
                start_day=start,
                window_days=window_days,
                K=K,
                use_corr=use_corr,
            )
            results.append(result)
            if progress is not None and (idx == total_chunks or idx % max(1, step_days) == 0):
                progress.event(
                    "rolling_chunk_finished",
                    stage="rolling",
                    completed_chunks=idx,
                    total_chunks=total_chunks,
                    start_day=int(start),
                )
    else:
        window_specs = list(enumerate(starts))
        chunks = _chunk_sequence(window_specs, workers)
        with tempfile.TemporaryDirectory(prefix="pelger_cn_roll_") as tmpdir, _temporary_blas_thread_env(1):
            tmpdir_path = Path(tmpdir)
            R_path = _save_array_for_memmap(tmpdir_path / "R.npy", np.asarray(R, dtype=np.float64))
            tasks = [
                {
                    "R_path": str(R_path),
                    "day_offsets": day_offsets.tolist(),
                    "window_days": int(window_days),
                    "K": int(K),
                    "use_corr": bool(use_corr),
                    "window_specs": [(int(window_index), int(start_day)) for window_index, start_day in chunk],
                }
                for chunk in chunks
            ]
            results = []
            try:
                with _spawn_process_pool(workers) as executor:
                    futures = [executor.submit(rolling_window_chunk_worker, task) for task in tasks]
                    completed_chunks = 0
                    total_chunks = len(tasks)
                    for future in as_completed(futures):
                        chunk_result = future.result()
                        results.extend(chunk_result)
                        completed_chunks += 1
                        if progress is not None:
                            progress.event(
                                "rolling_chunk_finished",
                                stage="rolling",
                                completed_chunks=completed_chunks,
                                total_chunks=total_chunks,
                            )
            except Exception as exc:
                if not _is_spawn_context_failure(exc):
                    raise
                _warn_parallel_fallback("rolling PCA", str(exc))
                results = []
                total_chunks = len(tasks)
                for idx, task in enumerate(tasks, start=1):
                    results.extend(rolling_window_chunk_worker(task))
                    if progress is not None:
                        progress.event(
                            "rolling_chunk_finished",
                            stage="rolling",
                            completed_chunks=idx,
                            total_chunks=total_chunks,
                            message="fallback single-process chunk finished",
                        )
            results.sort(key=lambda item: int(item["window_index"]))
            for item in results:
                item.pop("window_index", None)
            return results

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


def rolling_gc_and_explained_variation_from_results(
    rolling_results: Sequence[Dict[str, Any]],
    global_Lambda: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """从已计算好的 rolling PCA 结果一次性拼出 GC 和 explained variation。"""
    if not rolling_results:
        return np.zeros((0, K)), np.zeros(0, dtype=float)

    GC = np.zeros((len(rolling_results), K))
    ratios: List[float] = []
    for i, result in enumerate(rolling_results):
        gc = generalized_correlations(global_Lambda, result["Lambda"])
        GC[i, : len(gc)] = gc
        eigvals = np.asarray(result["eigvals"], dtype=float)
        total = float(eigvals.sum())
        ratios.append(float(eigvals[:K].sum() / total) if total > 0 else 0.0)
    return GC, np.asarray(ratios, dtype=float)


def _rolling_weight_summary_from_results(
    panel: HFPanel,
    R_cont: np.ndarray,
    K: int,
    rolling_results: Sequence[Dict[str, Any]],
    step_days: int = 21,
    top_n: int = 8,
) -> pd.DataFrame:
    """Reuse rolling PCA outputs to avoid recomputing local PCA windows."""
    W_global = factor_portfolio_weights(pca_factors(R_cont, K=K, use_corr=True))
    selected_idx = np.argsort(-np.abs(W_global[:, 0]))[: min(top_n, panel.N)]
    rows: List[Dict[str, Any]] = []
    for result in rolling_results:
        tmp = type("_RollingPCA", (), {})()
        tmp.Lambda = result["Lambda"]
        tmp.scales = result["scales"]
        W_local = factor_portfolio_weights(tmp)  # type: ignore[arg-type]
        for idx in selected_idx:
            rows.append({
                "start_day": int(result["start_day"]),
                "end_day": int(result["end_day"]),
                "symbol": panel.tickers[idx],
                "weight_factor_1": float(W_local[idx, 0]),
            })
    return pd.DataFrame(rows)


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
    F_daily: Optional[np.ndarray] = None,
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
    F_daily_use = F_intra + F_night if F_daily is None else np.asarray(F_daily, dtype=float)
    _, sr_daily = tangency_portfolio(F_daily_use, np.asarray(rf_intra) + np.asarray(rf_night))

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
    """把高频因子收益按交易日聚合成日频收益."""
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
    display_factor_count: int = DISPLAY_CONTINUOUS_FACTOR_COUNT

    pca_hf: Optional[PCAResult] = None
    pca_cont: Optional[PCAResult] = None
    pca_jump: Optional[PCAResult] = None
    pca_cont_display: Optional[PCAResult] = None

    F_cont_daily_intra: Optional[np.ndarray] = None
    F_cont_daily_night: Optional[np.ndarray] = None
    F_cont_daily_total: Optional[np.ndarray] = None
    F_cont_display_daily_intra: Optional[np.ndarray] = None
    F_cont_display_daily_night: Optional[np.ndarray] = None
    F_cont_display_daily_total: Optional[np.ndarray] = None
    sharpes: Dict[str, float] = field(default_factory=dict)

    def step1_decompose(self) -> None:
        stats: Dict[str, float] = {}
        self.R_cont, self.R_jump = detect_jumps(self.panel, a=self.jump_a, stats_out=stats)
        self.jump_stats = dict(stats) if stats else jump_summary_stats(self.R_cont, self.R_jump)

    def step2_determine_K(self) -> None:
        N = self.panel.N
        r_hf = pca_factors(self.panel.R_5min_full, K=1, use_corr=self.use_corr)
        r_cont = pca_factors(self.R_cont, K=1, use_corr=self.use_corr)
        r_jump = pca_factors(self.R_jump, K=1, use_corr=self.use_corr)
        self.K_hf_hat, _ = perturbed_eigenvalue_ratio(r_hf.eigvals, g_fn=self.g_fn, N=N, gamma=self.gamma, K_max=self.K_max)
        self.K_cont_hat, _ = perturbed_eigenvalue_ratio(r_cont.eigvals, g_fn=self.g_fn, N=N, gamma=self.gamma, K_max=self.K_max)
        self.K_jump_hat, _ = perturbed_eigenvalue_ratio(r_jump.eigvals, g_fn=self.g_fn, N=N, gamma=self.gamma, K_max=self.K_max)

        self.K_hf_hat = max(1, self.K_hf_hat)
        self.K_cont_hat = max(1, self.K_cont_hat)
        self.K_jump_hat = max(1, self.K_jump_hat)

    def step3_extract_factors(self) -> None:
        self.pca_hf = pca_factors(self.panel.R_5min_full, K=self.K_hf_hat, use_corr=self.use_corr)
        self.pca_cont = pca_factors(self.R_cont, K=self.K_cont_hat, use_corr=self.use_corr)
        self.pca_jump = pca_factors(self.R_jump, K=self.K_jump_hat, use_corr=self.use_corr)
        display_k = min(max(1, int(self.display_factor_count)), self.panel.N)
        if self.pca_cont.Lambda.shape[1] >= display_k:
            self.pca_cont_display = _truncate_pca_result(self.pca_cont, display_k)
        else:
            self.pca_cont_display = pca_factors(self.R_cont, K=display_k, use_corr=self.use_corr)

    def step4_asset_pricing(self) -> None:
        W = factor_portfolio_weights(self.pca_cont)
        F_intra_daily = self.panel.R_intra @ W
        F_night_daily = self.panel.R_night @ W
        F_daily_total = self.panel.R_daily @ W
        if not np.allclose(F_daily_total, F_intra_daily + F_night_daily, equal_nan=True, atol=1e-12):
            raise ValueError("Daily factor returns must equal intraday plus overnight returns")

        self.F_cont_daily_intra = F_intra_daily
        self.F_cont_daily_night = F_night_daily
        self.F_cont_daily_total = F_daily_total
        self.sharpes = intraday_overnight_sharpes(
            F_intra=F_intra_daily,
            F_night=F_night_daily,
            F_daily=F_daily_total,
            rf_intra=self.panel.rf_intra,
            rf_night=self.panel.rf_night,
        )

        display_pca = self.pca_cont_display if self.pca_cont_display is not None else self.pca_cont
        W_display = factor_portfolio_weights(display_pca)
        F_display_intra_daily = self.panel.R_intra @ W_display
        F_display_night_daily = self.panel.R_night @ W_display
        F_display_daily_total = self.panel.R_daily @ W_display
        if not np.allclose(
            F_display_daily_total,
            F_display_intra_daily + F_display_night_daily,
            equal_nan=True,
            atol=1e-12,
        ):
            raise ValueError("Display factor daily returns must equal intraday plus overnight returns")

        self.F_cont_display_daily_intra = F_display_intra_daily
        self.F_cont_display_daily_night = F_display_night_daily
        self.F_cont_display_daily_total = F_display_daily_total

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
    print(f" Panel scheme  : {pipe.panel.panel_return_scheme}")
    if pipe.panel.requested_return_mode is not None:
        print(f" Requested mode : {pipe.panel.requested_return_mode} (deprecated no-op)")
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
    universe_summary: Dict[str, Any]
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
    paper_table_i: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_table_ii: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_table_iii: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_table_iv: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_table_v: pd.DataFrame = field(default_factory=pd.DataFrame)
    replication_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    proxy_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_pca_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_weight_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_return_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    cumulative_factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    plot_status: pd.DataFrame = field(default_factory=pd.DataFrame)
    exported_files: Dict[str, str] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    resource_plan: Dict[str, Any] = field(default_factory=dict)


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


def build_paper_jump_stats_comparison(
    balanced_panel: HFPanel,
    proc_root: Path,
    universe: pd.DataFrame,
    thresholds: Sequence[float] = (3.0, 4.0, 4.5, 5.0),
    workers: Optional[int] = None,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """Build paper-style Table I via the shared yearly analysis cache."""
    return build_yearly_paper_outputs(
        balanced_panel=balanced_panel,
        proc_root=proc_root,
        universe=universe,
        thresholds=thresholds,
        jump_a=3.0,
        gamma=0.08,
        g_fn="median_N",
        k_max=10,
        workers=workers,
        max_stocks=max_stocks,
    ).table_i


def _panel_pca(R: np.ndarray, K: int, use_corr: bool = True) -> PCAResult:
    """按矩阵是否含缺失值，自动选择普通 PCA 或 pairwise PCA。"""
    if np.isnan(R).any():
        return pca_factors_pairwise(R, K=K, use_corr=use_corr, psd_fix=True)
    return pca_factors(R, K=K, use_corr=use_corr)


def _table_i_rows_for_panel(
    panel: HFPanel,
    panel_block: str,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    year = int(panel.dates[0].year) if panel.dates else -1
    for a in thresholds:
        R_cont, R_jump = detect_jumps(panel, a=float(a))
        stats = jump_summary_stats(R_cont, R_jump)
        jump_res = _panel_pca(R_jump, K=1, use_corr=True)
        cont_res = _panel_pca(R_cont, K=min(4, panel.N), use_corr=True)
        rows.append(
            {
                "panel_block": panel_block,
                "year": year,
                "threshold_a": float(a),
                "n_symbols": int(panel.N),
                "n_days": int(panel.D),
                "frac_jump_increments": stats["frac_jump_increments"],
                "frac_qv_explained_by_jumps": stats["frac_qv_explained_by_jumps"],
                "jump_corr_explained_by_first_factor": _explained_variation_from_pca(jump_res, K=1),
                "continuous_corr_explained_by_first_four": _explained_variation_from_pca(cont_res, K=min(4, cont_res.eigvals.shape[0])),
            }
        )
    return rows


def _factor_count_rows_for_panel(
    panel: HFPanel,
    panel_block: str,
    jump_a: float,
    k_max: int,
    gamma: float,
    g_fn: str,
) -> List[Dict[str, Any]]:
    year = int(panel.dates[0].year) if panel.dates else -1
    R_cont, R_jump = detect_jumps(panel, a=jump_a)
    rows: List[Dict[str, Any]] = []
    for return_component, eigvals in {
        "hf": _panel_pca(panel.R_5min_full, K=1, use_corr=True).eigvals,
        "continuous": _panel_pca(R_cont, K=1, use_corr=True).eigvals,
        "jump": _panel_pca(R_jump, K=1, use_corr=True).eigvals,
    }.items():
        K_hat, er = perturbed_eigenvalue_ratio(eigvals, g_fn=g_fn, N=panel.N, gamma=gamma, K_max=k_max)
        row = {
            "panel_block": panel_block,
            "year": year,
            "return_component": return_component,
            "K_hat": int(max(1, K_hat)),
            "g_fn": g_fn,
            "gamma": float(gamma),
        }
        for idx, value in enumerate(er[:k_max], start=1):
            row[f"er_{idx}"] = float(value)
        rows.append(row)
    return rows


def _table_ii_rows_for_year(
    year: int,
    balanced_year: HFPanel,
    unbalanced_year: HFPanel,
    balanced_cont: np.ndarray,
    balanced_jump: np.ndarray,
    unbalanced_cont: np.ndarray,
    unbalanced_jump: np.ndarray,
    gamma: float,
    g_fn: str,
    k_max: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for component, bal_matrix, unb_matrix in [
        ("hf", balanced_year.R_5min_full, unbalanced_year.R_5min_full),
        ("continuous", balanced_cont, unbalanced_cont),
        ("jump", balanced_jump, unbalanced_jump),
    ]:
        bal_res = _panel_pca(bal_matrix, K=1, use_corr=True)
        unb_res = _panel_pca(unb_matrix, K=1, use_corr=True)
        K_bal, _ = perturbed_eigenvalue_ratio(bal_res.eigvals, g_fn=g_fn, N=balanced_year.N, gamma=gamma, K_max=k_max)
        K_unb, _ = perturbed_eigenvalue_ratio(unb_res.eigvals, g_fn=g_fn, N=unbalanced_year.N, gamma=gamma, K_max=k_max)
        K_use = max(1, min(K_bal, K_unb, bal_res.F.shape[1], unb_res.F.shape[1]))
        gc = generalized_correlations(bal_res.F[:, :K_use], unb_res.F[:, :K_use])
        row = {
            "year": int(year),
            "return_component": component,
            "balanced_n_symbols": int(balanced_year.N),
            "unbalanced_n_symbols": int(unbalanced_year.N),
            "K_balanced": int(max(1, K_bal)),
            "K_unbalanced": int(max(1, K_unb)),
            "gc_mean": float(np.nanmean(gc)) if len(gc) else np.nan,
        }
        for idx, value in enumerate(gc, start=1):
            row[f"gc_{idx}"] = float(value)
        rows.append(row)
    return rows


def _analysis_panel_key(panel_block: str) -> str:
    return "balanced" if panel_block == "Balanced panel" else "unbalanced"


def _panel_matrix_for_component(
    analysis: YearPanelAnalysis,
    panel_block: str,
    component: str,
    threshold: float,
) -> np.ndarray:
    panel_key = _analysis_panel_key(panel_block)
    panel = analysis.balanced_year if panel_key == "balanced" else analysis.unbalanced_year
    if component == "hf":
        return panel.R_5min_full
    decomp = analysis.decompositions[(panel_key, float(threshold))]
    if component == "continuous":
        return decomp.R_cont
    if component == "jump":
        return decomp.R_jump
    raise ValueError(f"Unknown return component: {component}")


def _cached_panel_jump_decomposition(
    analysis: YearPanelAnalysis,
    panel_block: str,
    threshold: float,
    retain_arrays: bool = True,
) -> YearJumpDecomposition:
    panel_key = _analysis_panel_key(panel_block)
    cache_key = (panel_key, float(threshold))
    cached = analysis.decompositions.get(cache_key)
    if cached is not None:
        return cached
    panel = analysis.balanced_year if panel_key == "balanced" else analysis.unbalanced_year
    scratch_root = None if analysis.scratch_root is None else analysis.scratch_root / f"year_{analysis.year}" / f"{panel_key}_{threshold:.1f}"
    memmap_ctx = MemmapContext(scratch_root=scratch_root, prefix=f"{panel_key}_{threshold:.1f}", cleanup=not retain_arrays)
    stats: Dict[str, float] = {}
    R_cont, R_jump = detect_jumps(panel, a=float(threshold), memmap_ctx=memmap_ctx, stats_out=stats)
    if not stats:
        stats = jump_summary_stats(R_cont, R_jump)
    if retain_arrays:
        cont_store = R_cont
        jump_store = R_jump
    else:
        cont_store = np.zeros((0, 0), dtype=np.float64)
        jump_store = np.zeros((0, 0), dtype=np.float64)
        memmap_ctx.cleanup_files()
    cached = YearJumpDecomposition(
        threshold=float(threshold),
        R_cont=cont_store,
        R_jump=jump_store,
        stats=stats,
        arrays_retained=bool(retain_arrays),
    )
    analysis.decompositions[cache_key] = cached
    return cached


def _cached_panel_pca_from_analysis(
    analysis: YearPanelAnalysis,
    panel_block: str,
    component: str,
    threshold: float,
    K: int,
    use_corr: bool = True,
) -> PCAResult:
    panel_key = _analysis_panel_key(panel_block)
    cache_key = (panel_key, component, float(threshold))
    cached = analysis.pca_cache.get(cache_key)
    required_k = max(1, int(K))
    if cached is not None and cached.F.shape[1] >= required_k:
        return cached
    matrix = _panel_matrix_for_component(analysis, panel_block=panel_block, component=component, threshold=threshold)
    res = _panel_pca(matrix, K=required_k, use_corr=use_corr)
    analysis.pca_cache[cache_key] = res
    return res


def _cached_explained_variation_from_analysis(
    analysis: YearPanelAnalysis,
    panel_block: str,
    component: str,
    threshold: float,
    K: int,
    use_corr: bool = True,
    matrix_override: Optional[np.ndarray] = None,
) -> float:
    panel_key = _analysis_panel_key(panel_block)
    K_eff = max(0, int(K))
    cache_key = (panel_key, component, float(threshold), K_eff)
    cached = analysis.explained_cache.get(cache_key)
    if cached is not None:
        return cached
    matrix = matrix_override
    if matrix is None:
        matrix = _panel_matrix_for_component(analysis, panel_block=panel_block, component=component, threshold=threshold)
    value = _panel_explained_variation_top_k(matrix, K=K_eff, use_corr=use_corr)
    analysis.explained_cache[cache_key] = value
    return value


def _build_year_panel_analysis(
    proc_root: Path,
    year: int,
    year_dates: Sequence[str],
    tickers: Sequence[str],
    max_stocks: Optional[int],
    panel_workers: int = 1,
    scratch_root: Optional[Path] = None,
) -> YearPanelAnalysis:
    timings: Dict[str, float] = {}
    t_panel = time.perf_counter()
    balanced_year = load_proc_5min_panel(
        proc_root=proc_root,
        years=[year],
        max_stocks=max_stocks,
    )
    unbalanced_year = _build_unbalanced_year_5min_panel(
        proc_root=proc_root,
        year=year,
        year_dates=year_dates,
        tickers=tickers,
        panel_workers=panel_workers,
        memmap_ctx=MemmapContext(
            scratch_root=None if scratch_root is None else scratch_root / f"year_{year}",
            prefix=f"unbalanced_{year}",
        ),
    )
    timings["panel_build_sec"] = time.perf_counter() - t_panel
    return YearPanelAnalysis(
        year=int(year),
        balanced_year=balanced_year,
        unbalanced_year=unbalanced_year,
        stage_timings=timings,
        scratch_root=scratch_root,
    )


def _table_i_rows_from_analysis(
    analysis: YearPanelAnalysis,
    thresholds: Sequence[float],
    progress_hook: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for panel_block in ("Balanced panel", "Unbalanced panel"):
        for threshold in thresholds:
            retain_arrays = bool(abs(float(threshold) - 3.0) < 1e-12)
            t_jump = time.perf_counter()
            _cached_panel_jump_decomposition(analysis, panel_block, float(threshold), retain_arrays=retain_arrays)
            analysis.stage_timings["jump_decompose_sec"] = analysis.stage_timings.get("jump_decompose_sec", 0.0) + (time.perf_counter() - t_jump)
            if progress_hook is not None:
                progress_hook(
                    "jump_threshold_done",
                    year=int(analysis.year),
                    panel_block=panel_block,
                    threshold_a=float(threshold),
                    message="jump decomposition ready",
                )

    for panel_block in ("Balanced panel", "Unbalanced panel"):
        panel = analysis.balanced_year if panel_block == "Balanced panel" else analysis.unbalanced_year
        year = int(panel.dates[0].year) if panel.dates else analysis.year
        for threshold in thresholds:
            decomp = analysis.decompositions[(_analysis_panel_key(panel_block), float(threshold))]
            if decomp.arrays_retained and decomp.R_jump.size and decomp.R_cont.size:
                jump_matrix = decomp.R_jump
                cont_matrix = decomp.R_cont
            else:
                scratch_root = None if analysis.scratch_root is None else analysis.scratch_root / f"year_{analysis.year}" / f"tmp_{panel_block}_{threshold:.1f}"
                tmp_ctx = MemmapContext(scratch_root=scratch_root, prefix=f"tmp_{threshold:.1f}", cleanup=True)
                panel_use = analysis.balanced_year if panel_block == "Balanced panel" else analysis.unbalanced_year
                t_jump = time.perf_counter()
                cont_matrix, jump_matrix = detect_jumps(panel_use, a=float(threshold), memmap_ctx=tmp_ctx)
                analysis.stage_timings["jump_decompose_sec"] = analysis.stage_timings.get("jump_decompose_sec", 0.0) + (time.perf_counter() - t_jump)
                if progress_hook is not None:
                    progress_hook(
                        "jump_threshold_done",
                        year=int(analysis.year),
                        panel_block=panel_block,
                        threshold_a=float(threshold),
                        message="temporary jump decomposition ready",
                    )
            t_pca = time.perf_counter()
            jump_ev = _cached_explained_variation_from_analysis(
                analysis, panel_block=panel_block, component="jump", threshold=float(threshold), K=1, use_corr=True, matrix_override=jump_matrix
            )
            cont_ev = _cached_explained_variation_from_analysis(
                analysis, panel_block=panel_block, component="continuous", threshold=float(threshold), K=min(4, panel.N), use_corr=True, matrix_override=cont_matrix
            )
            analysis.stage_timings["pca_sec"] = analysis.stage_timings.get("pca_sec", 0.0) + (time.perf_counter() - t_pca)
            rows.append(
                {
                    "panel_block": panel_block,
                    "year": year,
                    "threshold_a": float(threshold),
                    "n_symbols": int(panel.N),
                    "n_days": int(panel.D),
                    "frac_jump_increments": decomp.stats["frac_jump_increments"],
                    "frac_qv_explained_by_jumps": decomp.stats["frac_qv_explained_by_jumps"],
                    "jump_corr_explained_by_first_factor": jump_ev,
                    "continuous_corr_explained_by_first_four": cont_ev,
                }
            )
            if progress_hook is not None:
                progress_hook(
                    "pca_done",
                    year=int(analysis.year),
                    panel_block=panel_block,
                    component="table_i_metrics",
                    threshold_a=float(threshold),
                    message="Table I metrics ready",
                )
    return rows


def _factor_count_rows_from_analysis(
    analysis: YearPanelAnalysis,
    jump_a: float,
    k_max: int,
    gamma: float,
    g_fn: str,
    progress_hook: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for panel_block in ("Balanced panel", "Unbalanced panel"):
        t_jump = time.perf_counter()
        _cached_panel_jump_decomposition(analysis, panel_block, float(jump_a))
        analysis.stage_timings["jump_decompose_sec"] = analysis.stage_timings.get("jump_decompose_sec", 0.0) + (time.perf_counter() - t_jump)
        if progress_hook is not None:
            progress_hook(
                "jump_threshold_done",
                year=int(analysis.year),
                panel_block=panel_block,
                threshold_a=float(jump_a),
                message="factor-count jump decomposition ready",
            )

    for panel_block in ("Balanced panel", "Unbalanced panel"):
        panel = analysis.balanced_year if panel_block == "Balanced panel" else analysis.unbalanced_year
        year = int(panel.dates[0].year) if panel.dates else analysis.year
        for component in ("hf", "continuous", "jump"):
            t_pca = time.perf_counter()
            res = _cached_panel_pca_from_analysis(
                analysis,
                panel_block=panel_block,
                component=component,
                threshold=float(jump_a),
                K=min(k_max + 1, panel.N),
                use_corr=True,
            )
            K_hat, er = perturbed_eigenvalue_ratio(res.eigvals, g_fn=g_fn, N=panel.N, gamma=gamma, K_max=k_max)
            row = {
                "panel_block": panel_block,
                "year": year,
                "return_component": component,
                "K_hat": int(max(1, K_hat)),
                "g_fn": g_fn,
                "gamma": float(gamma),
            }
            for idx, value in enumerate(er[:k_max], start=1):
                row[f"er_{idx}"] = float(value)
            rows.append(row)
            analysis.stage_timings["pca_sec"] = analysis.stage_timings.get("pca_sec", 0.0) + (time.perf_counter() - t_pca)
            if progress_hook is not None:
                progress_hook(
                    "pca_done",
                    year=int(analysis.year),
                    panel_block=panel_block,
                    component=component,
                    threshold_a=float(jump_a),
                    message="factor-count PCA ready",
                )
    return rows


def _table_ii_rows_from_analysis(
    analysis: YearPanelAnalysis,
    jump_a: float,
    gamma: float,
    g_fn: str,
    k_max: int,
    progress_hook: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    t_jump = time.perf_counter()
    _cached_panel_jump_decomposition(analysis, "Balanced panel", float(jump_a))
    _cached_panel_jump_decomposition(analysis, "Unbalanced panel", float(jump_a))
    analysis.stage_timings["jump_decompose_sec"] = analysis.stage_timings.get("jump_decompose_sec", 0.0) + (time.perf_counter() - t_jump)
    if progress_hook is not None:
        progress_hook(
            "jump_threshold_done",
            year=int(analysis.year),
            panel_block="Balanced panel",
            threshold_a=float(jump_a),
            message="Table II balanced jump decomposition ready",
        )
        progress_hook(
            "jump_threshold_done",
            year=int(analysis.year),
            panel_block="Unbalanced panel",
            threshold_a=float(jump_a),
            message="Table II unbalanced jump decomposition ready",
        )

    balanced_year = analysis.balanced_year
    unbalanced_year = analysis.unbalanced_year
    for component in ("hf", "continuous", "jump"):
        t_pca = time.perf_counter()
        bal_res = _cached_panel_pca_from_analysis(
            analysis,
            panel_block="Balanced panel",
            component=component,
            threshold=float(jump_a),
            K=min(k_max + 1, balanced_year.N),
            use_corr=True,
        )
        unb_res = _cached_panel_pca_from_analysis(
            analysis,
            panel_block="Unbalanced panel",
            component=component,
            threshold=float(jump_a),
            K=min(k_max + 1, unbalanced_year.N),
            use_corr=True,
        )
        K_bal, _ = perturbed_eigenvalue_ratio(bal_res.eigvals, g_fn=g_fn, N=balanced_year.N, gamma=gamma, K_max=k_max)
        K_unb, _ = perturbed_eigenvalue_ratio(unb_res.eigvals, g_fn=g_fn, N=unbalanced_year.N, gamma=gamma, K_max=k_max)
        K_use = max(1, min(K_bal, K_unb, bal_res.F.shape[1], unb_res.F.shape[1]))
        gc = generalized_correlations(bal_res.F[:, :K_use], unb_res.F[:, :K_use])
        row = {
            "year": int(analysis.year),
            "return_component": component,
            "balanced_n_symbols": int(balanced_year.N),
            "unbalanced_n_symbols": int(unbalanced_year.N),
            "K_balanced": int(max(1, K_bal)),
            "K_unbalanced": int(max(1, K_unb)),
            "gc_mean": float(np.nanmean(gc)) if len(gc) else np.nan,
        }
        for idx, value in enumerate(gc, start=1):
            row[f"gc_{idx}"] = float(value)
        rows.append(row)
        analysis.stage_timings["pca_sec"] = analysis.stage_timings.get("pca_sec", 0.0) + (time.perf_counter() - t_pca)
        if progress_hook is not None:
            progress_hook(
                "pca_done",
                year=int(analysis.year),
                panel_block="Balanced/Unbalanced",
                component=component,
                threshold_a=float(jump_a),
                message="Table II PCA ready",
            )
    return rows


def yearly_paper_metrics_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    proc_root = Path(task["proc_root"])
    year = int(task["year"])
    thresholds = tuple(float(value) for value in task["thresholds"])
    jump_a = float(task["jump_a"])
    k_max = int(task["k_max"])
    gamma = float(task["gamma"])
    g_fn = str(task["g_fn"])
    max_stocks = task.get("max_stocks")
    year_dates = list(task["year_dates"])
    tickers = list(task["tickers"])
    scratch_root_text = task.get("scratch_root")
    scratch_root = None if not scratch_root_text else Path(str(scratch_root_text))
    blas_threads = int(task.get("blas_threads", 1))
    panel_workers = max(1, int(task.get("panel_workers", 1)))
    progress_path = task.get("progress_path")
    run_started_unix = task.get("run_started_unix")
    progress_state = dict(task.get("progress_state") or {})

    def emit(event_type: str, **payload: Any) -> None:
        _emit_progress_event_to_path(
            progress_path,
            event_type,
            run_started_unix=run_started_unix,
            state=progress_state,
            **payload,
        )

    analysis = _build_year_panel_analysis(
        proc_root=proc_root,
        year=year,
        year_dates=year_dates,
        tickers=tickers,
        max_stocks=max_stocks,
        panel_workers=panel_workers,
        scratch_root=scratch_root,
    )
    emit(
        "panel_loaded",
        stage="paper_tables",
        year=year,
        n_days=int(len(year_dates)),
        n_symbols=int(len(tickers)),
        message="balanced and unbalanced yearly panels loaded",
    )
    with _temporary_blas_thread_env(blas_threads):
        return {
            "year": year,
            "table_i_rows": _table_i_rows_from_analysis(analysis, thresholds, progress_hook=emit),
            "table_ii_rows": _table_ii_rows_from_analysis(
                analysis=analysis,
                jump_a=jump_a,
                gamma=gamma,
                g_fn=g_fn,
                k_max=k_max,
                progress_hook=emit,
            ),
            "factor_count_rows": _factor_count_rows_from_analysis(
                analysis=analysis,
                jump_a=jump_a,
                k_max=k_max,
                gamma=gamma,
                g_fn=g_fn,
                progress_hook=emit,
            ),
            "stage_timings": dict(analysis.stage_timings),
            "n_days": int(len(year_dates)),
            "n_symbols": int(len(tickers)),
        }


def build_yearly_paper_outputs(
    balanced_panel: HFPanel,
    proc_root: Path,
    universe: pd.DataFrame,
    thresholds: Sequence[float] = (3.0, 4.0, 4.5, 5.0),
    jump_a: float = 3.0,
    gamma: float = 0.08,
    g_fn: str = "median_N",
    k_max: int = 10,
    workers: Optional[int] = None,
    max_stocks: Optional[int] = None,
    runtime: Optional[RuntimeConfig] = None,
    progress: Optional[ProgressReporter] = None,
    resource_plan: Optional[ResourcePlan] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> YearlyPaperOutputs:
    years = sorted({date.year for date in balanced_panel.dates})
    summary = _universe_attr_summary(universe)
    if runtime is None:
        runtime = _build_runtime_config(DEFAULT_OUTPUT_ROOT, None, DEFAULT_PROGRESS_INTERVAL_SEC)
    if resource_plan is None:
        resource_plan = _build_paper_resource_plan(
            proc_root=proc_root,
            summary=summary,
            years=years,
            requested_workers=_normalize_worker_count(workers),
            requested_paper_workers=_normalize_worker_count(workers),
            requested_rolling_workers=_normalize_worker_count(workers),
            runtime=runtime,
            universe=universe,
            max_stocks=max_stocks,
        )
    base_tasks = [
        {
            "proc_root": str(proc_root),
            "year": int(year),
            "year_dates": _year_dates_from_universe_summary(summary, year),
            "tickers": _select_unbalanced_symbols_for_year(universe, year, max_stocks=max_stocks),
            "thresholds": [float(value) for value in thresholds],
            "jump_a": float(jump_a),
            "k_max": int(k_max),
            "gamma": float(gamma),
            "g_fn": str(g_fn),
            "max_stocks": None if max_stocks is None else int(max_stocks),
            "scratch_root": str(runtime.scratch_root),
            "blas_threads": int(resource_plan.paper_blas_threads),
            "progress_path": None if progress is None else str(progress.path),
            "progress_state": {} if progress is None else progress.snapshot_state(),
            "run_started_unix": None if progress is None else float(progress._start_time_unix),
        }
        for year in years
    ]
    year_estimates = {int(item["year"]): dict(item) for item in resource_plan.paper_year_estimates}
    worker_count = _safe_process_pool_worker_count("paper tables", int(resource_plan.paper_workers_effective))
    yearly_results: List[Dict[str, Any]] = []
    if checkpoint_manager is not None:
        checkpoint_manager.mark_paper_plan(years)
    completed_years_set = (
        {int(year) for year in checkpoint_manager.completed_paper_years() if int(year) in {int(v) for v in years}}
        if checkpoint_manager is not None
        else set()
    )
    completed_years = len(completed_years_set)
    if progress is not None:
        progress.update_state(
            stage="paper_tables",
            total_years=len(years),
            completed_years=completed_years,
            paper_workers_effective=worker_count,
            memory_budget_gb=runtime.memory_budget_gb,
        )
        progress.event("stage_started", message="yearly paper outputs")

    pending_tasks: List[Dict[str, Any]] = []
    for task in base_tasks:
        year = int(task["year"])
        if year in completed_years_set:
            if progress is not None:
                progress.event(
                    "paper_year_skipped",
                    stage="paper_tables",
                    year=year,
                    completed_years=completed_years,
                    total_years=len(years),
                    message="reused completed paper checkpoint",
                )
            continue
        pending_tasks.append(task)

    def build_task_payload(task: Dict[str, Any], active_years: int) -> Dict[str, Any]:
        task_payload = dict(task)
        task_payload["panel_workers"] = _paper_panel_thread_count(runtime.logical_cpus, active_years=active_years)
        return task_payload

    def finalize_item(item: Dict[str, Any]) -> None:
        nonlocal completed_years
        year = int(item["year"])
        if checkpoint_manager is not None:
            _write_paper_year_checkpoint(checkpoint_manager.paper_year_dir(year), item)
            checkpoint_manager.mark_paper_year_complete(year)
            completed_years_set.add(year)
        else:
            yearly_results.append(item)
        completed_years += 1

    if worker_count == 1 or len(pending_tasks) <= 1:
        for task in pending_tasks:
            if progress is not None:
                estimate = year_estimates.get(int(task["year"]), {})
                progress.event(
                    "paper_year_started",
                    stage="paper_tables",
                    year=int(task["year"]),
                    memory_reserved_gb=float(estimate.get("peak_memory_gb", 0.0)),
                    completed_years=completed_years,
                    active_years=1,
                )
            item = yearly_paper_metrics_worker(build_task_payload(task, active_years=1))
            finalize_item(item)
            if progress is not None:
                progress.event(
                    "paper_year_finished",
                    stage="paper_tables",
                    year=int(task["year"]),
                    completed_years=completed_years,
                    total_years=len(years),
                    active_years=0,
                )
                progress.heartbeat(force=True, completed_years=completed_years, total_years=len(years), active_years=0, memory_reserved_gb=0.0)
    else:
        active_reserved_bytes = 0
        active_futures: Dict[Any, Tuple[Dict[str, Any], int]] = {}
        with _temporary_blas_thread_env(1):
            try:
                with _spawn_process_pool(worker_count) as executor:
                    while pending_tasks or active_futures:
                        launched = False
                        idx = 0
                        while idx < len(pending_tasks) and len(active_futures) < worker_count:
                            task = pending_tasks[idx]
                            estimate = year_estimates.get(int(task["year"]), {})
                            reserve_bytes = int(estimate.get("peak_memory_bytes", 0))
                            if active_reserved_bytes + reserve_bytes > runtime.memory_budget_bytes and active_futures:
                                idx += 1
                                continue
                            pending_tasks.pop(idx)
                            task_payload = build_task_payload(task, active_years=len(active_futures) + 1)
                            future = executor.submit(yearly_paper_metrics_worker, task_payload)
                            active_futures[future] = (task_payload, reserve_bytes)
                            active_reserved_bytes += reserve_bytes
                            launched = True
                            if progress is not None:
                                progress.event(
                                    "paper_year_started",
                                    stage="paper_tables",
                                    year=int(task["year"]),
                                    active_years=len(active_futures),
                                    completed_years=completed_years,
                                    total_years=len(years),
                                    memory_reserved_gb=_format_bytes_gb(active_reserved_bytes),
                                )
                        if not active_futures and pending_tasks and not launched:
                            task = pending_tasks.pop(0)
                            estimate = year_estimates.get(int(task["year"]), {})
                            reserve_bytes = int(estimate.get("peak_memory_bytes", 0))
                            task_payload = build_task_payload(task, active_years=1)
                            future = executor.submit(yearly_paper_metrics_worker, task_payload)
                            active_futures[future] = (task_payload, reserve_bytes)
                            active_reserved_bytes += reserve_bytes
                            if progress is not None:
                                progress.event(
                                    "paper_year_started",
                                    stage="paper_tables",
                                    year=int(task["year"]),
                                    active_years=len(active_futures),
                                    completed_years=completed_years,
                                    total_years=len(years),
                                    memory_reserved_gb=_format_bytes_gb(active_reserved_bytes),
                                )

                        progress and progress.heartbeat(
                            completed_years=completed_years,
                            total_years=len(years),
                            active_years=len(active_futures),
                            memory_reserved_gb=_format_bytes_gb(active_reserved_bytes),
                        )

                        if not active_futures:
                            continue

                        try:
                            done_future = next(
                                as_completed(
                                    list(active_futures.keys()),
                                    timeout=progress.interval_sec if progress is not None else None,
                                )
                            )
                        except FuturesTimeoutError:
                            continue
                        task, reserve_bytes = active_futures.pop(done_future)
                        try:
                            item = done_future.result()
                        except Exception as exc:
                            peak_gb = _format_bytes_gb(reserve_bytes)
                            raise RuntimeError(
                                f"paper year {task['year']} failed; n_days={len(task['year_dates'])}, "
                                f"n_symbols={len(task['tickers'])}, peak_memory_gb~={peak_gb:.2f}, "
                                f"active_workers={len(active_futures) + 1}, budget_gb={runtime.memory_budget_gb:.2f}"
                            ) from exc
                        active_reserved_bytes -= reserve_bytes
                        finalize_item(item)
                        if progress is not None:
                            progress.event(
                                "paper_year_finished",
                                stage="paper_tables",
                                year=int(task["year"]),
                                completed_years=completed_years,
                                total_years=len(years),
                                active_years=len(active_futures),
                                memory_reserved_gb=_format_bytes_gb(active_reserved_bytes),
                            )
            except Exception as exc:
                if not _is_spawn_context_failure(exc):
                    raise
                _warn_parallel_fallback("paper tables", str(exc))
                pending_years = (
                    [
                        task
                        for task in base_tasks
                        if int(task["year"]) not in completed_years_set
                    ]
                )
                for task in pending_years:
                    item = yearly_paper_metrics_worker(build_task_payload(task, active_years=1))
                    finalize_item(item)
    if progress is not None:
        progress.event("stage_finished", stage="paper_tables", message="yearly paper outputs complete")
    if checkpoint_manager is not None:
        return _assemble_yearly_paper_outputs_from_checkpoints(checkpoint_manager, years)
    yearly_results.sort(key=lambda item: int(item["year"]))
    return _assemble_yearly_paper_outputs_from_items(yearly_results)


def build_paper_table_i(
    balanced_panel: HFPanel,
    proc_root: Path,
    universe: pd.DataFrame,
    thresholds: Sequence[float] = (3.0, 4.0, 4.5, 5.0),
    workers: Optional[int] = None,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """Public wrapper for the canonical Table I construction path."""
    return build_yearly_paper_outputs(
        balanced_panel=balanced_panel,
        proc_root=proc_root,
        universe=universe,
        thresholds=thresholds,
        jump_a=3.0,
        gamma=0.08,
        g_fn="median_N",
        k_max=10,
        workers=workers,
        max_stocks=max_stocks,
    ).table_i


def build_paper_table_ii(
    balanced_panel: HFPanel,
    proc_root: Path,
    universe: pd.DataFrame,
    jump_a: float = 3.0,
    gamma: float = 0.08,
    g_fn: str = "median_N",
    k_max: int = 10,
    workers: Optional[int] = None,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """Build paper-style Table II via the shared yearly analysis cache."""
    return build_yearly_paper_outputs(
        balanced_panel=balanced_panel,
        proc_root=proc_root,
        universe=universe,
        thresholds=(3.0, 4.0, 4.5, 5.0),
        jump_a=jump_a,
        gamma=gamma,
        g_fn=g_fn,
        k_max=k_max,
        workers=workers,
        max_stocks=max_stocks,
    ).table_ii


def build_paper_table_iii() -> pd.DataFrame:
    """Document the inputs still missing for the paper's Table III."""
    return pd.DataFrame(
        [
            {
                "required_dataset": "industry_portfolios",
                "status": "external_data_required",
                "paper_role": "Industry portfolio returns used for generalized-correlation tests.",
                "expected_loader": "load_test_asset_csv(...)",
                "notes": "Current repository does not ship the paper's industry test-asset returns.",
            },
            {
                "required_dataset": "ffc_factors",
                "status": "external_data_required",
                "paper_role": "Fama-French-Carhart factors used in the paper's factor-space comparison.",
                "expected_loader": "load_external_factor_csv(..., freq='daily')",
                "notes": "Provide daily factor CSV or Parquet with a documented schema to replace this placeholder.",
            },
        ]
    )


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
            "hf": pca_factors(year_panel.R_5min_full, K=1, use_corr=True).eigvals,
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


def build_paper_factor_counts_comparison(
    balanced_panel: HFPanel,
    proc_root: Path,
    universe: pd.DataFrame,
    k_max: int,
    gamma: float,
    g_fn: str,
    jump_a: float,
    workers: Optional[int] = None,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """Build paper-style factor-count diagnostics via the shared yearly analysis cache."""
    return build_yearly_paper_outputs(
        balanced_panel=balanced_panel,
        proc_root=proc_root,
        universe=universe,
        thresholds=(3.0, 4.0, 4.5, 5.0),
        jump_a=jump_a,
        gamma=gamma,
        g_fn=g_fn,
        k_max=k_max,
        workers=workers,
        max_stocks=max_stocks,
    ).factor_counts


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
        F_daily = pipe.F_cont_daily_total if pipe.F_cont_daily_total is not None else pipe.F_cont_daily_intra + pipe.F_cont_daily_night
        for idx in range(pipe.F_cont_daily_intra.shape[1]):
            rows.append({
                "portfolio": f"{idx + 1}. Continuous PCA Factor",
                "SR_intraday": float(np.nanmean(pipe.F_cont_daily_intra[:, idx]) / (np.nanstd(pipe.F_cont_daily_intra[:, idx], ddof=1) or np.nan) * scale),
                "SR_overnight": float(np.nanmean(pipe.F_cont_daily_night[:, idx]) / (np.nanstd(pipe.F_cont_daily_night[:, idx], ddof=1) or np.nan) * scale),
                "SR_daily": float(np.nanmean(F_daily[:, idx]) / (np.nanstd(F_daily[:, idx], ddof=1) or np.nan) * scale),
            })
    return pd.DataFrame(rows)


def build_paper_table_iv(rolling_gc_df: pd.DataFrame, rolling_explained_df: pd.DataFrame) -> pd.DataFrame:
    """构造论文 Table IV 的时间变化分解汇总。"""
    if rolling_gc_df.empty and rolling_explained_df.empty:
        return pd.DataFrame(columns=["metric", "mean", "median", "min", "max"])

    rows: List[Dict[str, Any]] = []
    gc_cols = [col for col in rolling_gc_df.columns if col.startswith("gc_")]
    for col in gc_cols:
        rows.append({
            "metric": col,
            "mean": float(rolling_gc_df[col].mean()),
            "median": float(rolling_gc_df[col].median()),
            "min": float(rolling_gc_df[col].min()),
            "max": float(rolling_gc_df[col].max()),
        })
    if not rolling_explained_df.empty and "explained_variation" in rolling_explained_df.columns:
        series = rolling_explained_df["explained_variation"]
        rows.append({
            "metric": "explained_variation",
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
        })
    return pd.DataFrame(rows)


def build_paper_table_v(pipe: PelgerPipeline) -> pd.DataFrame:
    """构造论文 Table V 的 intraday / overnight / daily Sharpe 汇总。"""
    return build_factor_sharpe_table(pipe)


def build_replication_coverage_report() -> pd.DataFrame:
    """列出论文表图在当前 A 股数据条件下的复现状态。"""
    rows = [
        ("Table I", "Summary Statistics for Continuous and Jump Returns", "implemented_adapted", "已按 balanced / unbalanced 两块输出年度跳跃与连续收益统计。"),
        ("Table II", "Balanced and Unbalanced Panel Results", "implemented_adapted", "已按 balanced vs unbalanced 输出 factor-space generalized correlations。"),
        ("Table III", "Generalized Correlations with Industry and FFC Factors", "external_data_required", "需要行业组合收益与 Fama-French-Carhart 因子 CSV。"),
        ("Table IV", "Time-Variation Decomposition across Frequencies", "implemented_adapted", "已输出 rolling GC / explained variation 的时间变化汇总。"),
        ("Table V", "Intraday / Overnight / Daily Sharpe Ratios", "implemented_adapted", "已输出连续 PCA 因子的日内、隔夜、日度 Sharpe。"),
        ("Figure 1", "Number of HF Factors, Unbalanced Panel", "implemented_adapted", "已输出非平衡面板的 perturbed eigenvalue ratio 诊断图。"),
        ("Figure 2", "Number of HF Factors, Balanced Panel", "implemented_adapted", "已输出严格平衡面板的 perturbed eigenvalue ratio 诊断图。"),
        ("Figure 3", "Proxy Factor Portfolio Weights", "implemented_adapted", "已输出 proxy factors 权重热图。"),
        ("Figure 4", "Continuous PCA Factor Portfolio Weights", "implemented_adapted", "已输出连续 PCA 因子权重热图。"),
        ("Figure 5", "Monthly PCA Factor Portfolio Weights", "implemented_adapted", "已输出月频 PCA 因子权重热图。"),
        ("Figure 6", "Time Variation in Loadings", "implemented_adapted", "已输出滚动 GC 稳定性曲线。"),
        ("Figure 7", "Locally Estimated Continuous Factors", "implemented_adapted", "已输出局部连续因子与全局因子的 GC 曲线。"),
        ("Figure 8", "Time-Varying Portfolio Weights", "implemented_adapted", "已输出滚动窗口下 top 权重股票变化。"),
        ("Figure 9", "Time-Varying Explained Variation", "implemented_adapted", "已输出滚动解释度。"),
        ("Figure 10", "Factor-Structure Time Variation Decomposition", "implemented_adapted", "已输出 GC 与解释度的结构分解图。"),
        ("Figure 11", "Continuous Factor-Structure Decomposition", "implemented_adapted", "已输出连续因子的结构分解图。"),
        ("Figure 12", "Expected Intraday and Overnight Returns", "implemented_adapted", "已输出连续 PCA 因子的日内/隔夜/日度平均收益。"),
        ("Figure 13", "Cumulative Factor Returns", "implemented_adapted", "已输出连续 PCA 因子的累计收益。"),
        ("Figure 14", "Asset Pricing of Industry Portfolios", "external_data_required", "需要行业组合收益；当前仅输出占位图。"),
        ("Figure 15", "Asset Pricing of Size- and Value-Sorted Portfolios", "external_data_required", "需要 size/value 测试资产组合；当前仅输出占位图。"),
    ]
    return pd.DataFrame(rows, columns=["paper_item", "paper_content", "status", "notes"])


def build_weight_tables(pipe: PelgerPipeline, panel: HFPanel, top_n: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """导出连续 PCA 权重和 proxy 权重，支持论文 Figure 3-4。"""
    pca_display = pipe.pca_cont_display if pipe.pca_cont_display is not None else pipe.pca_cont
    W = factor_portfolio_weights(pca_display)
    W_proxy, _ = build_proxy_factors(W, pipe.R_cont)
    k_count = min(W.shape[1], DISPLAY_CONTINUOUS_FACTOR_COUNT)
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
    F_intra = pipe.F_cont_display_daily_intra if pipe.F_cont_display_daily_intra is not None else pipe.F_cont_daily_intra
    F_night = pipe.F_cont_display_daily_night if pipe.F_cont_display_daily_night is not None else pipe.F_cont_daily_night
    F_daily = (
        pipe.F_cont_display_daily_total
        if pipe.F_cont_display_daily_total is not None
        else (pipe.F_cont_daily_total if pipe.F_cont_daily_total is not None else F_intra + F_night)
    )
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


def refresh_replication_result_views(result: ReplicationResult) -> ReplicationResult:
    """Refresh lightweight presentation-layer tables from an existing ReplicationResult.

    This is intentionally cheap: it only recomputes display-facing derivatives from the
    already-loaded balanced panel and continuous-return decomposition.
    """
    pipe = result.pipeline
    panel = result.panel
    if getattr(pipe, "R_cont", None) is None or getattr(pipe, "pca_cont", None) is None:
        return result

    display_k = min(max(1, int(DISPLAY_CONTINUOUS_FACTOR_COUNT)), int(panel.N))
    current_display = getattr(pipe, "pca_cont_display", None)
    if current_display is None or int(current_display.Lambda.shape[1]) != display_k:
        if int(pipe.pca_cont.Lambda.shape[1]) >= display_k:
            pipe.pca_cont_display = _truncate_pca_result(pipe.pca_cont, display_k)
        else:
            pipe.pca_cont_display = pca_factors(pipe.R_cont, K=display_k, use_corr=pipe.use_corr)

    W_display = factor_portfolio_weights(pipe.pca_cont_display)
    pipe.F_cont_display_daily_intra = panel.R_intra @ W_display
    pipe.F_cont_display_daily_night = panel.R_night @ W_display
    pipe.F_cont_display_daily_total = panel.R_daily @ W_display

    result.pca_weights, result.proxy_weights = build_weight_tables(pipe, panel)
    result.factor_return_summary, result.cumulative_factor_returns = build_factor_return_tables(pipe, panel)
    if result.monthly_pca_weights.empty:
        result.monthly_pca_weights = build_monthly_pca_weights(panel)
    return result


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
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    return True


def _plot_status_row(figure_id: str, title: str, output_path: Path, status: str, data_source: str, notes: str) -> Dict[str, str]:
    try:
        figure_number = int(str(figure_id).split("_")[-1])
    except Exception:
        figure_number = -1
    return {
        "figure_number": figure_number,
        "figure_id": figure_id,
        "title": title,
        "file_path": str(output_path),
        "status": status,
        "data_source": data_source,
        "notes": notes,
    }


def _copy_alias_files(source_path: Path, alias_paths: Sequence[Path]) -> None:
    """Copy one exported file to one or more compatibility alias paths."""
    for alias_path in alias_paths:
        if source_path.resolve() == alias_path.resolve():
            continue
        _atomic_copy_file(source_path, alias_path)


def _write_csv_with_aliases(df: pd.DataFrame, canonical_path: Path, alias_paths: Optional[Sequence[Path]] = None) -> None:
    """Write one canonical CSV and then mirror it to compatibility aliases."""
    _atomic_to_csv(df, canonical_path, index=False, encoding="utf-8-sig")
    _copy_alias_files(canonical_path, alias_paths or [])


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
    _atomic_save_figure(fig, output_path, dpi=160)
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
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)


def _save_cumulative_factor_grid_plot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    ylabel: str = "Cumulative log return",
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    factors = sorted(int(value) for value in plot_df["factor"].dropna().unique().tolist())
    ncols = 1 if len(factors) <= 1 else 2
    nrows = max(1, (len(factors) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    line_specs = [
        ("cum_intraday", "Intraday"),
        ("cum_overnight", "Overnight"),
        ("cum_daily", "Daily"),
    ]

    for ax, factor in zip(axes_arr, factors):
        factor_df = plot_df.loc[plot_df["factor"].eq(factor)].sort_values("date")
        for column, label in line_specs:
            if column in factor_df.columns:
                ax.plot(factor_df["date"], factor_df[column], label=label, linewidth=1.3)
        ax.set_title(f"Factor {factor}")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    for ax in axes_arr[len(factors):]:
        ax.set_visible(False)

    if factors:
        axes_arr[0].legend(loc="best", fontsize=8)
    for ax in axes_arr[: len(factors)]:
        ax.set_xlabel("date")
    fig.suptitle(title)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _atomic_save_figure(fig, output_path, dpi=160)
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
    _atomic_save_figure(fig, output_path, dpi=160)
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
    _atomic_save_figure(fig, output_path, dpi=160)
    plt.close(fig)


def export_all_paper_figures(
    result: ReplicationResult,
    figures_dir: Path,
    rolling_gc_df: pd.DataFrame,
    rolling_explained_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Export canonical paper-numbered figures and record generation status."""
    exported: Dict[str, str] = {}
    status_rows: List[Dict[str, str]] = []
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_specs = {
        1: ("Figure_1", "Figure_1_number_of_hf_factors_unbalanced.png", "Figure 1. Number of High-Frequency Factors, Unbalanced Panel"),
        2: ("Figure_2", "Figure_2_number_of_hf_factors_balanced.png", "Figure 2. Number of High-Frequency Factors, Balanced Panel"),
        3: ("Figure_3", "Figure_3_proxy_factor_portfolio_weights.png", "Figure 3. Proxy Factor Portfolio Weights"),
        4: ("Figure_4", "Figure_4_continuous_pca_factor_portfolio_weights.png", "Figure 4. Continuous PCA Factor Portfolio Weights"),
        5: ("Figure_5", "Figure_5_monthly_pca_factor_portfolio_weights.png", "Figure 5. Monthly PCA Factor Portfolio Weights"),
        6: ("Figure_6", "Figure_6_time_variation_in_loadings.png", "Figure 6. Time Variation in Loadings"),
        7: ("Figure_7", "Figure_7_locally_estimated_continuous_factors.png", "Figure 7. Locally Estimated Continuous Factors"),
        8: ("Figure_8", "Figure_8_time_varying_portfolio_weights.png", "Figure 8. Time-Varying Portfolio Weights"),
        9: ("Figure_9", "Figure_9_time_varying_explained_variation.png", "Figure 9. Time-Varying Explained Variation"),
        10: ("Figure_10", "Figure_10_factor_structure_time_variation_decomposition.png", "Figure 10. Factor-Structure Time Variation Decomposition"),
        11: ("Figure_11", "Figure_11_continuous_factor_structure_decomposition.png", "Figure 11. Continuous Factor-Structure Decomposition"),
        12: ("Figure_12", "Figure_12_expected_intraday_and_overnight_returns.png", "Figure 12. Expected Intraday and Overnight Returns"),
        13: ("Figure_13", "Figure_13_cumulative_factor_returns.png", "Figure 13. Cumulative Factor Returns"),
        14: ("Figure_14", "Figure_14_asset_pricing_of_industry_portfolios.png", "Figure 14. Asset Pricing of Industry Portfolios"),
        15: ("Figure_15", "Figure_15_asset_pricing_of_size_and_value_sorted_portfolios.png", "Figure 15. Asset Pricing of Size- and Value-Sorted Portfolios"),
    }

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        for _, (figure_id, file_name, title) in figure_specs.items():
            output_path = figures_dir / file_name
            status_rows.append(_plot_status_row(figure_id, title, output_path, "error", "matplotlib", repr(exc)))
        return pd.DataFrame(status_rows).sort_values("figure_number").reset_index(drop=True), exported

    def run_plot(figure_number: int, data_source: str, plot_func, notes: str = "") -> None:
        figure_id, file_name, title = figure_specs[figure_number]
        output_path = figures_dir / file_name
        try:
            plot_func(output_path)
            status = "placeholder" if "external data required" in notes.lower() else "generated"
            status_rows.append(_plot_status_row(figure_id, title, output_path, status, data_source, notes))
            exported[figure_id] = str(output_path)
        except Exception as exc:
            _save_placeholder_figure(output_path, title, f"Plot generation failed: {exc!r}")
            status_rows.append(_plot_status_row(figure_id, title, output_path, "error", data_source, repr(exc)))
            exported[figure_id] = str(output_path)

    def _save_er_panel(output_path: Path, panel_block: str, title: str) -> None:
        df = result.paper_factor_counts.copy()
        if df.empty:
            _save_placeholder_figure(output_path, title, "No factor-count diagnostics are available.")
            return
        df = df.loc[df["panel_block"].eq(panel_block) & df["return_component"].eq("hf")].copy()
        er_cols = [col for col in df.columns if col.startswith("er_")]
        if df.empty or not er_cols:
            _save_placeholder_figure(output_path, title, f"No HF perturbed eigenvalue-ratio data are available for {panel_block.lower()}.")
            return
        x = np.arange(1, len(er_cols) + 1)
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for _, row in df.sort_values("year").iterrows():
            y = [float(row[col]) for col in er_cols]
            label = f"{int(row['year'])} (K={int(row['K_hat'])})"
            ax.plot(x, y, marker="o", linewidth=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel("Perturbed eigenvalue ratio")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        _atomic_save_figure(fig, output_path, dpi=160)
        plt.close(fig)

    def weight_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
        if df.empty:
            _save_placeholder_figure(output_path, title, "No portfolio-weight data are available.")
            return
        pivot = df.pivot_table(index="factor", columns="symbol", values="weight", aggfunc="first").fillna(0.0)
        _save_heatmap(pivot.to_numpy(), pivot.columns.tolist(), [f"Factor {idx}" for idx in pivot.index], title, output_path)

    run_plot(1, "paper_factor_count_diagnostics", lambda path: _save_er_panel(path, "Unbalanced panel", figure_specs[1][2]))
    run_plot(2, "paper_factor_count_diagnostics", lambda path: _save_er_panel(path, "Balanced panel", figure_specs[2][2]))
    run_plot(3, "proxy_weights", lambda path: weight_heatmap(result.proxy_weights, figure_specs[3][2], path))
    run_plot(4, "pca_weights", lambda path: weight_heatmap(result.pca_weights, figure_specs[4][2], path))
    run_plot(5, "monthly_pca_weights", lambda path: weight_heatmap(result.monthly_pca_weights, figure_specs[5][2], path))

    gc_cols = [col for col in rolling_gc_df.columns if col.startswith("gc_")]
    run_plot(6, "rolling_gc", lambda path: _save_line_plot(rolling_gc_df, "window_index", gc_cols, figure_specs[6][2], path, ylabel="Generalized correlation"))
    run_plot(7, "rolling_gc", lambda path: _save_line_plot(rolling_gc_df, "window_index", gc_cols[: min(4, len(gc_cols))], figure_specs[7][2], path, ylabel="Generalized correlation"))

    def fig08(output_path: Path) -> None:
        if result.rolling_weight_summary.empty:
            _save_placeholder_figure(output_path, figure_specs[8][2], "No rolling weight summary is available.")
            return
        pivot = result.rolling_weight_summary.pivot_table(index="start_day", columns="symbol", values="weight_factor_1", aggfunc="first").reset_index()
        y_cols = [col for col in pivot.columns if col != "start_day"]
        _save_line_plot(pivot, "start_day", y_cols, figure_specs[8][2], output_path, ylabel="Factor 1 weight")

    def fig10(output_path: Path) -> None:
        if rolling_gc_df.empty:
            _save_placeholder_figure(output_path, figure_specs[10][2], "No rolling generalized-correlation data are available.")
            return
        df = rolling_gc_df.copy()
        df["avg_gc"] = df[gc_cols].mean(axis=1) if gc_cols else np.nan
        if not rolling_explained_df.empty:
            df = df.merge(rolling_explained_df, on="window_index", how="left")
        _save_line_plot(df, "window_index", [col for col in ["avg_gc", "explained_variation"] if col in df.columns], figure_specs[10][2], output_path)

    def fig11(output_path: Path) -> None:
        if rolling_gc_df.empty:
            _save_placeholder_figure(output_path, figure_specs[11][2], "No rolling generalized-correlation data are available.")
            return
        df = rolling_gc_df.copy()
        df["min_gc"] = df[gc_cols].min(axis=1) if gc_cols else np.nan
        df["mean_gc"] = df[gc_cols].mean(axis=1) if gc_cols else np.nan
        _save_line_plot(df, "window_index", ["min_gc", "mean_gc"], figure_specs[11][2], output_path, ylabel="Generalized correlation")

    def fig12(output_path: Path) -> None:
        if result.factor_return_summary.empty:
            _save_placeholder_figure(output_path, figure_specs[12][2], "No factor return summary is available.")
            return
        long_df = result.factor_return_summary.melt(id_vars="factor", var_name="segment", value_name="mean_return")
        _save_bar_plot(long_df, "factor", "mean_return", figure_specs[12][2], output_path, group_col="segment", ylabel="Mean return")

    def fig13(output_path: Path) -> None:
        if result.cumulative_factor_returns.empty:
            _save_placeholder_figure(output_path, figure_specs[13][2], "No cumulative factor return data are available.")
            return
        _save_cumulative_factor_grid_plot(
            result.cumulative_factor_returns,
            figure_specs[13][2],
            output_path,
            ylabel="Cumulative log return",
        )

    run_plot(8, "rolling_weight_summary", fig08)
    run_plot(9, "rolling_explained_variation", lambda path: _save_line_plot(rolling_explained_df, "window_index", ["explained_variation"], figure_specs[9][2], path, ylabel="Explained variation"))
    run_plot(10, "rolling_gc_and_explained_variation", fig10)
    run_plot(11, "rolling_gc", fig11)
    run_plot(12, "factor_return_summary", fig12)
    run_plot(13, "cumulative_factor_returns", fig13)
    run_plot(14, "external_test_assets", lambda path: _save_placeholder_figure(path, figure_specs[14][2], "Industry portfolio returns are not available in the current repository. Provide external test-asset CSV files to replace this placeholder."), notes="External data required: industry portfolio returns")
    run_plot(15, "external_test_assets", lambda path: _save_placeholder_figure(path, figure_specs[15][2], "Size/value sorted portfolio returns are not available in the current repository. Provide external test-asset CSV files to replace this placeholder."), notes="External data required: size/value portfolio returns")

    plot_status = pd.DataFrame(status_rows).sort_values("figure_number").reset_index(drop=True)
    return plot_status, exported


def export_replication_outputs(
    result: ReplicationResult,
    save_plots: bool = True,
) -> Dict[str, str]:
    """Export canonical paper-numbered tables/figures plus compatibility aliases."""
    output_root = result.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    diagnostics_dir = output_root / "diagnostics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    universe_summary = result.universe_summary or summarize_cn_universe(result.universe)
    sample_report = result.panel.sample_report or {}
    main_summary = {
        "sample_report": sample_report,
        "jump_stats": result.pipeline.jump_stats,
        "factor_counts": {
            "K_hf_hat": result.pipeline.K_hf_hat,
            "K_cont_hat": result.pipeline.K_cont_hat,
            "K_jump_hat": result.pipeline.K_jump_hat,
            "display_cont_factor_count": int(
                result.pipeline.pca_cont_display.Lambda.shape[1]
            ) if result.pipeline.pca_cont_display is not None else 0,
            "g_fn": result.pipeline.g_fn,
            "gamma": result.pipeline.gamma,
        },
        "sharpes": result.pipeline.sharpes,
        "panel_return_scheme": result.panel.panel_return_scheme,
        "requested_return_mode": result.panel.requested_return_mode,
        "stage_timings": result.stage_timings,
    }

    universe_csv = diagnostics_dir / "universe_scan.csv"
    _atomic_to_csv(result.universe, universe_csv, index=False, encoding="utf-8-sig")
    _write_json(diagnostics_dir / "universe_summary.json", universe_summary)
    _write_json(diagnostics_dir / "main_summary.json", main_summary)
    stage_timings_path = diagnostics_dir / "stage_timings.json"
    _write_json(stage_timings_path, result.stage_timings)

    sample_summary_path = tables_dir / "Table_01_sample_summary.csv"
    _atomic_to_csv(
        pd.DataFrame(
        [
            {
                "adjustment": sample_report.get("adjustment", "raw_or_unknown"),
                "sample_mode": result.panel.sample_mode,
                "panel_return_scheme": result.panel.panel_return_scheme,
                "requested_return_mode": result.panel.requested_return_mode,
                "years": sample_report.get("years"),
                "n_symbols_selected": result.panel.N,
                "n_days_selected": result.panel.D,
                "bars_per_day": result.panel.M_per_day,
                "selected_calendar_start": sample_report.get("selected_calendar_start"),
                "selected_calendar_end": sample_report.get("selected_calendar_end"),
                "universe_total_symbols": universe_summary.get("total_symbols"),
                "strict_balanced_symbols": universe_summary.get("strict_balanced_symbols"),
                "global_start": universe_summary.get("global_start"),
                "global_end": universe_summary.get("global_end"),
            }
        ]
        ),
        sample_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    jump_stats_path = tables_dir / "Table_02_jump_stats.csv"
    _atomic_to_csv(pd.DataFrame([result.pipeline.jump_stats]), jump_stats_path, index=False, encoding="utf-8-sig")
    factor_counts_path = tables_dir / "Table_03_factor_counts.csv"
    _atomic_to_csv(
        pd.DataFrame(
        [
            {
                "K_hf_hat": result.pipeline.K_hf_hat,
                "K_cont_hat": result.pipeline.K_cont_hat,
                "K_jump_hat": result.pipeline.K_jump_hat,
                "g_fn": result.pipeline.g_fn,
                "gamma": result.pipeline.gamma,
            }
        ]
        ),
        factor_counts_path,
        index=False,
        encoding="utf-8-sig",
    )
    sharpes_path = tables_dir / "Table_04_sharpes.csv"
    _atomic_to_csv(pd.DataFrame([result.pipeline.sharpes]), sharpes_path, index=False, encoding="utf-8-sig")

    sample_symbols_path = diagnostics_dir / "main_sample_symbols.csv"
    _atomic_to_csv(pd.DataFrame({"symbol": result.panel.tickers}), sample_symbols_path, index=False, encoding="utf-8-sig")

    rolling_gc_df, rolling_explained_df = _rolling_output_frames(result.rolling_gc, result.rolling_explained_variation)
    rolling_gc_path = tables_dir / "Table_05_rolling_gc.csv"
    rolling_ev_path = tables_dir / "Table_06_rolling_explained_variation.csv"
    _atomic_to_csv(rolling_gc_df, rolling_gc_path, index=False, encoding="utf-8-sig")
    _atomic_to_csv(rolling_explained_df, rolling_ev_path, index=False, encoding="utf-8-sig")

    canonical_table_paths = {
        "Table_I": tables_dir / "Table_I_summary_statistics_for_continuous_and_jump_returns.csv",
        "Table_II": tables_dir / "Table_II_balanced_and_unbalanced_panel_results.csv",
        "Table_III": tables_dir / "Table_III_generalized_correlations_with_industry_and_ffc_factors.csv",
        "Table_IV": tables_dir / "Table_IV_time_variation_decomposition.csv",
        "Table_V": tables_dir / "Table_V_intraday_overnight_daily_sharpe_ratios.csv",
    }
    legacy_table_aliases = {
        "Table_I": [tables_dir / "Table_08_paper_style_yearly_jump_stats.csv"],
        "Table_V": [tables_dir / "Table_10_paper_style_factor_sharpes.csv"],
    }

    _write_csv_with_aliases(result.paper_table_i, canonical_table_paths["Table_I"], legacy_table_aliases.get("Table_I"))
    _write_csv_with_aliases(result.paper_table_ii, canonical_table_paths["Table_II"])
    _write_csv_with_aliases(result.paper_table_iii, canonical_table_paths["Table_III"])
    _write_csv_with_aliases(result.paper_table_iv, canonical_table_paths["Table_IV"])
    _write_csv_with_aliases(result.paper_table_v, canonical_table_paths["Table_V"], legacy_table_aliases.get("Table_V"))

    factor_count_diag_path = diagnostics_dir / "paper_factor_count_diagnostics.csv"
    _write_csv_with_aliases(result.paper_factor_counts, factor_count_diag_path, [tables_dir / "Table_09_paper_style_factor_count_diagnostics.csv"])

    pca_weights_path = tables_dir / "Table_11_continuous_pca_weights.csv"
    _atomic_to_csv(result.pca_weights, pca_weights_path, index=False, encoding="utf-8-sig")
    proxy_weights_path = tables_dir / "Table_12_proxy_factor_weights.csv"
    _atomic_to_csv(result.proxy_weights, proxy_weights_path, index=False, encoding="utf-8-sig")
    monthly_weights_path = tables_dir / "Table_13_monthly_pca_weights.csv"
    _atomic_to_csv(result.monthly_pca_weights, monthly_weights_path, index=False, encoding="utf-8-sig")
    factor_return_summary_path = tables_dir / "Table_14_factor_return_summary.csv"
    _atomic_to_csv(result.factor_return_summary, factor_return_summary_path, index=False, encoding="utf-8-sig")

    corp_action_path = diagnostics_dir / "corp_action_risk_after_adjustment.csv"
    _atomic_to_csv(result.corp_action_risk, corp_action_path, index=False, encoding="utf-8-sig")
    coverage_path = diagnostics_dir / "replication_coverage_report.csv"
    _atomic_to_csv(result.replication_coverage, coverage_path, index=False, encoding="utf-8-sig")
    rolling_weight_path = diagnostics_dir / "rolling_weight_summary.csv"
    _atomic_to_csv(result.rolling_weight_summary, rolling_weight_path, index=False, encoding="utf-8-sig")
    cumulative_returns_path = diagnostics_dir / "cumulative_factor_returns.csv"
    _atomic_to_csv(result.cumulative_factor_returns, cumulative_returns_path, index=False, encoding="utf-8-sig")

    exported_files = {
        "universe_scan": str(universe_csv),
        "universe_summary": str(diagnostics_dir / "universe_summary.json"),
        "main_summary": str(diagnostics_dir / "main_summary.json"),
        "stage_timings": str(stage_timings_path),
        "resource_plan": str(diagnostics_dir / "resource_plan.json"),
        "progress_log": str(diagnostics_dir / "progress.jsonl"),
        "sample_summary": str(sample_summary_path),
        "jump_stats": str(jump_stats_path),
        "factor_counts": str(factor_counts_path),
        "sharpes": str(sharpes_path),
        "main_sample_symbols": str(sample_symbols_path),
        "rolling_gc": str(rolling_gc_path),
        "rolling_explained_variation": str(rolling_ev_path),
        "Table_I": str(canonical_table_paths["Table_I"]),
        "Table_II": str(canonical_table_paths["Table_II"]),
        "Table_III": str(canonical_table_paths["Table_III"]),
        "Table_IV": str(canonical_table_paths["Table_IV"]),
        "Table_V": str(canonical_table_paths["Table_V"]),
        "paper_factor_count_diagnostics": str(factor_count_diag_path),
        "continuous_pca_weights": str(pca_weights_path),
        "proxy_factor_weights": str(proxy_weights_path),
        "monthly_pca_weights": str(monthly_weights_path),
        "factor_return_summary": str(factor_return_summary_path),
        "rolling_weight_summary": str(rolling_weight_path),
        "cumulative_factor_returns": str(cumulative_returns_path),
        "corp_action_risk": str(corp_action_path),
        "replication_coverage_report": str(coverage_path),
        "Table_08": str(tables_dir / "Table_08_paper_style_yearly_jump_stats.csv"),
        "Table_09": str(tables_dir / "Table_09_paper_style_factor_count_diagnostics.csv"),
        "Table_10": str(tables_dir / "Table_10_paper_style_factor_sharpes.csv"),
    }

    legacy_figure_aliases = {
        "Figure_1": [figures_dir / "Figure_01_number_of_hf_factors_unbalanced.png"],
        "Figure_2": [figures_dir / "Figure_02_number_of_hf_factors_balanced.png"],
        "Figure_3": [figures_dir / "Figure_03_proxy_factor_weights.png"],
        "Figure_4": [figures_dir / "Figure_04_continuous_pca_weights.png"],
        "Figure_5": [figures_dir / "Figure_05_monthly_pca_weights.png"],
        "Figure_6": [figures_dir / "Figure_06_time_variation_loadings_gc.png"],
        "Figure_7": [figures_dir / "Figure_07_local_continuous_factor_gc.png"],
        "Figure_8": [figures_dir / "Figure_08_time_varying_portfolio_weights.png"],
        "Figure_9": [figures_dir / "Figure_09_time_varying_explained_variation.png"],
        "Figure_10": [figures_dir / "Figure_10_factor_structure_decomposition.png"],
        "Figure_11": [figures_dir / "Figure_11_continuous_factor_structure_decomposition.png"],
        "Figure_12": [figures_dir / "Figure_12_expected_intraday_overnight_returns.png"],
        "Figure_13": [figures_dir / "Figure_13_cumulative_factor_returns.png"],
        "Figure_14": [figures_dir / "Figure_14_asset_pricing_industry_portfolios.png"],
        "Figure_15": [figures_dir / "Figure_15_asset_pricing_size_value_portfolios.png"],
    }

    if save_plots:
        plot_status, figure_files = export_all_paper_figures(result, figures_dir, rolling_gc_df, rolling_explained_df)
        for figure_id, path in figure_files.items():
            alias_paths = legacy_figure_aliases.get(figure_id, [])
            _copy_alias_files(Path(path), alias_paths)
    else:
        plot_status = pd.DataFrame(
            [
                {
                    "figure_number": idx,
                    "figure_id": f"Figure_{idx}",
                    "title": f"Figure {idx}",
                    "file_path": "",
                    "status": "skipped",
                    "data_source": "--no-plots",
                    "notes": "Plot export skipped because --no-plots was used.",
                }
                for idx in range(1, 16)
            ]
        )
        figure_files = {}

    plot_status = plot_status.copy()
    plot_status["legacy_alias_path"] = plot_status["figure_id"].map(
        lambda figure_id: ";".join(str(path) for path in legacy_figure_aliases.get(figure_id, []))
    )
    result.plot_status = plot_status

    for figure_id, path in figure_files.items():
        exported_files[figure_id] = path
        alias_paths = legacy_figure_aliases.get(figure_id, [])
        for alias_path in alias_paths:
            try:
                alias_key = alias_path.stem.split("_")[0] + "_" + alias_path.stem.split("_")[1]
            except Exception:
                alias_key = alias_path.stem
            exported_files[alias_key] = str(alias_path)

    plot_status_path = diagnostics_dir / "plot_export_status.csv"
    _atomic_to_csv(plot_status, plot_status_path, index=False, encoding="utf-8-sig")
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
    return_mode: str = "open_close",
    max_stocks: Optional[int] = None,
    jump_a: float = 3.0,
    k_max: int = 10,
    gamma: float = 0.08,
    g_fn: str = "median_N",
    save_plots: bool = True,
    universe: Optional[pd.DataFrame] = None,
    workers: Optional[int] = None,
    paper_workers: Optional[int] = None,
    rolling_workers: Optional[int] = None,
    memory_budget_gb: Optional[float] = None,
    progress_interval_sec: float = DEFAULT_PROGRESS_INTERVAL_SEC,
    restart: bool = False,
) -> ReplicationResult:
    """Run the China A-share replication from preprocessed proc_Data panels only."""
    proc_root = _ensure_path(proc_root)
    output_root = _ensure_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    workers = _normalize_worker_count(workers)
    paper_workers = _resolve_stage_workers(workers, paper_workers)
    rolling_workers = _resolve_stage_workers(workers, rolling_workers)
    runtime = _build_runtime_config(output_root=output_root, memory_budget_gb=memory_budget_gb, progress_interval_sec=progress_interval_sec)
    run_signature = _build_run_signature(
        proc_root=proc_root,
        years=years,
        max_stocks=max_stocks,
        jump_a=jump_a,
        k_max=k_max,
        gamma=gamma,
        g_fn=g_fn,
        return_mode=return_mode,
    )
    checkpoint_manager = CheckpointManager(output_root=output_root, signature=run_signature, restart=bool(restart))
    checkpoint_info = checkpoint_manager.prepare()
    progress = ProgressReporter(
        diagnostics_dir=diagnostics_dir,
        interval_sec=runtime.progress_interval_sec,
        reset_existing=not bool(checkpoint_info.get("resumed")),
    )
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    progress.update_state(stage="startup", resumed_from_previous=bool(checkpoint_info.get("resumed")))
    if checkpoint_info.get("cleaned_items"):
        progress.event(
            "checkpoint_cleaned",
            stage="startup",
            cleaned_count=len(checkpoint_info["cleaned_items"]),
            message=f"cleaned {len(checkpoint_info['cleaned_items'])} incomplete checkpoint/export artifacts",
        )
    if checkpoint_info.get("resumed"):
        progress.event("run_resumed", stage="startup", message="resuming from compatible checkpoints")
        progress.event(
            "checkpoint_reused",
            stage="startup",
            rolling_chunks=len(checkpoint_manager.completed_rolling_chunks()),
            paper_years=len(checkpoint_manager.completed_paper_years()),
            message="reusing previously completed checkpoint units",
        )
    progress.event("run_started", stage="startup", message="replication started")
    result: Optional[ReplicationResult] = None
    resource_plan: Optional[ResourcePlan] = None
    try:
        manifest_path = proc_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Processed data manifest not found: {manifest_path}. Run Code/preprocess_cn_data.py first."
            )

        t = time.perf_counter()
        if universe is None:
            universe = load_proc_universe(proc_root)
        timings["load_proc_universe_sec"] = time.perf_counter() - t if universe is not None else 0.0
        universe_summary = summarize_cn_universe(universe)
        checkpoint_manager.update(stage="load_panel", export_completed=False)
        progress.update_state(stage="load_panel")
        progress.event("stage_started", message="loading strict balanced panel")

        t = time.perf_counter()
        panel = load_proc_hf_panel(
            proc_root=proc_root,
            sample_mode=STRICT_BALANCED_SAMPLE,
            years=years,
            return_mode=return_mode,
            max_stocks=max_stocks,
        )
        timings["load_panel_sec"] = time.perf_counter() - t
        progress.event("stage_finished", stage="load_panel", message="strict balanced panel loaded")
        resource_plan = _build_paper_resource_plan(
            proc_root=proc_root,
            summary=_universe_attr_summary(universe),
            years=sorted({date.year for date in panel.dates}),
            requested_workers=workers,
            requested_paper_workers=paper_workers,
            requested_rolling_workers=rolling_workers,
            runtime=runtime,
            universe=universe,
            max_stocks=max_stocks,
        )
        _write_json(diagnostics_dir / "resource_plan.json", _json_ready(resource_plan.__dict__))

        t = time.perf_counter()
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
        timings["pipeline_core_sec"] = time.perf_counter() - t
        checkpoint_manager.update(stage="rolling", export_completed=False)
        progress.update_state(stage="rolling")
        progress.event("stage_started", message="rolling PCA")

        rolling_window = 21 if panel.D >= 21 else max(5, panel.D // 3)
        t = time.perf_counter()
        global_weights = factor_portfolio_weights(pipeline.pca_cont)
        selected_idx = np.argsort(-np.abs(global_weights[:, 0]))[: min(8, panel.N)]
        selected_symbols = [panel.tickers[int(idx)] for idx in selected_idx.tolist()]
        rolling_gc, rolling_ev, rolling_weight_summary = run_checkpointed_rolling_pca(
            R=pipeline.R_cont,
            day_ids=panel.day_ids,
            window_days=max(rolling_window, 2),
            K=pipeline.K_cont_hat,
            global_Lambda=pipeline.pca_cont.Lambda,
            checkpoint_manager=checkpoint_manager,
            selected_symbols=selected_symbols,
            selected_idx=selected_idx,
            step_days=1,
            use_corr=True,
            workers=resource_plan.rolling_workers_effective,
            progress=progress,
        )
        timings["rolling_sec"] = time.perf_counter() - t
        progress.event("stage_finished", stage="rolling", message="rolling PCA complete")

        robustness = pd.DataFrame()

        t = time.perf_counter()
        checkpoint_manager.update(stage="paper_tables", export_completed=False)
        progress.update_state(
            stage="paper_tables",
            paper_workers_effective=resource_plan.paper_workers_effective,
            memory_budget_gb=runtime.memory_budget_gb,
        )
        yearly_paper_outputs = build_yearly_paper_outputs(
            balanced_panel=panel,
            proc_root=proc_root,
            universe=universe,
            thresholds=(3.0, 4.0, 4.5, 5.0),
            jump_a=jump_a,
            gamma=gamma,
            g_fn=g_fn,
            k_max=k_max,
            workers=resource_plan.paper_workers_effective,
            max_stocks=max_stocks,
            runtime=runtime,
            progress=progress,
            resource_plan=resource_plan,
            checkpoint_manager=checkpoint_manager,
        )
        paper_table_i = yearly_paper_outputs.table_i
        paper_table_ii = yearly_paper_outputs.table_ii
        paper_factor_counts = yearly_paper_outputs.factor_counts
        paper_table_iii = build_paper_table_iii()
        replication_coverage = build_replication_coverage_report()
        pca_weights, proxy_weights = build_weight_tables(pipeline, panel)
        monthly_pca_weights = build_monthly_pca_weights(panel)
        factor_return_summary, cumulative_factor_returns = build_factor_return_tables(pipeline, panel)
        rolling_gc_df, rolling_explained_df = _rolling_output_frames(rolling_gc, rolling_ev)
        paper_table_iv = build_paper_table_iv(rolling_gc_df, rolling_explained_df)
        paper_table_v = build_paper_table_v(pipeline)
        timings["paper_panel_build_sec"] = float(yearly_paper_outputs.stage_timings.get("panel_build_sec", 0.0))
        timings["paper_jump_decompose_sec"] = float(yearly_paper_outputs.stage_timings.get("jump_decompose_sec", 0.0))
        timings["paper_pca_sec"] = float(yearly_paper_outputs.stage_timings.get("pca_sec", 0.0))
        timings["paper_table_assemble_sec"] = float(yearly_paper_outputs.stage_timings.get("table_assemble_sec", 0.0))
        timings["paper_tables_sec"] = time.perf_counter() - t

        result = ReplicationResult(
            universe=universe,
            universe_summary=universe_summary,
            panel=panel,
            pipeline=pipeline,
            rolling_gc=rolling_gc,
            rolling_explained_variation=rolling_ev,
            robustness=robustness,
            output_root=output_root,
            corp_action_risk=corp_action_risk,
            paper_jump_stats=paper_table_i,
            paper_factor_counts=paper_factor_counts,
            paper_factor_sharpes=paper_table_v,
            paper_table_i=paper_table_i,
            paper_table_ii=paper_table_ii,
            paper_table_iii=paper_table_iii,
            paper_table_iv=paper_table_iv,
            paper_table_v=paper_table_v,
            replication_coverage=replication_coverage,
            pca_weights=pca_weights,
            proxy_weights=proxy_weights,
            monthly_pca_weights=monthly_pca_weights,
            rolling_weight_summary=rolling_weight_summary,
            factor_return_summary=factor_return_summary,
            cumulative_factor_returns=cumulative_factor_returns,
            stage_timings=dict(timings),
            resource_plan=_json_ready(resource_plan.__dict__),
        )
        t = time.perf_counter()
        checkpoint_manager.update(stage="export", export_completed=False)
        progress.update_state(stage="export")
        progress.event("stage_started", message="export outputs")
        export_replication_outputs(result, save_plots=save_plots)
        timings["export_sec"] = time.perf_counter() - t
        timings["total_sec"] = time.perf_counter() - t_total
        result.stage_timings = dict(timings)
        _write_json(diagnostics_dir / "stage_timings.json", timings)
        _write_json(diagnostics_dir / "resource_plan.json", result.resource_plan)
        counts = result.plot_status["status"].value_counts().to_dict() if not result.plot_status.empty else {}
        main_summary = {
            "sample_report": result.panel.sample_report or {},
            "jump_stats": result.pipeline.jump_stats,
            "factor_counts": {
                "K_hf_hat": result.pipeline.K_hf_hat,
                "K_cont_hat": result.pipeline.K_cont_hat,
                "K_jump_hat": result.pipeline.K_jump_hat,
                "display_cont_factor_count": int(
                    result.pipeline.pca_cont_display.Lambda.shape[1]
                ) if result.pipeline.pca_cont_display is not None else 0,
                "g_fn": result.pipeline.g_fn,
                "gamma": result.pipeline.gamma,
            },
            "sharpes": result.pipeline.sharpes,
            "panel_return_scheme": result.panel.panel_return_scheme,
            "requested_return_mode": result.panel.requested_return_mode,
            "stage_timings": timings,
            "plots_generated_count": int(counts.get("generated", 0)),
            "plots_placeholder_count": int(counts.get("placeholder", 0)),
            "plots_failed_count": int(counts.get("error", 0)),
            "plots_skipped_count": int(counts.get("skipped", 0)),
        }
        _write_json(diagnostics_dir / "main_summary.json", main_summary)
        checkpoint_manager.mark_export_complete()
        progress.event("stage_finished", stage="export", message="export complete")
        progress.event("run_finished", stage="done", message="replication finished")
        return result
    except Exception as exc:
        timings["total_sec"] = time.perf_counter() - t_total
        _write_json(diagnostics_dir / "stage_timings.json", timings)
        if resource_plan is not None:
            _write_json(diagnostics_dir / "resource_plan.json", _json_ready(resource_plan.__dict__))
        progress.event(
            "run_failed",
            stage=progress.snapshot_state().get("stage", "failed"),
            message=f"{type(exc).__name__}: {exc}",
        )
        checkpoint_manager.mark_failed(stage=str(progress.snapshot_state().get("stage", "failed")))
        raise
    finally:
        if runtime.scratch_root.exists():
            shutil.rmtree(runtime.scratch_root, ignore_errors=True)

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
    print(f" Invalid-day symbols    : {summary['invalid_day_symbols']}")
    corp = summary.get("corp_action_risk_summary")
    if corp:
        print(f" Suspicious overnight>12% symbols : {corp['symbols_with_suspicious_overnight_012']}")
        print(f" Suspicious overnight>20% symbols : {corp['symbols_with_suspicious_overnight_020']}")
        print(f" Max abs overnight                : {corp['max_abs_overnight_overall']:.4f}")
    print("=" * 78)
