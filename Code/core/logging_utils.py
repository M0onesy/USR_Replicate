"""
core/logging_utils.py
=====================

Lightweight console logging helpers used by main.py and standalone tasks.
The engine already owns structured JSONL progress logs; this module keeps
human-readable console output compact and real-time.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _emit(tag: str, kind: str, message: str) -> None:
    line = f"[{_ts()}] [{tag:<15}] [{kind}] {message}"
    print(line, flush=True)


def log_start(tag: str, message: str) -> None:
    _emit(tag, "开始  ", message)


def log_step(tag: str, message: str) -> None:
    _emit(tag, "数据处理", message)


def log_render(tag: str, message: str) -> None:
    _emit(tag, "图表输出", message)


def log_done(tag: str, message: str) -> None:
    _emit(tag, "完成  ", message)


def log_info(tag: str, message: str) -> None:
    _emit(tag, "信息  ", message)


def log_warn(tag: str, message: str) -> None:
    _emit(tag, "警告  ", message)


class Heartbeat:
    """
    Background heartbeat for long-running tasks.

    If a runtime progress.jsonl is attached, heartbeat prefers showing the
    latest structured pipeline state. Otherwise it falls back to the older
    task/done/total summary maintained by main.py.
    """

    def __init__(self, interval_sec: float = 10.0, label: str = "HEARTBEAT") -> None:
        self.interval_sec = max(1.0, float(interval_sec))
        self.label = label
        self._lock = threading.Lock()
        self._current_task: str = "(未开始)"
        self._done = 0
        self._total = 0
        self._t0 = time.perf_counter()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._progress_path: Optional[Path] = None
        self._progress_min_mtime_unix: float = 0.0

    def set_status(self, task: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
        with self._lock:
            self._current_task = task
            if done is not None:
                self._done = done
            if total is not None:
                self._total = total

    def attach_progress_log(self, path: str | Path) -> None:
        with self._lock:
            self._progress_path = Path(path)
            self._progress_min_mtime_unix = time.time()

    def clear_progress_log(self) -> None:
        with self._lock:
            self._progress_path = None
            self._progress_min_mtime_unix = 0.0

    def _snapshot(self) -> tuple[str, int, int, float]:
        with self._lock:
            return self._current_task, self._done, self._total, time.perf_counter() - self._t0

    def _read_latest_progress_record(self) -> Optional[dict]:
        with self._lock:
            progress_path = self._progress_path
            min_mtime_unix = self._progress_min_mtime_unix
        if progress_path is None or not progress_path.exists():
            return None
        try:
            if progress_path.stat().st_mtime < min_mtime_unix:
                return None
            with progress_path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                offset = max(0, size - 16384)
                fh.seek(offset)
                blob = fh.read()
        except OSError:
            return None
        lines = blob.splitlines()
        if offset > 0 and lines:
            lines = lines[1:]
        for raw_line in reversed(lines):
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if isinstance(record, dict):
                return record
        return None

    def _format_progress_message(self, elapsed: float) -> Optional[str]:
        record = self._read_latest_progress_record()
        if not record:
            return None
        parts = [f"已运行 {elapsed:.0f}s"]
        stage = record.get("stage")
        if stage:
            parts.append(f"stage={stage}")
        if record.get("completed_years") is not None and record.get("total_years") is not None:
            parts.append(f"years={record.get('completed_years')}/{record.get('total_years')}")
        elif record.get("completed_chunks") is not None and record.get("total_chunks") is not None:
            parts.append(f"chunks={record.get('completed_chunks')}/{record.get('total_chunks')}")
        if record.get("active_years") is not None:
            parts.append(f"active={record.get('active_years')}")
        if record.get("paper_workers_effective") is not None:
            parts.append(f"workers={record.get('paper_workers_effective')}")
        if record.get("memory_reserved_gb") is not None and record.get("memory_budget_gb") is not None:
            parts.append(
                f"mem={float(record.get('memory_reserved_gb')):.2f}/{float(record.get('memory_budget_gb')):.2f}GB"
            )
        event = str(record.get("event") or "")
        message = str(record.get("message") or "").strip()
        if message and event not in {"heartbeat", "stage_started", "stage_finished"}:
            parts.append(message)
        return " | ".join(parts)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            task, done, total, elapsed = self._snapshot()
            progress_message = self._format_progress_message(elapsed)
            if progress_message:
                print(f"[{_ts()}] [{self.label}] {progress_message}", flush=True)
                continue
            progress = f"{done}/{total}" if total else f"{done}"
            print(
                f"[{_ts()}] [{self.label}] 已运行 {elapsed:.0f}s | 当前任务 {task} | 已完成 {progress}",
                flush=True,
            )

    def start(self) -> "Heartbeat":
        self._t0 = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
            self._thread = None

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
