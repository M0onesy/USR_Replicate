"""
core/logging_utils.py
=====================

统一的控制台输出 / 心跳工具，服务两个需求：

  1. 每篇图 / 表脚本被运行到时，打印一条"相关信息"的输出（开始、阶段、完成）。
  2. main.py 作为控制台，需要"不断心跳报告"，方便观察整体进度、定位卡顿。

为什么自己写而不直接用 logging：
  这里要的是给人看的、带时间戳和阶段标签的简洁中文行，并且要能在后台线程里
  做"还在跑"的心跳。一个轻量封装比配置 logging 更直观，也不干扰 engine 内部
  已有的 ProgressReporter（那套是写 JSONL 的结构化日志，互不冲突）。

输出格式示例：
  [12:01:03] [figure_08      ] [数据处理] 正在从滚动权重摘要切片 Factor-1 权重轨迹
  [12:01:03] [figure_08      ] [完成]    Figure 8 已保存 -> .../Figure_8_xxx.png  (0.42s)
  [12:01:13] [HEARTBEAT] 已运行 10s | 当前任务 figure_08 | 已完成 7/20
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _emit(tag: str, kind: str, message: str) -> None:
    """底层输出：固定宽度对齐，立即 flush，方便实时观察。"""
    line = f"[{_ts()}] [{tag:<15}] [{kind}] {message}"
    print(line, flush=True)


def log_start(tag: str, message: str) -> None:
    """脚本开始时的"我被运行到了"提示。"""
    _emit(tag, "开始  ", message)


def log_step(tag: str, message: str) -> None:
    """数据处理阶段：标记"重活在这一步"，卡住时能看出是哪段。"""
    _emit(tag, "数据处理", message)


def log_render(tag: str, message: str) -> None:
    """图表最终输出阶段：与数据处理区分开。"""
    _emit(tag, "图表输出", message)


def log_done(tag: str, message: str) -> None:
    _emit(tag, "完成  ", message)


def log_info(tag: str, message: str) -> None:
    _emit(tag, "信息  ", message)


def log_warn(tag: str, message: str) -> None:
    _emit(tag, "警告  ", message)


class Heartbeat:
    """后台心跳线程：main.py 在长任务运行期间持续报告"还活着"。

    用法：
        hb = Heartbeat(interval_sec=10)
        hb.set_status("figure_08", done=7, total=20)
        hb.start()
        ...                       # 跑任务，期间不断 set_status
        hb.stop()

    心跳与具体任务解耦：任务线程只负责更新 status，心跳线程定时打印。
    这样即使某张图卡住（任务线程不动），心跳仍会持续输出，立刻能看出"卡在哪"。
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

    def set_status(self, task: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
        with self._lock:
            self._current_task = task
            if done is not None:
                self._done = done
            if total is not None:
                self._total = total

    def _snapshot(self) -> tuple[str, int, int, float]:
        with self._lock:
            return self._current_task, self._done, self._total, time.perf_counter() - self._t0

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            task, done, total, elapsed = self._snapshot()
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
