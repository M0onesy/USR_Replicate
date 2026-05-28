"""
main.py —— 论文复现图表生成总控台
==================================

本脚本是拆分后项目的"控制台"，统一调度 figcode / tablecode 下的所有图表脚本。
对应拆分方案 Option A：昂贵的上游计算（ReplicationResult）只构建一次，所有图、表
复用同一份结果；既能"全部运行"，也能"只跑某一篇"方便维护调试。

核心特性
--------
1. 一次计算，多处复用
   先调用 pipeline_cache.get_result() 把 ReplicationResult 准备好（命中缓存则秒回，
   否则构建一次并落盘缓存），再依次把它喂给每个图表脚本。绝不重复跑 PCA / 滚动分析。

2. 灵活调度
     python main.py --only all            # 全部图 + 全部表
     python main.py --only figures         # 只跑所有图
     python main.py --only tables          # 只跑所有表
     python main.py --only fig8            # 只跑 Figure 8
     python main.py --only fig8 table_i    # 跑 Figure 8 和 Table I
     python main.py --list                 # 列出所有可用任务

3. 持续心跳
   后台心跳线程每隔 --heartbeat-sec 秒报告一次"已运行多久 / 当前任务 / 已完成几项"，
   长任务卡住时一眼能看出停在哪一步。

4. 小样本调试
     python main.py --only all --years 2016 --max-stocks 10
   只用 2016 年前 10 只股票快速跑通整条链路。

5. 逐任务隔离
   单个图表脚本抛错不会中断整体（默认 --keep-going），最后给出成功/失败汇总；
   想要一遇错就停，可加 --fail-fast。
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List

# 确保以"python main.py"运行时能 import core / figcode / tablecode 包。
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig
from core.logging_utils import Heartbeat, log_info, log_warn, log_done
from core.pipeline_cache import get_result
from core.registry import all_tasks, resolve_keys, Task
from core.runner import run_generator


def _build_config(args: argparse.Namespace) -> RunConfig:
    """根据命令行参数构造 RunConfig（数值参数 + 路径 + 并行）。"""
    cfg = RunConfig()
    if args.proc_root is not None:
        cfg.proc_root = Path(args.proc_root)
    if args.output_root is not None:
        cfg.output_root = Path(args.output_root)
    if args.years is not None:
        cfg.years = args.years
    if args.max_stocks is not None:
        cfg.max_stocks = args.max_stocks
    if args.jump_a is not None:
        cfg.jump_a = args.jump_a
    if args.k_max is not None:
        cfg.k_max = args.k_max
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.g_fn is not None:
        cfg.g_fn = args.g_fn
    if args.workers is not None:
        cfg.workers = args.workers
    if args.paper_workers is not None:
        cfg.paper_workers = args.paper_workers
    if args.rolling_workers is not None:
        cfg.rolling_workers = args.rolling_workers
    if args.memory_budget_gb is not None:
        cfg.memory_budget_gb = args.memory_budget_gb
    cfg.progress_interval_sec = args.heartbeat_sec
    cfg.restart = bool(args.restart)
    # 图由 figcode 负责输出，构建结果时不让 engine 再顺带画全套图，避免重复画。
    cfg.save_plots = False
    return cfg


def _print_task_list() -> None:
    print("=" * 78)
    print(" 可用任务列表（--only 接受这些短名，或 all / figures / tables）")
    print("=" * 78)
    print(" [图 figure]")
    for t in all_tasks():
        if t.kind == "figure":
            print(f"   {t.key:<8} {t.desc}")
    print(" [表 table]")
    for t in all_tasks():
        if t.kind == "table":
            print(f"   {t.key:<10} {t.desc}")
    print("=" * 78)


def _run_tasks(tasks: List[Task], cfg: RunConfig, heartbeat: Heartbeat, fail_fast: bool) -> tuple[list, list]:
    """依次执行任务，复用同一个 ReplicationResult。"""
    # 关键：上游计算只在这里触发一次（命中缓存则直接返回）。
    log_info("main", "准备 ReplicationResult（命中缓存则秒回，否则构建一次）…")
    heartbeat.set_status("准备 ReplicationResult", done=0, total=len(tasks))
    result = get_result(cfg)
    log_done("main", "ReplicationResult 就绪，开始逐个生成图表")

    succeeded: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for idx, task in enumerate(tasks, start=1):
        heartbeat.set_status(f"{task.key}（{task.kind}）", done=len(succeeded) + len(failed), total=len(tasks))
        log_info("main", f"[{idx}/{len(tasks)}] 开始 {task.key} — {task.desc}")
        try:
            generate = task.load_generate()
            output_path = run_generator(task.key, generate, result=result, cfg=cfg)
            succeeded.append((task.key, str(output_path)))
        except Exception as exc:
            failed.append((task.key, f"{type(exc).__name__}: {exc}"))
            log_warn("main", f"任务 {task.key} 失败：{type(exc).__name__}: {exc}")
            if fail_fast:
                log_warn("main", "已启用 --fail-fast，立即中止")
                traceback.print_exc()
                break
        finally:
            heartbeat.set_status(f"{task.key} 结束", done=len(succeeded) + len(failed), total=len(tasks))

    return succeeded, failed


def _print_summary(succeeded: list, failed: list, total_elapsed: float) -> None:
    print("=" * 78)
    print(" 运行汇总")
    print("=" * 78)
    print(f" 成功 {len(succeeded)} 项 / 失败 {len(failed)} 项 / 总用时 {total_elapsed:.1f}s")
    if succeeded:
        print(" [成功]")
        for key, path in succeeded:
            print(f"   {key:<10} -> {path}")
    if failed:
        print(" [失败]")
        for key, err in failed:
            print(f"   {key:<10} : {err}")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pelger(2020) 复现图表生成总控台（Option A：一次计算，多处复用）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 任务选择
    p.add_argument("--only", nargs="+", default=["all"],
                   help="要运行的任务：all / figures / tables / 具体短名（如 fig8 table_i）。默认 all")
    p.add_argument("--list", action="store_true", help="列出所有可用任务后退出")

    # 心跳与容错
    p.add_argument("--heartbeat-sec", type=float, default=10.0, help="心跳间隔秒数（默认 10）")
    p.add_argument("--fail-fast", action="store_true", help="任一任务失败立即中止（默认继续跑其余任务）")
    p.add_argument("--no-heartbeat", action="store_true", help="关闭后台心跳线程")

    # 路径
    p.add_argument("--proc-root", default=None, help="预处理数据目录")
    p.add_argument("--output-root", default=None, help="结果输出目录")

    # 样本
    p.add_argument("--years", nargs="+", type=int, default=None, help="指定年份，如 --years 2016 2017")
    p.add_argument("--max-stocks", type=int, default=None, help="只用前 N 只股票（smoke test）")

    # 论文方法参数
    p.add_argument("--jump-a", type=float, default=None, help="跳跃阈值倍数（默认 3.0）")
    p.add_argument("--k-max", type=int, default=None, help="因子个数搜索上界（默认 10）")
    p.add_argument("--gamma", type=float, default=None, help="扰动特征值比阈值（默认 0.08）")
    p.add_argument("--g-fn", default=None, choices=["median_N", "median_sqrtN", "logN", "none"],
                   help="扰动平移函数（默认 median_N）")

    # 并行 / 资源
    p.add_argument("--workers", type=int, default=None, help="通用并行 worker 数")
    p.add_argument("--paper-workers", type=int, default=None, help="逐年论文表并行 worker 数")
    p.add_argument("--rolling-workers", type=int, default=None, help="滚动 PCA 并行 worker 数")
    p.add_argument("--memory-budget-gb", type=float, default=None, help="自适应内存预算 GB")

    # 运行行为
    p.add_argument("--restart", action="store_true", help="丢弃兼容 checkpoint 重跑上游 pipeline")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        _print_task_list()
        return 0

    try:
        tasks = resolve_keys(args.only)
    except KeyError as exc:
        log_warn("main", str(exc))
        _print_task_list()
        return 2
    if not tasks:
        log_warn("main", "没有解析到任何任务，请检查 --only 参数")
        return 2

    cfg = _build_config(args)
    log_info("main", f"任务数 {len(tasks)}：{', '.join(t.key for t in tasks)}")
    log_info("main", f"输出目录 {cfg.output_root}")
    if cfg.years or cfg.max_stocks:
        log_info("main", f"样本限制 years={cfg.years} max_stocks={cfg.max_stocks}")

    heartbeat = Heartbeat(interval_sec=args.heartbeat_sec)
    t0 = time.perf_counter()
    if not args.no_heartbeat:
        heartbeat.start()
    try:
        succeeded, failed = _run_tasks(tasks, cfg, heartbeat, fail_fast=args.fail_fast)
    finally:
        heartbeat.stop()

    _print_summary(succeeded, failed, time.perf_counter() - t0)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
