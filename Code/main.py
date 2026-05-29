from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig
from core.engine import _atomic_to_csv, _plot_status_row
from core.logging_utils import Heartbeat, log_done, log_info, log_warn
from core.pipeline_cache import get_result
from core.registry import Task, all_tasks, resolve_keys
from core.runner import run_generator


def _build_config(args: argparse.Namespace) -> RunConfig:
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
    cfg.restart = bool(args.restart and args.rebuild_result)
    return cfg


def _should_export_engine_figures(only_args: List[str]) -> bool:
    selected = {str(item).lower() for item in only_args}
    return "all" in selected or "figures" in selected


def _is_figure_task(task: Task) -> bool:
    return task.kind == "figure"


def _has_figure_tasks(tasks: List[Task]) -> bool:
    return any(_is_figure_task(task) for task in tasks)


def _write_plot_status_for_export_only(tasks: List[Task], succeeded: list[tuple[str, str]], failed: list[tuple[str, str]], cfg: RunConfig) -> None:
    diagnostics_dir = Path(cfg.output_root) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    succeeded_map = {key: path for key, path in succeeded}
    failed_map = {key: err for key, err in failed}
    rows = []
    for task in tasks:
        if not _is_figure_task(task):
            continue
        try:
            figure_number = int(task.key.replace("fig", ""))
        except Exception:
            continue
        figure_id = f"Figure_{figure_number}"
        title = task.desc
        if task.key in succeeded_map:
            rows.append(
                _plot_status_row(
                    figure_id,
                    title,
                    Path(succeeded_map[task.key]),
                    "generated",
                    "reused_replication_result",
                    "Re-exported from an existing ReplicationResult without rebuilding the pipeline.",
                )
            )
        elif task.key in failed_map:
            rows.append(
                _plot_status_row(
                    figure_id,
                    title,
                    Path(cfg.output_root) / "figures" / f"{figure_id}.png",
                    "error",
                    "reused_replication_result",
                    failed_map[task.key],
                )
            )

    if rows:
        plot_status = sorted(rows, key=lambda item: int(item.get("figure_number", -1)))
        _atomic_to_csv(
            __import__("pandas").DataFrame(plot_status),
            diagnostics_dir / "plot_export_status.csv",
            index=False,
            encoding="utf-8-sig",
        )


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


def _resolve_result(cfg: RunConfig, *, rebuild_result: bool):
    if rebuild_result:
        log_info("main", "显式重建模式：将按当前参数构建新的 ReplicationResult")
        return get_result(cfg, allow_build=True, allow_fallback=False)
    log_info("main", "优先复用已有 ReplicationResult；若找到已完成缓存，将直接重导图表/表格")
    return get_result(cfg, allow_build=False, allow_fallback=True)


def _run_tasks(tasks: List[Task], cfg: RunConfig, heartbeat: Heartbeat, fail_fast: bool, rebuild_result: bool) -> tuple[list, list]:
    log_info("main", "准备 ReplicationResult（命中缓存则秒回，否则按模式决定是否构建）…")
    heartbeat.set_status("准备 ReplicationResult", done=0, total=len(tasks))
    result = _resolve_result(cfg, rebuild_result=rebuild_result)
    log_done("main", "ReplicationResult 就绪，开始逐个生成图表")

    succeeded: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for idx, task in enumerate(tasks, start=1):
        heartbeat.set_status(f"{task.key}（{task.kind}）", done=len(succeeded) + len(failed), total=len(tasks))
        log_info("main", f"[{idx}/{len(tasks)}] 开始 {task.key} - {task.desc}")
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

    if not rebuild_result and _has_figure_tasks(tasks):
        _write_plot_status_for_export_only(tasks, succeeded, failed, cfg)

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
        description="Pelger(2020) 论文图表生成总控台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", nargs="+", default=["all"], help="要运行的任务：all / figures / tables / 具体短名（如 fig8 table_i）")
    p.add_argument("--list", action="store_true", help="列出所有可用任务后退出")

    p.add_argument("--heartbeat-sec", type=float, default=10.0, help="心跳间隔秒数（默认 10）")
    p.add_argument("--fail-fast", action="store_true", help="任一任务失败立刻中止")
    p.add_argument("--no-heartbeat", action="store_true", help="关闭后台心跳线程")

    p.add_argument("--proc-root", default=None, help="预处理数据目录")
    p.add_argument("--output-root", default=None, help="结果输出目录")

    p.add_argument("--years", nargs="+", type=int, default=None, help="指定年份，如 --years 2016 2017")
    p.add_argument("--max-stocks", type=int, default=None, help="只用前 N 只股票")

    p.add_argument("--jump-a", type=float, default=None, help="跳跃阈值倍数")
    p.add_argument("--k-max", type=int, default=None, help="因子个数搜索上界")
    p.add_argument("--gamma", type=float, default=None, help="扰动特征值比阈值")
    p.add_argument("--g-fn", default=None, choices=["median_N", "median_sqrtN", "logN", "none"], help="扰动平移函数")

    p.add_argument("--workers", type=int, default=None, help="通用并行 worker 数")
    p.add_argument("--paper-workers", type=int, default=None, help="paper stage worker 数")
    p.add_argument("--rolling-workers", type=int, default=None, help="rolling stage worker 数")
    p.add_argument("--memory-budget-gb", type=float, default=None, help="自适应内存预算 GB")

    p.add_argument("--rebuild-result", action="store_true", help="显式重建上游 ReplicationResult，而不是优先复用已有结果")
    p.add_argument("--restart", action="store_true", help="仅与 --rebuild-result 配合使用：清理兼容 checkpoint 后重跑上游 pipeline")
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
    cfg.save_plots = _should_export_engine_figures(args.only)
    if args.restart and not args.rebuild_result:
        log_warn("main", "检测到仅传入 --restart。当前运行是导出优先模式，将忽略 --restart；若需重建请加 --rebuild-result。")

    log_info("main", f"任务数 {len(tasks)}：{', '.join(t.key for t in tasks)}")
    log_info("main", f"输出目录 {cfg.output_root}")
    if cfg.years or cfg.max_stocks:
        log_info("main", f"样本限制 years={cfg.years} max_stocks={cfg.max_stocks}")

    heartbeat = Heartbeat(interval_sec=args.heartbeat_sec)
    t0 = time.perf_counter()
    if not args.no_heartbeat:
        heartbeat.start()
    try:
        succeeded, failed = _run_tasks(
            tasks,
            cfg,
            heartbeat,
            fail_fast=args.fail_fast,
            rebuild_result=bool(args.rebuild_result),
        )
    finally:
        heartbeat.stop()

    _print_summary(succeeded, failed, time.perf_counter() - t0)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
