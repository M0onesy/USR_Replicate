from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import List

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.config import MainLaunchProfile, RunConfig, get_active_main_profile, profile_to_run_config
from core.engine import _atomic_to_csv, _plot_status_row
from core.logging_utils import Heartbeat, log_done, log_info, log_warn
from core.pipeline_cache import get_result
from core.registry import Task, all_tasks, resolve_keys
from core.runner import run_generator


def _ensure_no_cli_args(argv: List[str]) -> None:
    if len(argv) <= 1:
        return
    joined = " ".join(argv[1:])
    raise SystemExit(
        "main.py 现在不再接受命令行参数。\n"
        f"检测到传入参数：{joined}\n"
        "请改 Code/core/config.py 中的 ACTIVE_MAIN_PROFILE 或 MAIN_RUN_PROFILES，然后重新运行 main.py。"
    )


def _is_figure_task(task: Task) -> bool:
    return task.kind == "figure"


def _has_figure_tasks(tasks: List[Task]) -> bool:
    return any(_is_figure_task(task) for task in tasks)


def _write_plot_status_for_export_only(
    tasks: List[Task],
    succeeded: list[tuple[str, str]],
    failed: list[tuple[str, str]],
    cfg: RunConfig,
) -> None:
    diagnostics_dir = Path(cfg.runtime_root) / "diagnostics"
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
                    Path(cfg.final_result_root) / "figures" / f"{figure_id}.png",
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
    print("可用任务列表（请在 core/config.py 的 task_selectors 中使用这些短名）")
    print("=" * 78)
    print("[图 / figure]")
    for task in all_tasks():
        if task.kind == "figure":
            print(f"  {task.key:<8} {task.desc}")
    print("[表 / table]")
    for task in all_tasks():
        if task.kind == "table":
            print(f"  {task.key:<10} {task.desc}")
    print("=" * 78)


def _resolve_tasks(profile_name: str, profile: MainLaunchProfile) -> List[Task]:
    if profile.list_tasks_only:
        _print_task_list()
        raise SystemExit(0)
    try:
        tasks = resolve_keys(list(profile.task_selectors))
    except KeyError as exc:
        raise ValueError(f"profile {profile_name!r} 的 task_selectors 配置无效：{exc}") from exc
    if not tasks:
        raise ValueError(f"profile {profile_name!r} 没有解析出任何任务，请检查 task_selectors。")
    return tasks


def _resolve_result(cfg: RunConfig, *, rebuild_result: bool):
    if rebuild_result:
        log_info("main", "当前 profile 为显式重建模式：将按配置重建新的 ReplicationResult。")
        return get_result(cfg, allow_build=True, allow_fallback=False)
    log_info("main", "当前 profile 为复用优先模式：不会重建上游 ReplicationResult；若无可复用结果将直接报错。")
    log_info("main", "交稿轻量 strict 图表请运行 python Code/export_submission_core_fast.py。")
    return get_result(cfg, allow_build=False, allow_fallback=True)


def _run_tasks(
    tasks: List[Task],
    cfg: RunConfig,
    heartbeat: Heartbeat,
    *,
    fail_fast: bool,
    rebuild_result: bool,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    log_info("main", "准备 ReplicationResult（命中缓存则秒回，否则按当前 profile 决定是否重建）…")
    heartbeat.set_status("准备 ReplicationResult", done=0, total=len(tasks))
    result = _resolve_result(cfg, rebuild_result=rebuild_result)
    heartbeat.clear_progress_log()
    log_done("main", "ReplicationResult 就绪，开始逐个生成图表和表格。")

    succeeded: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for idx, task in enumerate(tasks, start=1):
        heartbeat.set_status(f"{task.key} ({task.kind})", done=len(succeeded) + len(failed), total=len(tasks))
        log_info("main", f"[{idx}/{len(tasks)}] 开始 {task.key} - {task.desc}")
        try:
            generate = task.load_generate()
            output_path = run_generator(task.key, generate, result=result, cfg=cfg)
            succeeded.append((task.key, str(output_path)))
        except Exception as exc:
            failed.append((task.key, f"{type(exc).__name__}: {exc}"))
            log_warn("main", f"任务 {task.key} 失败：{type(exc).__name__}: {exc}")
            if fail_fast:
                log_warn("main", "当前 profile 开启了 fail_fast，主流程将立即中止。")
                traceback.print_exc()
                break
        finally:
            heartbeat.set_status(f"{task.key} 结束", done=len(succeeded) + len(failed), total=len(tasks))

    if not rebuild_result and _has_figure_tasks(tasks):
        _write_plot_status_for_export_only(tasks, succeeded, failed, cfg)
    return succeeded, failed


def _print_summary(succeeded: list[tuple[str, str]], failed: list[tuple[str, str]], total_elapsed: float) -> None:
    print("=" * 78)
    print("运行汇总")
    print("=" * 78)
    print(f"成功 {len(succeeded)} 项 / 失败 {len(failed)} 项 / 总用时 {total_elapsed:.1f}s")
    if succeeded:
        print("[成功]")
        for key, path in succeeded:
            print(f"  {key:<10} -> {path}")
    if failed:
        print("[失败]")
        for key, err in failed:
            print(f"  {key:<10} : {err}")
    print("=" * 78)


def _build_run_config(profile: MainLaunchProfile, tasks: List[Task]) -> RunConfig:
    return profile_to_run_config(profile, save_plots=_has_figure_tasks(tasks))


def _log_resume_hint(cfg: RunConfig) -> None:
    checkpoint_root = Path(cfg.runtime_root) / "checkpoints"
    run_state_path = checkpoint_root / "run_state.json"
    if not run_state_path.exists():
        return
    try:
        state = json.loads(run_state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    rolling_done = len(state.get("completed_rolling_chunks") or [])
    paper_done = len(state.get("completed_paper_years") or [])
    status = str(state.get("status") or "")
    if cfg.restart:
        log_warn("main", f"本次配置 restart=True，将忽略已有 checkpoint（rolling={rolling_done}, paper_years={paper_done}）。")
        return
    if rolling_done or paper_done:
        log_info(
            "main",
            f"检测到已有 checkpoint：rolling={rolling_done} 块、paper={paper_done} 年；若 semantic signature 一致将自动续跑。",
        )


def main() -> int:
    _ensure_no_cli_args(sys.argv)
    profile_name, profile = get_active_main_profile()
    tasks = _resolve_tasks(profile_name, profile)
    cfg = _build_run_config(profile, tasks)

    log_info("main", f"当前 main profile：{profile_name}")
    log_info("main", f"任务数 {len(tasks)}：{', '.join(task.key for task in tasks)}")
    log_info("main", f"任务选择器：{', '.join(profile.task_selectors)}")
    log_info(
        "main",
        f"运行模式：{'重建上游 ReplicationResult' if profile.rebuild_result else '优先复用已有 ReplicationResult'}",
    )
    log_info("main", f"主样本模式：{cfg.balanced_mode}")
    log_info("main", f"运行目录 {cfg.runtime_root}")
    log_info("main", f"最终结果目录 {cfg.final_result_root}")
    log_info(
        "main",
        "执行层参数："
        f" restart={cfg.restart}"
        f" workers={cfg.workers}"
        f" paper_workers={cfg.paper_workers}"
        f" rolling_workers={cfg.rolling_workers}"
        f" memory_budget_gb={cfg.memory_budget_gb}",
    )
    if cfg.years or cfg.max_stocks:
        log_info("main", f"样本限制 years={cfg.years} max_stocks={cfg.max_stocks}")
    _log_resume_hint(cfg)

    heartbeat = Heartbeat(interval_sec=profile.heartbeat_sec)
    heartbeat.attach_progress_log(Path(cfg.runtime_root) / "diagnostics" / "progress.jsonl")
    t0 = time.perf_counter()
    if profile.enable_heartbeat:
        heartbeat.start()
    try:
        succeeded, failed = _run_tasks(
            tasks,
            cfg,
            heartbeat,
            fail_fast=profile.fail_fast,
            rebuild_result=profile.rebuild_result,
        )
    finally:
        heartbeat.stop()

    _print_summary(succeeded, failed, time.perf_counter() - t0)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
