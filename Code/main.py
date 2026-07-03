from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig, SUBMISSION_FROZEN_INDUSTRIES
from core.engine import STRICT_BALANCED_SAMPLE
from core.logging_utils import log_done, log_info, log_warn
from core.registry import Task, all_tasks, resolve_keys
from core.runner import run_generator
from core.submission_fast import build_submission_fast_result, submission_fast_runtime_root


DEFAULT_CONFIG_PATH = _CODE_DIR / "config.yaml"
DEFAULT_FIGURES = ("fig1", "fig2", "fig4", "fig7", "fig10", "fig12", "fig13", "fig14", "fig15")
DEFAULT_TABLES = ("table_i", "table_ii", "table_iii", "table_v")
DEFAULT_STAGES = ("figures", "tables")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise SystemExit("PyYAML is required for Code/config.yaml. Please run: pip install -r requirements.txt") from exc
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _as_list(value: Any, default: Sequence[str] = ()) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)


def _as_path(value: Any, default: str | Path) -> Path:
    raw = default if value in (None, "") else value
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = _CODE_DIR.parent / path
    return path.resolve()


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submission-version runner for the A-share Pelger replication.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path. Default: Code/config.yaml")
    parser.add_argument("--stages", help="Comma-separated stages: data,figures,tables")
    parser.add_argument("--data-steps", help="Comma-separated data steps: get_apidb,preprocess_panels,mom_5min")
    parser.add_argument("--figures", help="Comma-separated figure tasks, e.g. fig1,fig2,fig4")
    parser.add_argument("--tables", help="Comma-separated table tasks, e.g. table_i,table_ii")
    parser.add_argument("--all", action="store_true", help="Run data, figures and tables using configured/default selections.")
    parser.add_argument("--list", action="store_true", help="List available submission tasks and exit.")
    parser.add_argument("--output-root", help="Override final result root.")
    parser.add_argument("--proc-root", help="Override processed-data root.")
    parser.add_argument("--runtime-root", help="Override runtime root.")
    parser.add_argument("--external-data-root", help="Override external data root.")
    parser.add_argument("--paper-tail-root", help="Override paper_tail root.")
    parser.add_argument("--rf-file", help="Override risk-free-rate csv.")
    parser.add_argument("--workers", type=int, help="Override worker count for data steps/core config.")
    parser.add_argument("--panel-workers", type=int, help="Override panel preprocessing workers.")
    parser.add_argument("--paper-workers", type=int, help="Override paper workers.")
    parser.add_argument("--rolling-workers", type=int, help="Override rolling workers.")
    parser.add_argument("--memory-budget-gb", type=float, help="Override memory budget.")
    parser.add_argument("--refresh", action="store_true", help="Refresh selected data step outputs where supported.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed figure/table task.")
    parser.add_argument("--no-fail-fast", action="store_true", help="Continue after failed figure/table tasks.")
    parser.add_argument("--refresh-paper-tail", action="store_true", help="Refresh paper_tail views before rendering.")
    return parser.parse_args(argv)


def _print_tasks() -> None:
    print("Available submission tasks:")
    print("[figures]")
    for task in all_tasks():
        if task.kind == "figure":
            print(f"  {task.key:<8} {task.desc}")
    print("[tables]")
    for task in all_tasks():
        if task.kind == "table":
            print(f"  {task.key:<8} {task.desc}")
    print("[data steps]")
    print("  get_apidb          fetch raw data through the external API helper")
    print("  preprocess_panels  build strict_balanced/paper_lenient processed panels")
    print("  mom_5min           build 5-minute MOM factor")


def _selected_stages(config: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    if args.all:
        return ["data", "figures", "tables"]
    stages = _as_list(args.stages, default=())
    if stages:
        return stages
    return _as_list(config.get("stages"), default=DEFAULT_STAGES)


def _selected_tasks(config: Mapping[str, Any], args: argparse.Namespace, kind: str) -> list[str]:
    cli_value = getattr(args, kind)
    if cli_value:
        return _as_list(cli_value)
    configured = _as_list(config.get(kind), default=())
    if configured:
        return configured
    return list(DEFAULT_FIGURES if kind == "figures" else DEFAULT_TABLES)


def _selected_data_steps(config: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    if args.all and not args.data_steps:
        return ["get_apidb", "preprocess_panels", "mom_5min"]
    if args.data_steps:
        return _as_list(args.data_steps)
    return _as_list(config.get("data_steps"), default=())


def _build_run_config(config: Mapping[str, Any], args: argparse.Namespace, *, save_plots: bool) -> RunConfig:
    run_cfg = config.get("run") if isinstance(config.get("run"), Mapping) else {}
    cfg = RunConfig()
    cfg.proc_root = _as_path(args.proc_root, _get(config, "paths", "proc_root", default=cfg.proc_root))
    cfg.runtime_root = _as_path(args.runtime_root, _get(config, "paths", "runtime_root", default=cfg.runtime_root))
    final_root = _as_path(args.output_root, _get(config, "paths", "final_result_root", default=cfg.final_result_root))
    cfg.final_result_root = final_root
    cfg.output_root = final_root
    cfg.external_data_root = _as_path(args.external_data_root, _get(config, "paths", "external_data_root", default=cfg.external_data_root))
    cfg.paper_tail_root = _as_path(args.paper_tail_root, _get(config, "paths", "paper_tail_root", default=cfg.paper_tail_root))
    cfg.balanced_mode = str(run_cfg.get("balanced_mode") or STRICT_BALANCED_SAMPLE)
    cfg.strict_final_export = bool(run_cfg.get("strict_final_export", True))
    cfg.refresh_paper_tail = bool(run_cfg.get("refresh_paper_tail", False) or args.refresh_paper_tail)
    cfg.restart = bool(run_cfg.get("restart", False))
    cfg.workers = args.workers if args.workers is not None else run_cfg.get("workers")
    cfg.paper_workers = args.paper_workers if args.paper_workers is not None else run_cfg.get("paper_workers")
    cfg.rolling_workers = args.rolling_workers if args.rolling_workers is not None else run_cfg.get("rolling_workers")
    cfg.memory_budget_gb = args.memory_budget_gb if args.memory_budget_gb is not None else run_cfg.get("memory_budget_gb")
    cfg.industry_factors_frozen = list(SUBMISSION_FROZEN_INDUSTRIES)
    cfg.save_plots = save_plots
    rf_file = args.rf_file or _get(config, "paths", "rf_file", default=None)
    if rf_file:
        os.environ["PELGER_RF_FILE"] = str(_as_path(rf_file, rf_file))
    cfg.export_fidelity_env()
    return cfg


def _fail_fast(config: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if args.no_fail_fast:
        return False
    if args.fail_fast:
        return True
    return bool(_get(config, "run", "fail_fast", default=True))


def _run_data_steps(config: Mapping[str, Any], args: argparse.Namespace, cfg: RunConfig, steps: Sequence[str]) -> None:
    if not steps:
        log_info("data", "No data steps selected.")
        return
    data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    for step in steps:
        key = str(step).strip().lower()
        if key == "get_apidb":
            from dataPrepare import step_00_get_apidb

            log_info("data", "Running step_00_get_apidb with its own CLI defaults.")
            step_00_get_apidb.main()
        elif key == "preprocess_panels":
            from dataPrepare.step_01_preprocess_panels import preprocess_cn_data

            step_cfg = data_cfg.get("preprocess_panels") if isinstance(data_cfg.get("preprocess_panels"), Mapping) else {}
            log_info("data", "Running step_01_preprocess_panels.")
            preprocess_cn_data(
                proc_root=cfg.proc_root,
                years=step_cfg.get("years"),
                max_stocks=step_cfg.get("max_stocks"),
                refresh=bool(args.refresh or step_cfg.get("refresh", False)),
                workers=args.workers if args.workers is not None else cfg.workers,
                panel_workers=args.panel_workers if args.panel_workers is not None else step_cfg.get("panel_workers"),
                compress_symbol_returns=bool(step_cfg.get("compress_symbol_returns", False)),
            )
        elif key == "mom_5min":
            from dataPrepare import step_02_build_mom_5min

            step_cfg = data_cfg.get("mom_5min") if isinstance(data_cfg.get("mom_5min"), Mapping) else {}
            argv = [
                "--proc-root",
                str(_CODE_DIR.parent / "Data" / "proc_Data" / "mom_5min"),
                "--lookback-bars",
                str(step_cfg.get("lookback_bars", 48)),
                "--skip-bars",
                str(step_cfg.get("skip_bars", 1)),
                "--winner-pct",
                str(step_cfg.get("winner_pct", 0.3)),
                "--loser-pct",
                str(step_cfg.get("loser_pct", 0.3)),
                "--min-stocks",
                str(step_cfg.get("min_stocks", 5)),
            ]
            if args.workers is not None:
                argv.extend(["--workers", str(args.workers)])
            log_info("data", "Running step_02_build_mom_5min.")
            old_argv = sys.argv[:]
            try:
                sys.argv = ["step_02_build_mom_5min.py", *argv]
                step_02_build_mom_5min.main()
            finally:
                sys.argv = old_argv
        else:
            raise ValueError(f"Unknown data step: {step}")


def _resolve_render_tasks(figures: Sequence[str], tables: Sequence[str], stages: Sequence[str]) -> list[Task]:
    selectors: list[str] = []
    stage_set = {str(stage).strip().lower() for stage in stages}
    if "figures" in stage_set:
        selectors.extend(figures)
    if "tables" in stage_set:
        selectors.extend(tables)
    return resolve_keys(selectors) if selectors else []


def _write_export_summary(cfg: RunConfig, outputs: Mapping[str, str], failures: Sequence[tuple[str, str]]) -> None:
    path = submission_fast_runtime_root(cfg) / "diagnostics" / "export_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated": dict(outputs), "failures": [{"task": key, "error": err} for key, err in failures]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_render_tasks(tasks: Sequence[Task], cfg: RunConfig, *, fail_fast: bool) -> int:
    if not tasks:
        log_info("main", "No figure/table tasks selected.")
        return 0
    log_info("main", f"Building lightweight strict ReplicationResult for {len(tasks)} tasks.")
    result = build_submission_fast_result(cfg, strict_fail=fail_fast)
    outputs: dict[str, str] = {}
    failures: list[tuple[str, str]] = []
    for idx, task in enumerate(tasks, start=1):
        log_info("main", f"[{idx}/{len(tasks)}] {task.key} - {task.desc}")
        try:
            generate = task.load_generate()
            output_path = run_generator(task.key, generate, result=result, cfg=cfg)
            outputs[task.key] = str(output_path)
        except Exception as exc:  # noqa: BLE001
            failures.append((task.key, f"{type(exc).__name__}: {exc}"))
            log_warn("main", f"{task.key} failed: {type(exc).__name__}: {exc}")
            if fail_fast:
                traceback.print_exc()
                break
    _write_export_summary(cfg, outputs, failures)
    log_done("main", f"Generated {len(outputs)} tasks; failures={len(failures)}.")
    return 0 if not failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_yaml(_as_path(args.config, DEFAULT_CONFIG_PATH))
    if args.list:
        _print_tasks()
        return 0

    stages = _selected_stages(config, args)
    figures = _selected_tasks(config, args, "figures")
    tables = _selected_tasks(config, args, "tables")
    data_steps = _selected_data_steps(config, args)
    tasks = _resolve_render_tasks(figures, tables, stages)
    cfg = _build_run_config(config, args, save_plots=any(task.kind == "figure" for task in tasks))
    fail_fast = _fail_fast(config, args)

    log_info("main", f"Stages: {', '.join(stages)}")
    log_info("main", f"Figures: {', '.join(figures)}")
    log_info("main", f"Tables: {', '.join(tables)}")
    log_info("main", f"Data steps: {', '.join(data_steps) if data_steps else '(none)'}")
    log_info("main", f"Final result root: {cfg.final_result_root}")
    log_info("main", f"Runtime root: {submission_fast_runtime_root(cfg)}")

    start = time.perf_counter()
    if "data" in {stage.lower() for stage in stages}:
        _run_data_steps(config, args, cfg, data_steps)
    status = _run_render_tasks(tasks, cfg, fail_fast=fail_fast)
    log_done("main", f"Finished in {time.perf_counter() - start:.1f}s.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
