# -*- coding: utf-8 -*-
# python_verison == 3.10

import argparse
import faulthandler
import json
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd


DEFAULT_HOST = "140.206.44.234"
DEFAULT_BATCH_SIZE = 50
DEFAULT_PERIOD_NAME = "min5"
DEFAULT_BEGIN_DATE = 20040101
DEFAULT_END_DATE = 20161231
EVENTS_FILE_NAME = "events.jsonl"
FAILED_SYMBOLS_FILE_NAME = "failed_symbols.jsonl"
FAULTHANDLER_FILE_NAME = "faulthandler.log"
DATA_FILE_NAME = "data.bz2"
DATA_FILE_TMP_NAME = "data.bz2.tmp"
METADATA_SENTINEL = "__YHDATA_METADATA__"

TYPE_CONFIG = {
    "EXTRA_STOCK_A": {"active_api": "get_code_list", "source_type": "EXTRA_STOCK_A"},
}

_FAULT_HANDLER_FILE = None


@dataclass
class WorkerOutcome:
    returncode: int
    elapsed: float


def print_status(message):
    print(message, flush=True)


def as_symbol_list(values):
    if values is None:
        return []
    if isinstance(values, pd.Series):
        values = values.tolist()
    elif isinstance(values, (str, bytes)):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = list(values)
    return sorted({str(item).strip() for item in values if str(item).strip()})


def chunked(values, size):
    if size <= 0:
        raise ValueError("size must be positive")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def sanitize_path_part(part):
    text = str(part).strip()
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid else ch for ch in text)
    return cleaned or "_empty_"


def load_api_modules():
    import AmazingData as Api
    from api_AmazingData_professional import get_AmazingData_api

    return Api, get_AmazingData_api


def initialize_base_api(host):
    Api, get_amazingdata_api = load_api_modules()
    get_amazingdata_api(host=host)
    return Api, Api.BaseData()


def normalize_calendar_dates(calendar_values):
    calendar_texts = [str(item).replace("-", "").strip() for item in list(calendar_values) if str(item).replace("-", "").strip()]
    if not calendar_texts:
        raise RuntimeError("AmazingData calendar is empty")
    return int(calendar_texts[0]), int(calendar_texts[-1])


def resolve_default_date_range(calendar_values):
    calendar_begin_date, calendar_end_date = normalize_calendar_dates(calendar_values)
    begin_date = max(calendar_begin_date, DEFAULT_BEGIN_DATE)
    end_date = min(calendar_end_date, DEFAULT_END_DATE)
    if begin_date > end_date:
        raise RuntimeError(
            "Configured default date range "
            f"{DEFAULT_BEGIN_DATE}-{DEFAULT_END_DATE} is outside AmazingData calendar range "
            f"{calendar_begin_date}-{calendar_end_date}"
        )
    return begin_date, end_date


def load_all_symbols(bapi, storage_type, config):
    source_type = config["source_type"]
    active_fn = getattr(bapi, config["active_api"])
    active_symbols = as_symbol_list(active_fn(security_type=source_type))
    hist_symbols = as_symbol_list(bapi.get_hist_code_list(security_type=source_type))
    return sorted(set(active_symbols) | set(hist_symbols)), len(active_symbols), len(hist_symbols)


def make_run_id():
    return datetime.now().strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"


def ensure_run_dir(output_root, run_id=None):
    env_run_dir = os.environ.get("YHDATA_RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
    else:
        run_dir = output_root / "_runs" / (run_id or make_run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(file_path, payload):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_event(run_dir, event, **fields):
    payload = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event}
    payload.update(fields)
    append_jsonl(run_dir / EVENTS_FILE_NAME, payload)


def record_failed_symbol(run_dir, storage_type, symbol, returncode, attempt):
    append_jsonl(
        run_dir / FAILED_SYMBOLS_FILE_NAME,
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "storage_type": storage_type,
            "symbol": symbol,
            "returncode": returncode,
            "attempt": attempt,
        },
    )


def enable_fault_handler(log_path):
    global _FAULT_HANDLER_FILE
    if _FAULT_HANDLER_FILE is None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _FAULT_HANDLER_FILE = log_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_HANDLER_FILE, all_threads=True)


def count_frames_and_rows(value):
    if isinstance(value, pd.DataFrame):
        return 1, int(len(value))
    if isinstance(value, dict):
        file_count = 0
        row_count = 0
        for nested in value.values():
            nested_files, nested_rows = count_frames_and_rows(nested)
            file_count += nested_files
            row_count += nested_rows
        return file_count, row_count
    return 0, 0


def save_dataframe_atomic(frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / DATA_FILE_NAME
    tmp_path = output_dir / DATA_FILE_TMP_NAME
    try:
        pd.to_pickle(frame, tmp_path, compression="bz2")
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_nested_frames(value, output_dir, stats):
    if isinstance(value, pd.DataFrame):
        save_dataframe_atomic(value, output_dir)
        stats["files"] += 1
        stats["rows"] += int(len(value))
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            save_nested_frames(nested, output_dir / sanitize_path_part(key), stats)
        return
    stats["skipped"] += 1
    print_status(f"[WARN] skip non-tabular leaf: path={output_dir} type={type(value).__name__}")


def collect_completed_symbols(type_dir):
    completed = set()
    if not type_dir.exists():
        return completed
    for file_path in type_dir.rglob(DATA_FILE_NAME):
        if not file_path.is_file():
            continue
        if file_path.stat().st_size <= 0:
            continue
        relative = file_path.relative_to(type_dir)
        if relative.parts:
            completed.add(relative.parts[0])
    return completed


def filter_existing_symbols(symbols, type_dir):
    completed = collect_completed_symbols(type_dir)
    if not completed:
        return list(symbols), completed
    return [symbol for symbol in symbols if sanitize_path_part(symbol) not in completed], completed


def launch_worker_subprocess(
    script_path,
    storage_type,
    codes,
    begin_date,
    end_date,
    output_root,
    batch_size,
    skip_existing,
    run_dir,
):
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        "--storage-type",
        storage_type,
        "--codes",
        *list(codes),
        "--begin-date",
        str(begin_date),
        "--end-date",
        str(end_date),
        "--output-root",
        str(output_root),
        "--batch-size",
        str(batch_size),
    ]
    command.append("--skip-existing" if skip_existing else "--no-skip-existing")
    child_env = os.environ.copy()
    child_env["YHDATA_RUN_DIR"] = str(run_dir)
    child_env["YHDATA_FAULTHANDLER_LOG"] = str(run_dir / FAULTHANDLER_FILE_NAME)
    child_env["YHDATA_HOST"] = DEFAULT_HOST

    started = time.monotonic()
    completed = subprocess.run(command, cwd=str(script_path.parent), env=child_env, check=False)
    return WorkerOutcome(returncode=completed.returncode, elapsed=time.monotonic() - started)


def extract_metadata_payload(stdout_text):
    for line in reversed(stdout_text.splitlines()):
        if line.startswith(METADATA_SENTINEL):
            return json.loads(line[len(METADATA_SENTINEL) :])
    raise RuntimeError("metadata subprocess did not emit metadata payload")


def fetch_export_metadata(script_path, requested_storage_types, allowed_codes, output_root, run_dir):
    command = [sys.executable, str(script_path), "--metadata"]
    if requested_storage_types:
        command.extend(["--storage-type", *requested_storage_types])
    if allowed_codes:
        command.extend(["--codes", *allowed_codes])
    command.extend(["--output-root", str(output_root)])

    child_env = os.environ.copy()
    child_env["YHDATA_RUN_DIR"] = str(run_dir)
    child_env["YHDATA_FAULTHANDLER_LOG"] = str(run_dir / FAULTHANDLER_FILE_NAME)
    child_env["YHDATA_HOST"] = DEFAULT_HOST

    completed = subprocess.run(
        command,
        cwd=str(script_path.parent),
        env=child_env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stdout_tail = completed.stdout[-1000:] if completed.stdout else ""
        stderr_tail = completed.stderr[-1000:] if completed.stderr else ""
        stdout_safe = stdout_tail.encode("ascii", "backslashreplace").decode("ascii")
        stderr_safe = stderr_tail.encode("ascii", "backslashreplace").decode("ascii")
        raise RuntimeError(
            f"metadata subprocess failed with returncode={completed.returncode} "
            f"stdout_tail={stdout_safe!r} stderr_tail={stderr_safe!r}"
        )
    return extract_metadata_payload(completed.stdout)


def export_type_with_workers(
    storage_type,
    config,
    begin_date,
    end_date,
    output_root,
    run_dir,
    batch_size,
    skip_existing,
    symbol_loader,
    worker_launcher,
):
    all_symbols, active_count, hist_count = symbol_loader(storage_type, config)
    type_dir = output_root / storage_type
    type_dir.mkdir(parents=True, exist_ok=True)

    pending_symbols = list(all_symbols)
    existing_completed = set()
    if skip_existing:
        pending_symbols, existing_completed = filter_existing_symbols(pending_symbols, type_dir)

    if not pending_symbols:
        print_status(
            f"[DONE][type] storage_type={storage_type} symbols={len(all_symbols)} active={active_count} "
            f"hist={hist_count} existing={len(existing_completed)} pending=0 batches=0 failed=0 elapsed=0.000s"
        )
        log_event(
            run_dir,
            "type_done",
            storage_type=storage_type,
            symbols=len(all_symbols),
            active=active_count,
            hist=hist_count,
            existing=len(existing_completed),
            pending=0,
            failed=0,
            elapsed=0.0,
        )
        return {"symbols": len(all_symbols), "existing": len(existing_completed), "failed": 0, "success": 0}

    batch_total = (len(pending_symbols) + batch_size - 1) // batch_size
    print_status(
        f"[START][type] storage_type={storage_type} symbols={len(all_symbols)} active={active_count} "
        f"hist={hist_count} existing={len(existing_completed)} pending={len(pending_symbols)} "
        f"batches={batch_total} begin_date={begin_date} end_date={end_date}"
    )
    log_event(
        run_dir,
        "type_start",
        storage_type=storage_type,
        symbols=len(all_symbols),
        active=active_count,
        hist=hist_count,
        existing=len(existing_completed),
        pending=len(pending_symbols),
        batches=batch_total,
        begin_date=begin_date,
        end_date=end_date,
    )

    type_started = time.monotonic()
    queue = deque({"codes": list(batch), "attempt": 1} for batch in chunked(pending_symbols, batch_size))
    task_index = 0
    success_symbols = 0
    failed_symbols = 0

    while queue:
        task = queue.popleft()
        task_codes = list(task["codes"])
        if skip_existing:
            task_codes, _ = filter_existing_symbols(task_codes, type_dir)
        if not task_codes:
            log_event(run_dir, "batch_skipped_existing", storage_type=storage_type, attempt=task["attempt"])
            continue

        task_index += 1
        print_status(
            f"[START][batch] storage_type={storage_type} task={task_index} symbols={len(task_codes)} "
            f"attempt={task['attempt']} queue_remaining={len(queue)}"
        )
        log_event(
            run_dir,
            "batch_start",
            storage_type=storage_type,
            task=task_index,
            symbols=len(task_codes),
            attempt=task["attempt"],
            codes=task_codes,
            queue_remaining=len(queue),
        )

        outcome = worker_launcher(
            storage_type=storage_type,
            codes=task_codes,
            begin_date=begin_date,
            end_date=end_date,
            output_root=output_root,
            batch_size=batch_size,
            skip_existing=skip_existing,
            run_dir=run_dir,
        )

        if outcome.returncode == 0:
            success_symbols += len(task_codes)
            print_status(
                f"[DONE][batch] storage_type={storage_type} task={task_index} symbols={len(task_codes)} "
                f"attempt={task['attempt']} elapsed={outcome.elapsed:.3f}s"
            )
            log_event(
                run_dir,
                "batch_done",
                storage_type=storage_type,
                task=task_index,
                symbols=len(task_codes),
                attempt=task["attempt"],
                codes=task_codes,
                elapsed=round(outcome.elapsed, 3),
            )
            continue

        print_status(
            f"[FAIL][batch] storage_type={storage_type} task={task_index} symbols={len(task_codes)} "
            f"attempt={task['attempt']} returncode={outcome.returncode} elapsed={outcome.elapsed:.3f}s"
        )
        log_event(
            run_dir,
            "batch_fail",
            storage_type=storage_type,
            task=task_index,
            symbols=len(task_codes),
            attempt=task["attempt"],
            codes=task_codes,
            returncode=outcome.returncode,
            elapsed=round(outcome.elapsed, 3),
        )

        if task["attempt"] < 2:
            retry_task = {"codes": task_codes, "attempt": task["attempt"] + 1}
            queue.appendleft(retry_task)
            print_status(
                f"[RETRY][batch] storage_type={storage_type} symbols={len(task_codes)} "
                f"next_attempt={retry_task['attempt']}"
            )
            log_event(
                run_dir,
                "batch_retry",
                storage_type=storage_type,
                symbols=len(task_codes),
                next_attempt=retry_task["attempt"],
                codes=task_codes,
                returncode=outcome.returncode,
            )
            continue

        if len(task_codes) > 1:
            midpoint = len(task_codes) // 2
            left_codes = task_codes[:midpoint]
            right_codes = task_codes[midpoint:]
            queue.appendleft({"codes": right_codes, "attempt": 1})
            queue.appendleft({"codes": left_codes, "attempt": 1})
            print_status(
                f"[SPLIT][batch] storage_type={storage_type} symbols={len(task_codes)} "
                f"left={len(left_codes)} right={len(right_codes)}"
            )
            log_event(
                run_dir,
                "batch_split",
                storage_type=storage_type,
                symbols=len(task_codes),
                left_codes=left_codes,
                right_codes=right_codes,
                returncode=outcome.returncode,
            )
            continue

        failed_symbol = task_codes[0]
        failed_symbols += 1
        record_failed_symbol(run_dir, storage_type, failed_symbol, outcome.returncode, task["attempt"])
        print_status(
            f"[FAIL][symbol] storage_type={storage_type} symbol={failed_symbol} "
            f"returncode={outcome.returncode}"
        )
        log_event(
            run_dir,
            "symbol_fail",
            storage_type=storage_type,
            symbol=failed_symbol,
            returncode=outcome.returncode,
            attempt=task["attempt"],
        )

    elapsed = time.monotonic() - type_started
    print_status(
        f"[DONE][type] storage_type={storage_type} symbols={len(all_symbols)} active={active_count} "
        f"hist={hist_count} existing={len(existing_completed)} success={success_symbols} failed={failed_symbols} "
        f"elapsed={elapsed:.3f}s"
    )
    log_event(
        run_dir,
        "type_done",
        storage_type=storage_type,
        symbols=len(all_symbols),
        active=active_count,
        hist=hist_count,
        existing=len(existing_completed),
        success=success_symbols,
        failed=failed_symbols,
        elapsed=round(elapsed, 3),
    )
    return {"symbols": len(all_symbols), "existing": len(existing_completed), "failed": failed_symbols, "success": success_symbols}


def get_requested_storage_types(requested_storage_types):
    if not requested_storage_types:
        return list(TYPE_CONFIG.items())
    selected = []
    for storage_type in requested_storage_types:
        if storage_type not in TYPE_CONFIG:
            raise KeyError(f"Unknown storage_type: {storage_type}")
        selected.append((storage_type, TYPE_CONFIG[storage_type]))
    return selected


def create_parent_symbol_loader_from_metadata(metadata):
    def _loader(storage_type, config):
        type_info = metadata["types"][storage_type]
        return list(type_info["symbols"]), int(type_info["active_count"]), int(type_info["hist_count"])

    return _loader


def run_metadata(args):
    output_root = Path(args.output_root).resolve()
    run_dir = ensure_run_dir(output_root)
    enable_fault_handler(Path(os.environ.get("YHDATA_FAULTHANDLER_LOG", run_dir / FAULTHANDLER_FILE_NAME)))

    host = os.environ.get("YHDATA_HOST", DEFAULT_HOST)
    requested_types = get_requested_storage_types(args.storage_type)
    allowed_codes = set(args.codes or [])
    _, bapi = initialize_base_api(host)
    calendar = bapi.get_calendar()
    begin_date, end_date = resolve_default_date_range(calendar)

    types_payload = {}
    for storage_type, config in requested_types:
        symbols, active_count, hist_count = load_all_symbols(bapi, storage_type, config)
        if allowed_codes:
            symbols = [symbol for symbol in symbols if symbol in allowed_codes]
        types_payload[storage_type] = {
            "symbols": symbols,
            "active_count": active_count,
            "hist_count": hist_count,
        }

    payload = {"begin_date": begin_date, "end_date": end_date, "types": types_payload}
    print_status(METADATA_SENTINEL + json.dumps(payload, ensure_ascii=False))
    return 0


def run_worker(args):
    output_root = Path(args.output_root).resolve()
    run_dir = ensure_run_dir(output_root)
    enable_fault_handler(Path(os.environ.get("YHDATA_FAULTHANDLER_LOG", run_dir / FAULTHANDLER_FILE_NAME)))

    storage_type = args.storage_type[0]
    type_dir = output_root / storage_type
    codes = list(dict.fromkeys(args.codes or []))
    host = os.environ.get("YHDATA_HOST", DEFAULT_HOST)
    if not codes:
        raise ValueError("worker mode requires --codes")
    if args.begin_date is None or args.end_date is None:
        raise ValueError("worker mode requires --begin-date and --end-date")

    print_status(
        f"[START][worker] storage_type={storage_type} symbols={len(codes)} "
        f"begin_date={args.begin_date} end_date={args.end_date}"
    )
    log_event(
        run_dir,
        "worker_start",
        storage_type=storage_type,
        symbols=len(codes),
        codes=codes,
        begin_date=args.begin_date,
        end_date=args.end_date,
    )

    try:
        Api, bapi = initialize_base_api(host)
        calendar = bapi.get_calendar()
        mapi = Api.MarketData(calendar)
        period = getattr(Api.constant.Period, DEFAULT_PERIOD_NAME).value
        result = mapi.query_kline(
            code_list=codes,
            period=period,
            begin_date=args.begin_date,
            end_date=args.end_date,
        )
        expected_files, expected_rows = count_frames_and_rows(result)
        batch_stats = {"files": 0, "rows": 0, "skipped": 0}
        save_nested_frames(result, type_dir, batch_stats)
        print_status(
            f"[DONE][worker] storage_type={storage_type} symbols={len(codes)} "
            f"files={batch_stats['files']} rows={batch_stats['rows']} skipped={batch_stats['skipped']}"
        )
        log_event(
            run_dir,
            "worker_done",
            storage_type=storage_type,
            symbols=len(codes),
            codes=codes,
            files=batch_stats["files"],
            rows=batch_stats["rows"],
            skipped=batch_stats["skipped"],
            expected_files=expected_files,
            expected_rows=expected_rows,
        )
        return 0
    except Exception as exc:
        print_status(
            f"[FAIL][worker] storage_type={storage_type} symbols={len(codes)} "
            f"error={exc.__class__.__name__}: {exc}"
        )
        log_event(
            run_dir,
            "worker_fail",
            storage_type=storage_type,
            symbols=len(codes),
            codes=codes,
            error_type=exc.__class__.__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1


def run_parent(args):
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_run_dir(output_root)
    enable_fault_handler(run_dir / FAULTHANDLER_FILE_NAME)

    print_status(f"[RUN] run_dir={run_dir}")
    log_event(run_dir, "export_start", output_root=str(output_root))

    script_path = Path(__file__).resolve()
    requested_type_names = [storage_type for storage_type, _ in get_requested_storage_types(args.storage_type)]
    metadata = fetch_export_metadata(
        script_path=script_path,
        requested_storage_types=requested_type_names,
        allowed_codes=args.codes or [],
        output_root=output_root,
        run_dir=run_dir,
    )
    default_begin_date = int(metadata["begin_date"])
    default_end_date = int(metadata["end_date"])
    begin_date = args.begin_date if args.begin_date is not None else default_begin_date
    end_date = args.end_date if args.end_date is not None else default_end_date

    requested_types = get_requested_storage_types(args.storage_type)
    symbol_loader = create_parent_symbol_loader_from_metadata(metadata)

    def _launch(**kwargs):
        return launch_worker_subprocess(
            script_path=script_path,
            storage_type=kwargs["storage_type"],
            codes=kwargs["codes"],
            begin_date=kwargs["begin_date"],
            end_date=kwargs["end_date"],
            output_root=kwargs["output_root"],
            batch_size=kwargs["batch_size"],
            skip_existing=kwargs["skip_existing"],
            run_dir=kwargs["run_dir"],
        )

    print_status(
        f"[START][export] output={output_root} types={','.join(storage_type for storage_type, _ in requested_types)} "
        f"begin_date={begin_date} end_date={end_date} period={DEFAULT_PERIOD_NAME} "
        f"code_batch_size={args.batch_size} skip_existing={args.skip_existing}"
    )
    export_started = time.monotonic()
    failed_symbols = 0
    for storage_type, config in requested_types:
        summary = export_type_with_workers(
            storage_type=storage_type,
            config=config,
            begin_date=begin_date,
            end_date=end_date,
            output_root=output_root,
            run_dir=run_dir,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            symbol_loader=symbol_loader,
            worker_launcher=_launch,
        )
        failed_symbols += summary["failed"]

    elapsed = time.monotonic() - export_started
    print_status(f"[DONE][export] output={output_root} failed_symbols={failed_symbols} elapsed={elapsed:.3f}s")
    log_event(
        run_dir,
        "export_done",
        output_root=str(output_root),
        begin_date=begin_date,
        end_date=end_date,
        failed_symbols=failed_symbols,
        elapsed=round(elapsed, 3),
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Export AmazingData kline data with crash-isolated workers.")
    parser.add_argument("--metadata", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help="Run a single worker batch.")
    parser.add_argument("--storage-type", nargs="+", help="Storage type name(s) to export.")
    parser.add_argument("--codes", nargs="+", help="Specific symbol code(s) to export.")
    parser.add_argument("--begin-date", type=int, help="Inclusive begin date in YYYYMMDD format.")
    parser.add_argument("--end-date", type=int, help="Inclusive end date in YYYYMMDD format.")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "Kline_Data"),
        help="Output root directory.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Symbols per worker batch.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip symbols with an existing non-empty data.bz2 file.",
    )
    return parser


def validate_args(args):
    if args.metadata and args.worker:
        raise ValueError("--metadata and --worker cannot be used together")
    if args.worker and not args.storage_type:
        raise ValueError("worker mode requires --storage-type")
    if args.worker and len(args.storage_type) != 1:
        raise ValueError("worker mode requires exactly one --storage-type")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.metadata:
        return run_metadata(args)
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
