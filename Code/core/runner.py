from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.logging_utils import log_done, log_start, log_warn


GenerateFn = Callable[[ReplicationResult, RunConfig], Path]


def run_generator(
    tag: str,
    generate: GenerateFn,
    result: Optional[ReplicationResult] = None,
    cfg: Optional[RunConfig] = None,
) -> Path:
    cfg = cfg or RunConfig()
    if result is None:
        from core.pipeline_cache import get_result

        result = get_result(cfg, allow_build=True, allow_fallback=True)

    log_start(tag, "脚本开始执行")
    t0 = time.perf_counter()
    try:
        output_path = generate(result, cfg)
    except Exception as exc:
        log_warn(tag, f"生成失败: {exc!r}")
        raise
    elapsed = time.perf_counter() - t0
    log_done(tag, f"已输出 -> {output_path}  (用时 {elapsed:.2f}s)")
    return output_path


def run_standalone(tag: str, generate: GenerateFn) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=f"单独生成 {tag}")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--proc-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-build", action="store_true", help="当没有可复用缓存时，允许显式重建 ReplicationResult")
    args = parser.parse_args()

    cfg = RunConfig()
    if args.years is not None:
        cfg.years = args.years
    if args.max_stocks is not None:
        cfg.max_stocks = args.max_stocks
    if args.proc_root is not None:
        cfg.proc_root = Path(args.proc_root)
    if args.output_root is not None:
        cfg.output_root = Path(args.output_root)
    cfg.save_plots = False
    cfg.restart = False

    from core.pipeline_cache import get_result

    result = get_result(
        cfg,
        allow_build=bool(args.allow_build),
        allow_fallback=True,
    )
    run_generator(tag, generate, result=result, cfg=cfg)
    return 0
