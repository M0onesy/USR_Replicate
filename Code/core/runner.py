"""
core/runner.py
==============

每篇图 / 表脚本共用的"运行脚手架"，统一三件事：

  1. 入口约定：每个脚本暴露一个 ``generate(result, cfg) -> Path``，只负责
     "数据处理 + 输出"，不关心结果从哪来。
  2. 结果获取：脚本既能被 main.py 调用（直接传入已算好的 result，零重算），
     也能单独 ``python figcode/figure_08.py`` 运行（自动走缓存 / 必要时构建）。
  3. 日志：自动打印"开始 / 完成 / 耗时"，满足"被运行到时打印一条相关信息"。

这样 figcode / tablecode 下的脚本只需写两个东西：一个 generate() 函数，
一行 run_standalone()。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

# 允许 `python figcode/figure_08.py` 这种单独运行时找到 core 包。
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.logging_utils import log_start, log_done, log_warn

# generate 签名：generate(result, cfg) -> 输出文件路径
GenerateFn = Callable[[ReplicationResult, RunConfig], Path]


def run_generator(
    tag: str,
    generate: GenerateFn,
    result: Optional[ReplicationResult] = None,
    cfg: Optional[RunConfig] = None,
) -> Path:
    """执行单个图 / 表的生成，带统一日志与计时。

    - result 为 None 时：通过缓存层获取（单独运行场景）。
    - result 已传入时：直接复用（main.py 全量场景，零重算）。
    """
    cfg = cfg or RunConfig()
    if result is None:
        # 延迟导入，避免 main 全量场景里产生不必要的循环依赖。
        from core.pipeline_cache import get_result
        result = get_result(cfg)

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
    """脚本被直接 `python xxx.py` 运行时的入口。

    支持少量命令行参数做小样本调试：
      --years 2016 2017   只跑指定年份
      --max-stocks 10     只用前 N 只股票
      --output-root PATH  自定义输出根目录
    """
    import argparse

    parser = argparse.ArgumentParser(description=f"单独生成 {tag}")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--proc-root", default=None)
    parser.add_argument("--output-root", default=None)
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
    # 单独运行时不让 engine 再顺带画全套图，交给当前脚本自己输出。
    cfg.save_plots = False

    run_generator(tag, generate, result=None, cfg=cfg)
    return 0
