"""
core/pipeline_cache.py
======================

Option A 的关键：让昂贵的上游计算**只跑一次**。

背景：
  论文复现里真正耗时的是 ``run_cn_replication``（载入面板、跳跃分解、PCA、滚动
  PCA、逐年论文表）。它产出一个 ``ReplicationResult`` 对象，里面已经装好了所有
  图、表需要的数据字段。每一张图 / 表本身都很便宜——只是从这个对象里取字段、
  画图或落 CSV。

  如果让每个 figure/table 脚本各自重跑一遍 pipeline，"全部运行"就会把几小时的
  PCA 重复二十多次。所以这里做一层缓存：

    get_result(cfg)  ->  ReplicationResult

  - 内存级缓存：同一个进程内多次调用直接复用（main.py 一键全量时受益）。
  - 磁盘级缓存：把 ReplicationResult pickle 到 output_root/checkpoints/ 下，
    figure/table 脚本"单独运行"时直接 load，无需重算。
  - 缓存键由 RunConfig.cache_signature() 决定（只看影响数值的参数）。
    参数变了 -> 缓存失效 -> 自动重算。

  这样既满足"单独跑某一篇方便调试"，又不牺牲"全部运行"的速度，并且数值与
  原始 allcode_Need.py 完全一致（计算逻辑没有改动）。
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Optional

from core.config import RunConfig
from core.engine import ReplicationResult, run_cn_replication
from core.logging_utils import log_info, log_step

# 进程内单例缓存：键是缓存签名的哈希，值是 ReplicationResult。
_MEMORY_CACHE: dict[str, ReplicationResult] = {}


def _signature_hash(cfg: RunConfig) -> str:
    """把数值相关参数序列化成稳定哈希，作为缓存键。"""
    payload = json.dumps(cfg.cache_signature(), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _disk_cache_path(cfg: RunConfig) -> Path:
    """磁盘缓存文件路径：放在 output_root/checkpoints/ 下，和 engine 的其它中间产物同级。"""
    root = Path(cfg.output_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"replication_result_{_signature_hash(cfg)}.pkl"


def _meta_path(cfg: RunConfig) -> Path:
    return Path(cfg.output_root) / "checkpoints" / f"replication_result_{_signature_hash(cfg)}.meta.json"


def build_result(cfg: RunConfig) -> ReplicationResult:
    """无条件重新运行完整 pipeline，并把结果写入内存 + 磁盘缓存。

    数据处理（重活）全部发生在这一步。返回后即可被任意多张图 / 表复用。
    """
    sig = _signature_hash(cfg)
    log_step("pipeline", f"开始构建 ReplicationResult（缓存键 {sig}）—— 这是最耗时的一步")
    t0 = time.perf_counter()

    # engine 内部会自行打印心跳 / 进度到 diagnostics/progress.jsonl。
    # save_plots 交给 cfg 控制：拆分后通常关掉，把画图交给 figcode 下的脚本。
    result = run_cn_replication(**cfg.to_kwargs())

    elapsed = time.perf_counter() - t0
    log_info("pipeline", f"ReplicationResult 构建完成，用时 {elapsed:.1f}s")

    _MEMORY_CACHE[sig] = result

    # 写磁盘缓存，供"单独运行某一篇图表脚本"复用。
    cache_path = _disk_cache_path(cfg)
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(_meta_path(cfg), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "signature": cfg.cache_signature(),
                    "signature_hash": sig,
                    "build_seconds": elapsed,
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                fh,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        log_info("pipeline", f"已写入磁盘缓存 -> {cache_path}")
    except Exception as exc:  # 缓存写失败不应阻断主流程
        log_info("pipeline", f"磁盘缓存写入失败（忽略，不影响结果）: {exc!r}")

    return result


def get_result(cfg: RunConfig, *, allow_build: bool = True) -> ReplicationResult:
    """获取 ReplicationResult，优先复用缓存。

    查找顺序：
      1. 进程内内存缓存（同一次 main.py 全量运行里，所有图表共享一个对象）。
      2. 磁盘 pickle 缓存（单独运行某一篇脚本时命中，秒级返回）。
      3. 都没有 -> 调用 build_result 重新计算（allow_build=False 时改为报错）。
    """
    sig = _signature_hash(cfg)

    # 1) 内存
    if sig in _MEMORY_CACHE:
        log_info("cache", f"命中内存缓存（键 {sig}），跳过重算")
        return _MEMORY_CACHE[sig]

    # 2) 磁盘
    cache_path = _disk_cache_path(cfg)
    if cache_path.exists():
        try:
            log_step("cache", f"命中磁盘缓存 {cache_path.name}，正在载入…")
            t0 = time.perf_counter()
            with open(cache_path, "rb") as fh:
                result = pickle.load(fh)
            _MEMORY_CACHE[sig] = result
            log_info("cache", f"磁盘缓存载入完成，用时 {time.perf_counter() - t0:.2f}s（已跳过全部重算）")
            return result
        except Exception as exc:
            log_info("cache", f"磁盘缓存载入失败，将重新计算: {exc!r}")

    # 3) 重算
    if not allow_build:
        raise RuntimeError(
            f"未找到可用的 ReplicationResult 缓存（键 {sig}），且 allow_build=False。\n"
            f"请先运行 main.py 生成缓存，或在脚本里允许构建。"
        )
    log_info("cache", "未命中任何缓存，需要重新运行完整 pipeline")
    return build_result(cfg)


def clear_memory_cache() -> None:
    """清空进程内缓存（一般用不到，调试时可手动调用）。"""
    _MEMORY_CACHE.clear()
