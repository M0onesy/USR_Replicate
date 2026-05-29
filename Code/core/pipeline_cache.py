from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

from core.config import RunConfig
from core.engine import ReplicationResult, refresh_replication_result_views, run_cn_replication
from core.logging_utils import log_info, log_step


_MEMORY_CACHE: dict[str, ReplicationResult] = {}


def _signature_hash(cfg: RunConfig) -> str:
    payload = json.dumps(cfg.cache_signature(), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cache_root(cfg: RunConfig) -> Path:
    root = Path(cfg.output_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _disk_cache_path(cfg: RunConfig) -> Path:
    return _cache_root(cfg) / f"replication_result_{_signature_hash(cfg)}.pkl"


def _meta_path(cfg: RunConfig) -> Path:
    return _cache_root(cfg) / f"replication_result_{_signature_hash(cfg)}.meta.json"


def _load_result_pickle(cache_path: Path, *, log_label: str) -> ReplicationResult:
    log_step("cache", f"{log_label} {cache_path.name}，正在载入…")
    t0 = time.perf_counter()
    with open(cache_path, "rb") as fh:
        result = pickle.load(fh)
    result = refresh_replication_result_views(result)
    log_info("cache", f"{cache_path.name} 载入完成，用时 {time.perf_counter() - t0:.2f}s")
    return result


def _fallback_cache_candidates(cfg: RunConfig) -> List[Dict[str, Any]]:
    exact_cache_path = _disk_cache_path(cfg).resolve()
    candidates: List[Dict[str, Any]] = []
    for meta_path in _cache_root(cfg).glob("replication_result_*.meta.json"):
        cache_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".pkl"))
        if not cache_path.exists():
            continue
        try:
            if cache_path.resolve() == exact_cache_path:
                continue
        except Exception:
            pass
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception as exc:
            log_info("cache", f"跳过损坏的缓存元数据 {meta_path.name}: {exc!r}")
            continue

        built_at_text = str(meta.get("built_at", "") or "")
        try:
            built_at = dt.datetime.fromisoformat(built_at_text).timestamp() if built_at_text else cache_path.stat().st_mtime
        except Exception:
            built_at = cache_path.stat().st_mtime

        candidates.append(
            {
                "cache_path": cache_path,
                "meta_path": meta_path,
                "meta": meta,
                "sort_key": float(built_at),
            }
        )

    candidates.sort(key=lambda item: float(item["sort_key"]), reverse=True)
    return candidates


def build_result(cfg: RunConfig) -> ReplicationResult:
    sig = _signature_hash(cfg)
    log_step("pipeline", f"开始构建 ReplicationResult（缓存键 {sig}）—— 这是最耗时的一步")
    t0 = time.perf_counter()
    result = run_cn_replication(**cfg.to_kwargs())
    elapsed = time.perf_counter() - t0
    log_info("pipeline", f"ReplicationResult 构建完成，用时 {elapsed:.1f}s")

    _MEMORY_CACHE[sig] = result
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
    except Exception as exc:
        log_info("pipeline", f"磁盘缓存写入失败（忽略，不影响结果）: {exc!r}")

    return result


def get_existing_result(cfg: RunConfig, *, allow_fallback: bool = True) -> ReplicationResult:
    sig = _signature_hash(cfg)
    cache_errors: List[str] = []

    if sig in _MEMORY_CACHE:
        log_info("cache", f"命中内存缓存（键 {sig}），直接复用现有 ReplicationResult")
        return _MEMORY_CACHE[sig]

    cache_path = _disk_cache_path(cfg)
    if cache_path.exists():
        try:
            result = _load_result_pickle(cache_path, log_label="命中精确磁盘缓存")
            _MEMORY_CACHE[sig] = result
            return result
        except Exception as exc:
            cache_errors.append(f"{cache_path.name}: {exc!r}")
            log_info("cache", f"精确磁盘缓存载入失败，将尝试其它已完成缓存: {exc!r}")

    if allow_fallback:
        for candidate in _fallback_cache_candidates(cfg):
            fallback_path = Path(candidate["cache_path"])
            meta = dict(candidate.get("meta", {}))
            try:
                result = _load_result_pickle(fallback_path, log_label="复用旧版已完成缓存")
                _MEMORY_CACHE[sig] = result
                log_info(
                    "cache",
                    "发现旧 checkpoint，但当前运行是导出模式，已自动跳过 pipeline。"
                    f" 复用缓存 {fallback_path.name}（原签名 {meta.get('signature_hash', 'unknown')}）。",
                )
                return result
            except Exception as exc:
                cache_errors.append(f"{fallback_path.name}: {exc!r}")
                log_info("cache", f"候选旧缓存 {fallback_path.name} 载入失败，继续尝试其它缓存: {exc!r}")

    details = f" 已尝试的缓存: {'; '.join(cache_errors)}。" if cache_errors else ""
    raise RuntimeError(
        "当前没有可复用的 ReplicationResult。"
        f" 预期目录: {_cache_root(cfg)}。"
        f"{details} 若这是 main.py 的配置式运行，请去 Code/core/config.py 切换到重建 profile，"
        "或把当前 ACTIVE_MAIN_PROFILE 对应配置改成 rebuild_result=True、restart=True 后重新运行 main.py。"
        " 若这是单图/单表脚本独立运行，请显式加上 --allow-build。"
    )


def get_result(
    cfg: RunConfig,
    *,
    allow_build: bool = True,
    allow_fallback: bool = False,
) -> ReplicationResult:
    try:
        return get_existing_result(cfg, allow_fallback=allow_fallback)
    except RuntimeError:
        if not allow_build:
            raise
    log_info("cache", "未命中可复用缓存，准备显式重建 ReplicationResult")
    return build_result(cfg)


def clear_memory_cache() -> None:
    _MEMORY_CACHE.clear()
