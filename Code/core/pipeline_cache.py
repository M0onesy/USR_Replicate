from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.config import RunConfig
from core.engine import ReplicationResult, refresh_replication_result_views, run_cn_replication
from core.logging_utils import log_info, log_step


_MEMORY_CACHE: dict[str, ReplicationResult] = {}


def _signature_hash(cfg: RunConfig) -> str:
    payload = json.dumps(cfg.cache_signature(), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _primary_cache_root(cfg: RunConfig) -> Path:
    root = Path(cfg.runtime_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _legacy_cache_roots(cfg: RunConfig) -> List[Path]:
    roots: List[Path] = []
    for candidate in [Path(cfg.final_result_root) / "checkpoints", Path(cfg.output_root) / "checkpoints"]:
        if candidate not in roots:
            roots.append(candidate)
    return roots


def _all_cache_roots(cfg: RunConfig) -> List[Path]:
    roots = [_primary_cache_root(cfg)]
    for candidate in _legacy_cache_roots(cfg):
        if candidate not in roots and candidate.exists():
            roots.append(candidate)
    return roots


def _disk_cache_path(cfg: RunConfig) -> Path:
    return _primary_cache_root(cfg) / f"replication_result_{_signature_hash(cfg)}.pkl"


def _meta_path(cfg: RunConfig) -> Path:
    return _primary_cache_root(cfg) / f"replication_result_{_signature_hash(cfg)}.meta.json"


def _refresh_result(result: ReplicationResult, cfg: RunConfig) -> ReplicationResult:
    cfg.export_fidelity_env()
    if getattr(result, "runtime_root", None) is None:
        result.runtime_root = Path(cfg.runtime_root)
    if getattr(result, "output_root", None) is None:
        result.output_root = Path(cfg.final_result_root)
    refreshed = refresh_replication_result_views(
        result,
        proc_root=cfg.proc_root,
        external_data_root=cfg.external_data_root,
        paper_tail_root=cfg.paper_tail_root,
        paper_tail_weighting=cfg.paper_tail_weighting,
        refresh_paper_tail=cfg.refresh_paper_tail,
        strict_final_export=cfg.strict_final_export,
    )
    if getattr(refreshed, "runtime_root", None) is None:
        refreshed.runtime_root = Path(cfg.runtime_root)
    refreshed.output_root = Path(cfg.final_result_root)
    return refreshed


def _load_result_pickle(cache_path: Path, *, log_label: str, cfg: RunConfig) -> ReplicationResult:
    log_step("cache", f"{log_label}: {cache_path}")
    t0 = time.perf_counter()
    with cache_path.open("rb") as fh:
        result = pickle.load(fh)
    result = _refresh_result(result, cfg)
    log_info("cache", f"{cache_path.name} loaded in {time.perf_counter() - t0:.2f}s")
    return result


def _exact_cache_candidates(cfg: RunConfig) -> List[Path]:
    file_name = f"replication_result_{_signature_hash(cfg)}.pkl"
    candidates: List[Path] = []
    for root in _all_cache_roots(cfg):
        path = root / file_name
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates


def _iter_meta_paths(cfg: RunConfig) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in _all_cache_roots(cfg):
        for meta_path in root.glob("replication_result_*.meta.json"):
            try:
                resolved = meta_path.resolve()
            except Exception:
                resolved = meta_path
            if resolved in seen:
                continue
            seen.add(resolved)
            yield meta_path


def _fallback_cache_candidates(cfg: RunConfig) -> List[Dict[str, Any]]:
    exact_paths = {path.resolve() for path in _exact_cache_candidates(cfg)}
    candidates: List[Dict[str, Any]] = []
    for meta_path in _iter_meta_paths(cfg):
        cache_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".pkl"))
        if not cache_path.exists():
            continue
        try:
            if cache_path.resolve() in exact_paths:
                continue
        except Exception:
            pass
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception as exc:
            log_info("cache", f"skip broken meta {meta_path.name}: {exc!r}")
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
    cfg.export_fidelity_env()
    sig = _signature_hash(cfg)
    log_step("pipeline", f"building ReplicationResult (signature {sig})")
    t0 = time.perf_counter()
    result = run_cn_replication(**cfg.to_kwargs())
    result = _refresh_result(result, cfg)
    elapsed = time.perf_counter() - t0
    log_info("pipeline", f"ReplicationResult built in {elapsed:.1f}s")

    _MEMORY_CACHE[sig] = result
    cache_path = _disk_cache_path(cfg)
    try:
        with cache_path.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with _meta_path(cfg).open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "signature": cfg.cache_signature(),
                    "signature_hash": sig,
                    "build_seconds": elapsed,
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "runtime_root": str(cfg.runtime_root),
                    "final_result_root": str(cfg.final_result_root),
                },
                fh,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        log_info("pipeline", f"cache written -> {cache_path}")
    except Exception as exc:
        log_info("pipeline", f"cache write skipped due to error: {exc!r}")
    return result


def get_existing_result(cfg: RunConfig, *, allow_fallback: bool = True) -> ReplicationResult:
    sig = _signature_hash(cfg)
    cache_errors: List[str] = []

    if sig in _MEMORY_CACHE:
        log_info("cache", f"memory cache hit: {sig}")
        result = _refresh_result(_MEMORY_CACHE[sig], cfg)
        _MEMORY_CACHE[sig] = result
        return result

    for cache_path in _exact_cache_candidates(cfg):
        try:
            result = _load_result_pickle(cache_path, log_label="exact cache hit", cfg=cfg)
            _MEMORY_CACHE[sig] = result
            return result
        except Exception as exc:
            cache_errors.append(f"{cache_path.name}: {exc!r}")
            log_info("cache", f"exact cache load failed, trying next candidate: {exc!r}")

    if allow_fallback:
        for candidate in _fallback_cache_candidates(cfg):
            fallback_path = Path(candidate["cache_path"])
            meta = dict(candidate.get("meta", {}))
            try:
                result = _load_result_pickle(fallback_path, log_label="reusing completed historical cache", cfg=cfg)
                _MEMORY_CACHE[sig] = result
                log_info(
                    "cache",
                    "export-mode reuse active: skipped pipeline rebuild and loaded "
                    f"{fallback_path.name} (original signature {meta.get('signature_hash', 'unknown')}).",
                )
                return result
            except Exception as exc:
                cache_errors.append(f"{fallback_path.name}: {exc!r}")
                log_info("cache", f"fallback cache load failed, continue scanning: {exc!r}")

    details = f" Attempted caches: {'; '.join(cache_errors)}." if cache_errors else ""
    roots = ", ".join(str(path) for path in _all_cache_roots(cfg))
    raise RuntimeError(
        "No reusable ReplicationResult was found."
        f" Searched checkpoint roots: {roots}.{details} "
        "If this run is driven by main.py, switch ACTIVE_MAIN_PROFILE to a rebuild profile "
        "or set rebuild_result=True and restart=True in Code/core/config.py. "
        "If this is a standalone figure/table script, rerun it with --allow-build."
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
    log_info("cache", "no reusable cache hit; starting an explicit rebuild")
    return build_result(cfg)


def clear_memory_cache() -> None:
    _MEMORY_CACHE.clear()
