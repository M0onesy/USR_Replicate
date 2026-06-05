from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import _atomic_to_csv, table_path
from core.logging_utils import log_render, log_step
from core.runner import run_standalone


TAG = "table_i"
ROMAN = "I"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)
    df = result.paper_table_i
    log_step(TAG, f"使用 Table I 数据，共 {df.shape[0]} 行。")
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
