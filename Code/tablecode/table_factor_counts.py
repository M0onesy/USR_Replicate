from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult
from core.io_utils import _atomic_to_csv, diagnostics_dir
from core.logging_utils import log_render, log_step
from core.runner import run_standalone


TAG = "table_factor_counts"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = diagnostics_dir(result) / "paper_factor_count_diagnostics.csv"
    df = result.paper_factor_counts
    log_step(TAG, f"使用因子数诊断数据，共 {df.shape[0]} 行。")
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
