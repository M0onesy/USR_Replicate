from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import RunConfig
from core.engine import ReplicationResult, _copy_alias_files
from core.io_utils import table_path, tables_dir, _atomic_to_csv
from core.logging_utils import log_render, log_step
from core.runner import run_standalone


TAG = "table_v"
ROMAN = "V"


def generate(result: ReplicationResult, cfg: RunConfig) -> Path:
    canonical = table_path(result, ROMAN)
    alias = tables_dir(result) / "Table_10_paper_style_factor_sharpes.csv"
    df = result.paper_table_v.copy()
    log_step(TAG, f"使用 paper_tail Table V 数据，共 {df.shape[0]} 行。")
    log_render(TAG, f"写入 {canonical.name}")
    _atomic_to_csv(df, canonical, index=False, encoding="utf-8-sig")
    _copy_alias_files(canonical, [alias])
    return canonical


if __name__ == "__main__":
    raise SystemExit(run_standalone(TAG, generate))
