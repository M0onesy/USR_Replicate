from __future__ import annotations

import sys as _sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig, SUBMISSION_FROZEN_INDUSTRIES
from core.engine import STRICT_BALANCED_SAMPLE
from core.logging_utils import log_done, log_info
from core.submission_fast import export_submission_table_i_fast, submission_fast_runtime_root


def _build_cfg(output_root: str | None) -> RunConfig:
    cfg = RunConfig()
    cfg.balanced_mode = STRICT_BALANCED_SAMPLE
    cfg.strict_final_export = True
    cfg.industry_factors_frozen = list(SUBMISSION_FROZEN_INDUSTRIES)
    cfg.refresh_paper_tail = False
    cfg.save_plots = False
    if output_root:
        target = Path(output_root).expanduser().resolve()
        cfg.final_result_root = target
        cfg.output_root = target
    return cfg


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Lightweight Table I export aligned with submission Figure 1 and Figure 2 panel definitions."
    )
    parser.add_argument("--output-root", default=None, help="Optional final table output root.")
    args = parser.parse_args()

    cfg = _build_cfg(args.output_root)
    log_info("submission_fast", f"Using strict panel mode: {cfg.balanced_mode}")
    log_info("submission_fast", f"Final table root: {cfg.final_result_root}")
    log_info("submission_fast", f"Submission-fast runtime root: {submission_fast_runtime_root(cfg)}")
    output_path = export_submission_table_i_fast(cfg)
    log_done("submission_fast", f"Aligned Table I export finished -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
