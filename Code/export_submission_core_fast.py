from __future__ import annotations

import sys as _sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig, SUBMISSION_FROZEN_INDUSTRIES
from core.engine import STRICT_BALANCED_SAMPLE
from core.logging_utils import log_done, log_info
from core.submission_fast import export_submission_core_fast, submission_fast_runtime_root


def _build_cfg(output_root: str | None) -> RunConfig:
    cfg = RunConfig()
    cfg.balanced_mode = STRICT_BALANCED_SAMPLE
    cfg.strict_final_export = True
    cfg.industry_factors_frozen = list(SUBMISSION_FROZEN_INDUSTRIES)
    cfg.refresh_paper_tail = False
    cfg.save_plots = True
    if output_root:
        target = Path(output_root).expanduser().resolve()
        cfg.final_result_root = target
        cfg.output_root = target
    return cfg


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight submission-core figure export without paper_tables.")
    parser.add_argument("--output-root", default=None, help="Optional final figure output root.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow legacy Figure 13 fallback if strict yearly alignment fails.",
    )
    args = parser.parse_args()

    cfg = _build_cfg(args.output_root)
    log_info("submission_fast", f"Using strict panel mode: {cfg.balanced_mode}")
    log_info("submission_fast", f"Final figure root: {cfg.final_result_root}")
    log_info("submission_fast", f"Submission-fast runtime root: {submission_fast_runtime_root(cfg)}")
    export_submission_core_fast(cfg, strict_fail=not bool(args.allow_fallback))
    log_done("submission_fast", "Lightweight submission-core export finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
