from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from core.config import RunConfig, SUBMISSION_FROZEN_INDUSTRIES
from core.engine import STRICT_BALANCED_SAMPLE
from core.logging_utils import log_done, log_info, log_warn
from core.runner import run_generator
from core.submission_fast import (
    build_submission_fast_result,
    export_submission_table_iii_fast,
    export_submission_table_v_fast,
    submission_fast_runtime_root,
)


AFFECTED_FIGURES = (
    ("fig12", "figcode.figure_12"),
    ("fig13", "figcode.figure_13"),
    ("fig14", "figcode.figure_14"),
    ("fig15", "figcode.figure_15"),
)


def _build_cfg(output_root: str | None, rf_file: str | None) -> RunConfig:
    cfg = RunConfig()
    cfg.balanced_mode = STRICT_BALANCED_SAMPLE
    cfg.strict_final_export = True
    cfg.industry_factors_frozen = list(SUBMISSION_FROZEN_INDUSTRIES)
    cfg.industry_info_filename = "stock_full_info_std_industry_final.csv"
    cfg.industry_mapping_filename = "\u884c\u4e1a\u6620\u5c04\u8868_\u7ec8\u7248.csv"
    cfg.refresh_paper_tail = True
    cfg.save_plots = True
    if rf_file:
        import os as _os

        _os.environ["PELGER_RF_FILE"] = str(Path(rf_file).expanduser().resolve())
    else:
        default_rf = Path(__file__).resolve().parents[1] / "无风险利率" / "risk_free.csv"
        if default_rf.exists():
            import os as _os

            _os.environ["PELGER_RF_FILE"] = str(default_rf.resolve())
    if output_root:
        target = Path(output_root).expanduser().resolve()
        cfg.final_result_root = target
        cfg.output_root = target
    return cfg


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh RF-dependent submission figures/tables using the CSMAR treasury risk-free rate."
    )
    parser.add_argument("--output-root", default=None, help="Optional final output root.")
    parser.add_argument("--rf-file", default=None, help="Optional RF csv path. Defaults to repo/无风险利率/risk_free.csv.")
    parser.add_argument("--strict-fail", action="store_true", default=True, help="Fail on any affected figure/table error.")
    args = parser.parse_args()

    cfg = _build_cfg(args.output_root, args.rf_file)
    log_info("submission_fast", f"Using strict panel mode: {cfg.balanced_mode}")
    log_info("submission_fast", f"Final output root: {cfg.final_result_root}")
    log_info("submission_fast", f"Submission-fast runtime root: {submission_fast_runtime_root(cfg)}")
    log_info("submission_fast", "Refreshing paper_tail with the current RF input, then exporting Figure 12/13/14/15 and Table III/V.")

    result = build_submission_fast_result(
        cfg,
        strict_fail=bool(args.strict_fail),
        build_factor_count_diagnostics=False,
    )
    outputs: dict[str, str] = {}
    failures: list[dict[str, str]] = []

    for task_key, module_name in AFFECTED_FIGURES:
        module = __import__(module_name, fromlist=["generate"])
        generate = getattr(module, "generate")
        try:
            output_path = run_generator(task_key, generate, result=result, cfg=cfg)
            outputs[task_key] = str(output_path)
        except Exception as exc:
            failures.append({"task": task_key, "error": repr(exc)})
            if args.strict_fail:
                raise
            log_warn("submission_fast", f"{task_key} failed: {exc!r}")

    for table_key, exporter in [
        ("table_iii", export_submission_table_iii_fast),
        ("table_v", export_submission_table_v_fast),
    ]:
        try:
            output_path = exporter(cfg)
            outputs[table_key] = str(output_path)
        except Exception as exc:
            failures.append({"task": table_key, "error": repr(exc)})
            if args.strict_fail:
                raise
            log_warn("submission_fast", f"{table_key} failed: {exc!r}")

    summary = {
        "mode": "submission_tail_rf_fast",
        "outputs": outputs,
        "failures": failures,
    }
    summary_path = submission_fast_runtime_root(cfg) / "diagnostics" / "tail_rf_fast_export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_done("submission_fast", f"RF-dependent submission export finished -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
