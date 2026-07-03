from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "Result"

FIGURES = [
    "Figure_1_number_of_hf_factors_unbalanced.png",
    "Figure_2_number_of_hf_factors_balanced.png",
    "Figure_4_continuous_pca_factor_portfolio_weights.png",
    "Figure_7_locally_estimated_continuous_factors.png",
    "Figure_10_factor_structure_time_variation_decomposition.png",
    "Figure_12_expected_intraday_and_overnight_returns.png",
    "Figure_13_cumulative_factor_returns.png",
    "Figure_14_asset_pricing_of_industry_portfolios.png",
    "Figure_15_asset_pricing_of_size_and_value_sorted_portfolios.png",
]
TABLES = [
    "Table_I_summary_statistics_for_continuous_and_jump_returns.csv",
    "Table_II_balanced_and_unbalanced_panel_results.csv",
    "Table_III_generalized_correlations_with_industry_and_ffc_factors.csv",
    "Table_V_intraday_overnight_daily_sharpe_ratios.csv",
]


def _check_file(path: Path) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    print(f"[{'OK' if ok else 'MISS'}] {path.relative_to(REPO_ROOT)}")
    return ok


def main() -> int:
    print(f"[INFO] Checking submission outputs under {RESULT_ROOT.relative_to(REPO_ROOT)}")
    print("[INFO] If these files are missing after the path refactor, run: python Code/main.py")
    checks = [_check_file(RESULT_ROOT / "figures" / name) for name in FIGURES]
    checks.extend(_check_file(RESULT_ROOT / "tables" / name) for name in TABLES)
    summary = REPO_ROOT / "Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/export_summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        failures = payload.get("failures") or []
        print(f"[INFO] export_summary failures={len(failures)}")
        checks.append(len(failures) == 0)
    else:
        print(f"[WARN] missing diagnostics summary: {summary}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
