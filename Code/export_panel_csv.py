"""将严格平衡面板导出为可读 CSV。

用法很简单：只需要修改下面常量区的 `TARGET_PANEL_NAME`，
例如填 `full` 或 `year_2013`，然后直接运行本脚本即可。

脚本会自动读取：
`Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/<TARGET_PANEL_NAME>`
并在 `Code/<TARGET_PANEL_NAME>/` 下生成对应的 CSV 文件夹。
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from allcode_Need import load_proc_hf_panel


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PROC_ROOT = REPO_ROOT / "Data" / "proc_Data" / "pelger_cn_adjusted"
STRICT_BALANCED_ROOT = DEFAULT_PROC_ROOT / "panels" / "strict_balanced"

# 直接在这里改目标面板名即可，例如：full、year_2013、year_2016。
TARGET_PANEL_NAME = "full"


def _ensure_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _validate_panel_name(panel_name: str) -> str:
    panel_name = str(panel_name).strip()
    if not re.fullmatch(r"(full|year_\d{4})", panel_name):
        raise ValueError("TARGET_PANEL_NAME 只能是 full 或 year_YYYY，例如 year_2013。")
    return panel_name


def _panel_years(panel_name: str) -> list[int] | None:
    if panel_name == "full":
        return None
    return [int(panel_name.split("_", 1)[1])]


def _panel_source_path(panel_name: str) -> Path:
    return STRICT_BALANCED_ROOT / panel_name


def _clear_existing_csv(output_root: Path) -> None:
    if not output_root.exists():
        return
    for path in output_root.glob("*.csv"):
        path.unlink()


def _hf_bar_labels(panel: object, bar_times: list[str]) -> tuple[pd.Index, list[str]]:
    """为高频面板生成 `(date, bar_time)` 索引。"""
    dates_by_row = pd.Index(panel.dates)[panel.day_ids].strftime("%Y-%m-%d")
    if bar_times:
        bar_labels = [bar_times[idx % len(bar_times)] for idx in range(len(panel.day_ids))]
    else:
        n_days = max(len(panel.dates), 1)
        bars_per_day = len(panel.day_ids) // n_days if len(panel.day_ids) % n_days == 0 else 0
        if bars_per_day <= 0:
            raise ValueError("无法从面板推断每日日内 bar 数，请检查 R_5min_full 与 day_ids。")
        bar_labels = [f"bar_{idx % bars_per_day + 1:02d}" for idx in range(len(panel.day_ids))]
    return dates_by_row, bar_labels


def export_panel_csv(panel_name: str = TARGET_PANEL_NAME) -> Path:
    """读取指定严格平衡面板，并导出四张 R_* CSV 表。"""
    panel_name = _validate_panel_name(panel_name)
    years = _panel_years(panel_name)
    source_path = _panel_source_path(panel_name)
    output_root = SCRIPT_DIR / panel_name

    if not source_path.exists():
        available = sorted(p.name for p in STRICT_BALANCED_ROOT.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"找不到面板目录: {source_path}\n"
            f"可用面板: {', '.join(available)}"
        )

    panel = load_proc_hf_panel(proc_root=DEFAULT_PROC_ROOT, years=years, max_stocks=None)

    output_root.mkdir(parents=True, exist_ok=True)
    _clear_existing_csv(output_root)
    bar_times = list(panel.bar_times or [])

    date_index = [d.strftime("%Y-%m-%d") for d in panel.dates]

    df_daily = pd.DataFrame(panel.R_daily, index=date_index, columns=panel.tickers)
    df_daily.index.name = "date"
    df_daily.to_csv(output_root / "R_daily.csv", encoding="utf-8-sig")

    df_intra = pd.DataFrame(panel.R_intra, index=date_index, columns=panel.tickers)
    df_intra.index.name = "date"
    df_intra.to_csv(output_root / "R_intra.csv", encoding="utf-8-sig")

    df_night = pd.DataFrame(panel.R_night, index=date_index, columns=panel.tickers)
    df_night.index.name = "date"
    df_night.to_csv(output_root / "R_night.csv", encoding="utf-8-sig")

    if panel.R_5min_full is not None:
        dates_by_row, bar_labels = _hf_bar_labels(panel, bar_times)
        hf_index = pd.MultiIndex.from_arrays([dates_by_row, bar_labels], names=["date", "bar_time"])
        df_hf = pd.DataFrame(panel.R_5min_full, index=hf_index, columns=panel.tickers)
        df_hf.to_csv(output_root / "R_5min_full.csv", encoding="utf-8-sig")

        df_day_ids = pd.DataFrame({
            "row_id": np.arange(len(panel.day_ids), dtype=np.int32),
            "day_id": panel.day_ids,
            "date": dates_by_row,
            "bar_time": bar_labels,
        })
        df_day_ids.to_csv(output_root / "day_ids_map.csv", index=False, encoding="utf-8-sig")

    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a strict-balanced panel to CSV.")
    parser.add_argument(
        "--panel-name",
        default=TARGET_PANEL_NAME,
        help="Panel folder name under strict_balanced, e.g. full or year_2013",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = export_panel_csv(args.panel_name)
    print(f"[OK] CSV exports written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
