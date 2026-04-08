from __future__ import annotations
import argparse
import sys
import pandas as pd
from pathlib import Path

def get_output_csv_path(bz2_path: Path) -> Path:
    if not bz2_path.name.lower().endswith(".bz2"):
        raise ValueError(f"Not a .bz2 file: {bz2_path}")
    return bz2_path.with_name(f"{bz2_path.name[:-4]}.csv")


def find_bz2_files(target: Path) -> list[Path]:
    if target.is_file():
        if target.name.lower().endswith(".bz2"):
            return [target]
        raise ValueError(f"Expected a .bz2 file, got: {target}")
    if target.is_dir():
        return sorted(path for path in target.iterdir() if path.is_file() and path.name.lower().endswith(".bz2"))
    raise FileNotFoundError(f"Path does not exist: {target}")


def convert_file(bz2_path: Path) -> Path:
    csv_path = get_output_csv_path(bz2_path)
    df = pd.read_pickle(bz2_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Converted: {bz2_path.name} -> {csv_path.name} ({len(df)} rows)")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert pandas DataFrame pickles stored in .bz2 files to CSV.")
    parser.add_argument("path", nargs="?", default=".", help="A .bz2 file or a directory containing .bz2 files. Defaults to the current directory.")
    args = parser.parse_args()
    target = Path(args.path).expanduser().resolve()
    try:
        bz2_files = find_bz2_files(target)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not bz2_files:
        print(f"No .bz2 files found in: {target}")
        return 0
    failed = False
    for bz2_path in bz2_files:
        try:
            convert_file(bz2_path)
        except Exception as exc:
            failed = True
            print(f"Failed: {bz2_path.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
