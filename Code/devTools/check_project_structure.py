from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _check(path: str, *, should_exist: bool = True) -> bool:
    target = REPO_ROOT / path
    ok = target.exists() if should_exist else not target.exists()
    status = "OK" if ok else "FAIL"
    expectation = "exists" if should_exist else "absent"
    print(f"[{status}] {path} ({expectation})")
    return ok


def main() -> int:
    checks = [
        _check("Code/main.py"),
        _check("Code/config.yaml"),
        _check("Code/prepareCore"),
        _check("Code/dataPrepare"),
        _check("Code/figCode"),
        _check("Code/tableCode"),
        _check("Code/devTools"),
        _check("Code/core", should_exist=False),
        _check("Result"),
        _check("Data/external_Data/pelger_tail/factors/rf/risk_free.csv"),
    ]
    code_top = sorted(path.name for path in (REPO_ROOT / "Code").iterdir())
    print("[INFO] Code top-level:", ", ".join(code_top))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
