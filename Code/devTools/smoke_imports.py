from __future__ import annotations

import importlib
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def main() -> int:
    from prepareCore.registry import all_tasks

    failures: list[tuple[str, str]] = []
    for task in all_tasks():
        try:
            module = importlib.import_module(task.module)
            if not hasattr(module, "generate"):
                raise AttributeError(f"{task.module} has no generate()")
            print(f"[OK] {task.key:<8} {task.module}")
        except Exception as exc:  # noqa: BLE001
            failures.append((task.key, repr(exc)))
            print(f"[FAIL] {task.key:<8} {exc!r}")
    if failures:
        print(f"[SUMMARY] failures={len(failures)}")
        return 1
    print(f"[SUMMARY] imported {len(all_tasks())} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
