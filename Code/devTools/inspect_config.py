from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def main() -> int:
    import main as runner

    args = runner._parse_args([])
    config = runner._load_yaml(runner._as_path(args.config, runner.DEFAULT_CONFIG_PATH))
    stages = runner._selected_stages(config, args)
    figures = runner._selected_tasks(config, args, "figures")
    tables = runner._selected_tasks(config, args, "tables")
    data_steps = runner._selected_data_steps(config, args)
    tasks = runner._resolve_render_tasks(figures, tables, stages)
    cfg = runner._build_run_config(config, args, save_plots=any(task.kind == "figure" for task in tasks))
    print("stages:", ", ".join(stages))
    print("figures:", ", ".join(figures))
    print("tables:", ", ".join(tables))
    print("data_steps:", ", ".join(data_steps) if data_steps else "(none)")
    print("final_result_root:", cfg.final_result_root)
    print("figures_dir:", cfg.final_result_root / "figures")
    print("tables_dir:", cfg.final_result_root / "tables")
    print("proc_root:", cfg.proc_root)
    print("runtime_root:", cfg.runtime_root)
    print("external_data_root:", cfg.external_data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
