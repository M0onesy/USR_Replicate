"""Task registry for the submission-version figure and table generators."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Task:
    key: str
    kind: str
    module: str
    desc: str

    def load_generate(self) -> Callable:
        mod = importlib.import_module(self.module)
        return getattr(mod, "generate")


_FIGURE_TASKS: List[Task] = [
    Task("fig1", "figure", "figCode.figure_01", "Figure 1  yearwise balanced changing-universe HF factor counts"),
    Task("fig2", "figure", "figCode.figure_02", "Figure 2  fixed-intersection balanced HF factor counts"),
    Task("fig4", "figure", "figCode.figure_04", "Figure 4  continuous PCA portfolio weights"),
    Task("fig7", "figure", "figCode.figure_07", "Figure 7  local-vs-global continuous PCA weight GC"),
    Task("fig10", "figure", "figCode.figure_10", "Figure 10 factor-structure time variation"),
    Task("fig12", "figure", "figCode.figure_12", "Figure 12 expected intraday and overnight returns"),
    Task("fig13", "figure", "figCode.figure_13", "Figure 13 cumulative factor returns"),
    Task("fig14", "figure", "figCode.figure_14", "Figure 14 industry portfolio pricing"),
    Task("fig15", "figure", "figCode.figure_15", "Figure 15 size-value portfolio pricing"),
]

_TABLE_TASKS: List[Task] = [
    Task("table_i", "table", "tableCode.table_i", "Table I   continuous/jump return summary statistics"),
    Task("table_ii", "table", "tableCode.table_ii", "Table II  balanced/unbalanced panel PCA results"),
    Task("table_iii", "table", "tableCode.table_iii", "Table III industry/FFC factor generalized correlations"),
    Task("table_v", "table", "tableCode.table_v", "Table V   intraday/overnight/daily Sharpe ratios"),
]

ALL_TASKS: List[Task] = _FIGURE_TASKS + _TABLE_TASKS
_BY_KEY = {task.key: task for task in ALL_TASKS}


def all_tasks() -> List[Task]:
    return list(ALL_TASKS)


def figure_tasks() -> List[Task]:
    return list(_FIGURE_TASKS)


def table_tasks() -> List[Task]:
    return list(_TABLE_TASKS)


def core_tasks() -> List[Task]:
    return list(ALL_TASKS)


def auxiliary_table_tasks() -> List[Task]:
    return []


def get_task(key: str) -> Optional[Task]:
    return _BY_KEY.get(key)


def resolve_keys(selectors: List[str]) -> List[Task]:
    chosen: List[Task] = []
    seen = set()

    def _add(task: Task) -> None:
        if task.key not in seen:
            seen.add(task.key)
            chosen.append(task)

    for selector in selectors:
        value = str(selector).strip().lower()
        if value in {"all", "submission", "core"}:
            for task in ALL_TASKS:
                _add(task)
        elif value in {"figures", "figure", "figs", "fig"}:
            for task in _FIGURE_TASKS:
                _add(task)
        elif value in {"tables", "table"}:
            for task in _TABLE_TASKS:
                _add(task)
        else:
            task = get_task(value)
            if task is None:
                raise KeyError(
                    f"Unknown task selector {selector!r}. "
                    "Use all / figures / tables or one of: "
                    f"{', '.join(sorted(_BY_KEY))}"
                )
            _add(task)
    return chosen
