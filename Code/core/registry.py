"""
Task registry for all figure and table generators.
"""

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
    Task("fig1", "figure", "figcode.figure_01", "Figure 1  跨年非平衡但逐年平衡面板 HF 因子个数"),
    Task("fig2", "figure", "figcode.figure_02", "Figure 2  全样本固定交集平衡面板逐年 HF 因子个数"),
    Task("fig3", "figure", "figcode.figure_03", "Figure 3  代理因子组合权重热图"),
    Task("fig4", "figure", "figcode.figure_04", "Figure 4  连续 PCA 因子组合权重热图"),
    Task("fig5", "figure", "figcode.figure_05", "Figure 5  月频 PCA 因子组合权重热图"),
    Task("fig6", "figure", "figcode.figure_06", "Figure 6  载荷时间变化"),
    Task("fig7", "figure", "figcode.figure_07", "Figure 7  前七个连续 PCA 因子的局部权重 GC 时间变化"),
    Task("fig8", "figure", "figcode.figure_08", "Figure 8  随时间变化的组合权重"),
    Task("fig9", "figure", "figcode.figure_09", "Figure 9  随时间变化的解释方差"),
    Task("fig10", "figure", "figcode.figure_10", "Figure 10 因子结构时间变化分解"),
    Task("fig11", "figure", "figcode.figure_11", "Figure 11 连续因子结构分解"),
    Task("fig12", "figure", "figcode.figure_12", "Figure 12 预期日内与隔夜超额收益"),
    Task("fig13", "figure", "figcode.figure_13", "Figure 13 标准化累计因子收益"),
    Task("fig14", "figure", "figcode.figure_14", "Figure 14 行业组合资产定价"),
    Task("fig15", "figure", "figcode.figure_15", "Figure 15 规模-价值组合资产定价"),
]

_TABLE_TASKS: List[Task] = [
    Task("table_i", "table", "tablecode.table_i", "Table I   连续/跳跃收益汇总统计"),
    Task("table_ii", "table", "tablecode.table_ii", "Table II  平衡/非平衡面板因子空间 GC"),
    Task("table_iii", "table", "tablecode.table_iii", "Table III 行业/FFC 因子 GC"),
    Task("table_iv", "table", "tablecode.table_iv", "Table IV  时间变化分解汇总"),
    Task("table_v", "table", "tablecode.table_v", "Table V   因子组与前四个连续因子的日内/隔夜/日度夏普"),
    Task("table_fc", "table", "tablecode.table_factor_counts", "扰动特征值比诊断表"),
    Task("table_w", "table", "tablecode.table_weights", "因子权重表"),
    Task("table_fr", "table", "tablecode.table_factor_returns", "因子收益摘要"),
    Task("table_cov", "table", "tablecode.table_coverage", "复现覆盖度报告"),
]

ALL_TASKS: List[Task] = _FIGURE_TASKS + _TABLE_TASKS
_CORE_TABLE_KEYS = {"table_i", "table_ii", "table_iii", "table_iv", "table_v"}
_CORE_TASKS: List[Task] = _FIGURE_TASKS + [task for task in _TABLE_TASKS if task.key in _CORE_TABLE_KEYS]
_AUX_TABLE_TASKS: List[Task] = [task for task in _TABLE_TASKS if task.key not in _CORE_TABLE_KEYS]
_BY_KEY = {t.key: t for t in ALL_TASKS}


def all_tasks() -> List[Task]:
    return list(ALL_TASKS)


def figure_tasks() -> List[Task]:
    return list(_FIGURE_TASKS)


def table_tasks() -> List[Task]:
    return list(_TABLE_TASKS)


def core_tasks() -> List[Task]:
    return list(_CORE_TASKS)


def auxiliary_table_tasks() -> List[Task]:
    return list(_AUX_TABLE_TASKS)


def get_task(key: str) -> Optional[Task]:
    return _BY_KEY.get(key)


def resolve_keys(selectors: List[str]) -> List[Task]:
    chosen: List[Task] = []
    seen = set()

    def _add(task: Task) -> None:
        if task.key not in seen:
            seen.add(task.key)
            chosen.append(task)

    for sel in selectors:
        s = sel.strip().lower()
        if s == "all":
            for t in _CORE_TASKS:
                _add(t)
        elif s in ("figures", "figure", "figs", "fig"):
            for t in _FIGURE_TASKS:
                _add(t)
        elif s in ("tables", "table"):
            for t in _TABLE_TASKS:
                _add(t)
        elif s in ("diagnostics", "aux", "aux_tables", "diagnostic_tables"):
            for t in _AUX_TABLE_TASKS:
                _add(t)
        else:
            task = get_task(s)
            if task is None:
                raise KeyError(f"Unknown task selector {sel!r}. Use all / figures / tables / diagnostics or a concrete task key.")
            _add(task)
    return chosen
