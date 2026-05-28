"""
core/registry.py
================

所有图 / 表生成脚本的"总目录"。main.py 通过它来发现、列举、按名调度任务，
做到"全部运行"或"只跑某一篇"。

每个任务登记为一个 Task：
  - key:     命令行用的短名（如 "fig1" / "table_i"），唯一。
  - kind:    "figure" 或 "table"，用于分组与日志。
  - module:  脚本的模块路径（如 "figcode.figure_01"）。
  - desc:    一句话中文说明。
任务的 generate 函数统一签名 generate(result, cfg)，由 main 延迟导入后调用，
避免在仅运行单个任务时导入全部脚本。

新增一张图 / 表时，只需在这里加一行登记即可被 main 识别。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Task:
    key: str            # 命令行短名，唯一
    kind: str           # "figure" | "table"
    module: str         # 模块路径
    desc: str           # 中文说明

    def load_generate(self) -> Callable:
        """延迟导入对应模块并返回其 generate 函数。"""
        mod = importlib.import_module(self.module)
        return getattr(mod, "generate")


# ---------------------------------------------------------------------------
# 图任务（Figure 1-15）
# ---------------------------------------------------------------------------
_FIGURE_TASKS: List[Task] = [
    Task("fig1",  "figure", "figcode.figure_01", "Figure 1  非平衡面板 HF 因子个数（ER 折线）"),
    Task("fig2",  "figure", "figcode.figure_02", "Figure 2  平衡面板 HF 因子个数（ER 折线）"),
    Task("fig3",  "figure", "figcode.figure_03", "Figure 3  代理因子组合权重热图"),
    Task("fig4",  "figure", "figcode.figure_04", "Figure 4  连续 PCA 因子组合权重热图"),
    Task("fig5",  "figure", "figcode.figure_05", "Figure 5  月频 PCA 因子组合权重热图"),
    Task("fig6",  "figure", "figcode.figure_06", "Figure 6  载荷时间变化（全部 GC 序列）"),
    Task("fig7",  "figure", "figcode.figure_07", "Figure 7  局部连续因子（前 4 个 GC）"),
    Task("fig8",  "figure", "figcode.figure_08", "Figure 8  随时间变化的组合权重"),
    Task("fig9",  "figure", "figcode.figure_09", "Figure 9  随时间变化的解释方差"),
    Task("fig10", "figure", "figcode.figure_10", "Figure 10 因子结构时间变化分解（avg GC + 解释度）"),
    Task("fig11", "figure", "figcode.figure_11", "Figure 11 连续因子结构分解（min/mean GC）"),
    Task("fig12", "figure", "figcode.figure_12", "Figure 12 预期日内与隔夜收益（分段柱状）"),
    Task("fig13", "figure", "figcode.figure_13", "Figure 13 因子累计收益（Factor 1）"),
    Task("fig14", "figure", "figcode.figure_14", "Figure 14 行业组合资产定价（外部数据/占位）"),
    Task("fig15", "figure", "figcode.figure_15", "Figure 15 size/value 组合资产定价（外部数据/占位）"),
]

# ---------------------------------------------------------------------------
# 表任务（论文主表 I-V + 配套表）
# ---------------------------------------------------------------------------
_TABLE_TASKS: List[Task] = [
    Task("table_i",   "table", "tablecode.table_i",              "Table I   连续/跳跃收益汇总统计"),
    Task("table_ii",  "table", "tablecode.table_ii",             "Table II  平衡/非平衡面板因子空间 GC"),
    Task("table_iii", "table", "tablecode.table_iii",            "Table III 行业/FFC 因子 GC（外部数据/占位）"),
    Task("table_iv",  "table", "tablecode.table_iv",             "Table IV  时间变化分解汇总"),
    Task("table_v",   "table", "tablecode.table_v",              "Table V   日内/隔夜/日度夏普"),
    Task("table_fc",  "table", "tablecode.table_factor_counts",  "扰动特征值比诊断表（Fig 1-2 底座）"),
    Task("table_w",   "table", "tablecode.table_weights",        "连续/代理/月频因子权重表（Fig 3-5 底座）"),
    Task("table_fr",  "table", "tablecode.table_factor_returns", "因子收益摘要（Fig 12-13 底座）"),
    Task("table_cov", "table", "tablecode.table_coverage",       "复现覆盖度报告"),
]

ALL_TASKS: List[Task] = _FIGURE_TASKS + _TABLE_TASKS
_BY_KEY = {t.key: t for t in ALL_TASKS}


def all_tasks() -> List[Task]:
    return list(ALL_TASKS)


def figure_tasks() -> List[Task]:
    return list(_FIGURE_TASKS)


def table_tasks() -> List[Task]:
    return list(_TABLE_TASKS)


def get_task(key: str) -> Optional[Task]:
    return _BY_KEY.get(key)


def resolve_keys(selectors: List[str]) -> List[Task]:
    """把命令行选择器解析成任务列表。

    支持：
      - 具体短名：fig8 / table_i
      - 分组关键字：figures（所有图）/ tables（所有表）/ all（全部）
    顺序保持登记顺序，去重。
    """
    chosen: List[Task] = []
    seen = set()

    def _add(task: Task) -> None:
        if task.key not in seen:
            seen.add(task.key)
            chosen.append(task)

    for sel in selectors:
        s = sel.strip().lower()
        if s == "all":
            for t in ALL_TASKS:
                _add(t)
        elif s in ("figures", "figure", "figs", "fig"):
            for t in _FIGURE_TASKS:
                _add(t)
        elif s in ("tables", "table"):
            for t in _TABLE_TASKS:
                _add(t)
        else:
            task = get_task(s)
            if task is None:
                raise KeyError(f"未知任务选择器: {sel!r}（可用 all / figures / tables 或具体短名）")
            _add(task)
    return chosen
