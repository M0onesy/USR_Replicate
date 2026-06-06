# `Code/` 目录运行说明

当前拆分版已经不再依赖旧的 `allcode_Need.py` 单文件入口。`Code/` 下的真实运行方式如下。

## 主入口

- `main.py`
  - 主复现入口
  - 纯 `core/config.py` 配置驱动
  - 不再接受命令行参数

从仓库根目录运行：

```bash
python Code/main.py
```

从 `Code/` 目录运行：

```bash
python main.py
```

## 配置真源

`core/config.py` 是主入口唯一真源。

重点字段：

- `ACTIVE_MAIN_PROFILE`
  - 当前主运行 profile
- `MAIN_RUN_PROFILES`
  - 所有可切换预设

当前常用 profile：

- `reuse_export_smoke`
  - 优先复用已有 `ReplicationResult`，快速重导正式论文输出
- `rebuild_proc_and_result`
  - 进入显式重建模式，但不强制严格最终导出
- `final_paper_export`
  - 正式全量导出模式，使用 `paper_lenient`
- `diagnostics_only`
  - 只导出附加诊断表

## 任务分组

`core/registry.py` 当前把任务分成两层：

- `all`
  - 正式论文输出：Figure 1–15 和 Table I–V
- `diagnostics`
  - 附加诊断表，例如因子数诊断、权重表、因子收益摘要、覆盖报告

这意味着默认主运行不会再把诊断底座表写进 `Result/tables/`。

## 目录职责

- `core/`
  - 引擎、缓存、配置、任务注册、I/O
- `figcode/`
  - 各张图的单独导出脚本
- `tablecode/`
  - 各张表的单独导出脚本

## 单图 / 单表脚本

单图、单表脚本仍保留独立 CLI，用于局部调试。例如：

```bash
python Code/figcode/figure_13.py
python Code/tablecode/table_i.py
```

默认会优先复用已有 `ReplicationResult`。如果没有可复用缓存，且你明确允许它显式重建上游，可使用：

```bash
python Code/figcode/figure_13.py --allow-build
```

## 输出分层

- `Result/pelger_cn_adjusted/`
  - 正式论文输出
- `Data/proc_Data/pelger_cn_adjusted/runtime/`
  - checkpoint、进度日志、资源计划、诊断
- `Data/proc_Data/pelger_cn_adjusted/paper_tail/`
  - 后半段缓存与校验

如果运行行为与旧截图或旧文档冲突，应以当前代码实现为准。
