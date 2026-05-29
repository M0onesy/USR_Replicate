# `allcode_Need.py` 拆分版说明

本目录是把原先的大型单文件流程拆成“核心引擎 + 图表脚本 + 表格脚本 + 总控入口”的版本，目标是让结果复用、调试和维护更清晰。

当前有两个关键约定：

1. `main.py` 现在是 **纯配置驱动入口**。
2. 单图/单表脚本仍然保留各自的独立 CLI，用于局部调试。

## 目录结构

```text
Code/
├─ main.py                  总控入口：读取 core/config.py 中的主运行配置后执行
├─ core/
│  ├─ engine.py             核心复现引擎与重型计算流程
│  ├─ config.py             运行参数 RunConfig + main.py 集中式 profile 配置
│  ├─ pipeline_cache.py     ReplicationResult 的内存/磁盘缓存复用
│  ├─ logging_utils.py      控制台日志与心跳
│  ├─ io_utils.py           输出路径与通用 I/O 工具
│  ├─ runner.py             单图/单表脚本统一执行器
│  └─ registry.py           全部图表任务注册表
├─ figcode/                 每张图一个脚本
└─ tablecode/               每张表一个脚本
```

## `main.py` 现在怎么运行

`main.py` 不再接受 `--only`、`--restart`、`--rebuild-result` 之类命令行参数。

现在的用法是：

1. 打开 [core/config.py](/d:/Courses/机器学习/Reposit/Code/core/config.py:1)
2. 修改 `ACTIVE_MAIN_PROFILE`
3. 如有需要，调整 `MAIN_RUN_PROFILES` 中对应 profile 的常量
4. 在 PyCharm 里直接点运行 `main.py`，或在终端执行：

```bash
python main.py
```

如果你在仓库根目录运行，则使用：

```bash
python Code/main.py
```

## 主运行配置在哪里改

`core/config.py` 里现在有三层关键内容：

1. `RunConfig`
   底层运行参数容器，给 pipeline、缓存、图表导出共用。

2. `MAIN_RUN_PROFILES`
   `main.py` 的预设运行方案字典。

3. `ACTIVE_MAIN_PROFILE`
   当前真正生效的主入口 profile 名称。你在 PyCharm 点运行时，程序就按它执行。

默认内置了这些 profile：

- `export_all`
  优先复用已有 `ReplicationResult`，导出全部图和表。
- `figures_only`
  只导出全部图。
- `tables_only`
  只导出全部表。
- `fig13_only`
  只跑 `fig13`，适合局部调试。
- `rebuild_all`
  显式重建上游 `ReplicationResult`，然后再导出全部结果。

## 推荐工作流

### 1. 平时重导结果

把 `ACTIVE_MAIN_PROFILE` 设为：

```python
ACTIVE_MAIN_PROFILE = "export_all"
```

这会优先复用已有 `ReplicationResult`，不会默认重新跑 20 小时级上游计算。

### 2. 只看某一张图

把 `ACTIVE_MAIN_PROFILE` 设为：

```python
ACTIVE_MAIN_PROFILE = "fig13_only"
```

然后直接运行 `main.py`。

### 3. 真的需要重建上游

把 `ACTIVE_MAIN_PROFILE` 切到：

```python
ACTIVE_MAIN_PROFILE = "rebuild_all"
```

或者在 `MAIN_RUN_PROFILES` 里把当前 profile 改成：

- `rebuild_result=True`
- `restart=True`

然后再运行 `main.py`。

## 结果复用逻辑

`main.py` 会先尝试从：

- `Result/.../checkpoints/replication_result_*.pkl`

复用已有 `ReplicationResult`。

如果存在精确签名缓存，就直接命中。
如果精确签名没命中，但当前是导出模式，程序还会回退尝试最近一次已完成的缓存结果。

只有在 profile 明确要求 `rebuild_result=True` 时，才会进入重型 pipeline。

## 心跳与进度

`main.py` 的心跳现在也由 `core/config.py` 控制：

- `enable_heartbeat`
- `heartbeat_sec`

适合在 PyCharm 中直接看控制台进度，不必再传 `--heartbeat-sec` 或 `--no-heartbeat`。

## 单图 / 单表脚本

这次集中配置只收口 `main.py`。

因此下面这些入口仍然保持原有 CLI：

```bash
python figcode/figure_13.py
python tablecode/table_i.py --years 2024
```

如果单脚本运行时没有可复用缓存，而你又允许它显式构建上游，可以继续使用：

```bash
python figcode/figure_13.py --allow-build
```

## 注意事项

- `main.py` 若收到任何命令行参数，会直接报错，并提示去修改 `core/config.py`。
- `restart=True` 只能和 `rebuild_result=True` 一起使用；配置写错会在启动前直接报错。
- 如果只是展示层修改，优先使用已有 `ReplicationResult` 重导，不要轻易切到 `rebuild_all`。
