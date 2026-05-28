# allcode_Need.py 拆分版 —— 论文图表生成总控台

本目录是把原始单文件 `allcode_Need.py`（4900+ 行）按"每张图、每张表一篇独立脚本 +
一个总控台"重构后的版本。重构遵循 **Option A：上游昂贵计算只跑一次，所有图表复用**。

> 核心承诺：`core/engine.py` 里的计算逻辑是从 `allcode_Need.py` **逐字节迁移** 的，
> 没有改动任何数学，因此数值结果与原脚本完全一致。本次重构只做"拆分、编排、日志、
> 缓存"这些外围工程化工作。

---

## 目录结构

```
Code/
├─ main.py                  总控台：调度全部 / 单个图表，持续心跳，运行汇总
├─ core/                    共享层
│  ├─ engine.py             ★ 全部重活（迁移自 allcode_Need.py，数学不变）
│  ├─ config.py             RunConfig：单次运行参数
│  ├─ pipeline_cache.py     一次计算多处复用的缓存（内存 + 磁盘 pickle）
│  ├─ logging_utils.py      每脚本日志 + 后台心跳线程
│  ├─ io_utils.py           输出目录 / 权威文件名 / 滚动结果切片 / 底层绘图复用
│  ├─ runner.py             图表脚本统一运行脚手架
│  └─ registry.py           所有图/表任务的总目录（新增任务在此登记）
├─ figcode/                 每张图一篇（figure_01.py ... figure_15.py）
└─ tablecode/               每张表一篇（table_i ... table_v + 配套表）
```

把本 `Code/` 直接覆盖到原仓库的 `Code/` 位置即可，路径（`Data/proc_Data`、
`Result/`）与原脚本完全兼容。

---

## 为什么这样拆（Option A 设计说明）

原脚本的真实结构是"**先算后导**"：
`run_cn_replication()` 做完所有重活（载入面板、跳跃分解、PCA、滚动 PCA、逐年论文表），
产出一个 `ReplicationResult` 对象；之后每张图 / 表只是从这个对象里**取字段**画图或落表，
本身非常便宜。

因此如果让每个图表脚本各自重跑一遍 pipeline，"全部运行"就会把几小时的 PCA 重复二十多次。
Option A 的做法：

1. `main.py` 先通过 `pipeline_cache.get_result()` 把 `ReplicationResult` 准备好
   （命中缓存秒回，否则构建一次并落盘）。
2. 再把同一个 result 依次喂给每个图表脚本，**零重算**。
3. 单独运行某一篇脚本时，它会自动命中磁盘缓存；没有缓存才触发一次构建。

这样既满足"单独跑某一篇方便调试"，又不牺牲"全部运行"的速度。

---

## 快速开始

### 1. 列出所有可用任务

```bash
python main.py --list
```

### 2. 全部运行（15 图 + 9 表）

```bash
python main.py --only all
```

首次会构建一次 `ReplicationResult`（最耗时的一步），随后所有图表复用它。

### 3. 只跑图 / 只跑表 / 单独一篇

```bash
python main.py --only figures        # 所有图
python main.py --only tables         # 所有表
python main.py --only fig8           # 只跑 Figure 8
python main.py --only fig8 table_i   # Figure 8 + Table I
```

### 4. 小样本快速跑通（强烈建议先用它验证环境）

```bash
python main.py --only all --years 2016 --max-stocks 10
```

### 5. 单独运行某个脚本（不经过 main）

每个图表脚本都能独立执行，会自动走缓存：

```bash
python figcode/figure_08.py
python tablecode/table_i.py --years 2016 --max-stocks 10
```

---

## 心跳与调试

- **持续心跳**：`main.py` 后台每隔 `--heartbeat-sec` 秒（默认 10s）报告
  "已运行多久 / 当前任务 / 已完成几项"，长任务卡住时一眼看出停在哪。
  关闭用 `--no-heartbeat`。
- **每脚本日志**：每篇图表脚本被运行到时都会打印
  `[开始] / [数据处理] / [图表输出] / [完成(含耗时)]`。
  其中 **[数据处理] 与 [图表输出] 明确分段**，方便判断"是数据处理重还是绘图重、卡在哪一步"。
- **容错**：默认 `--keep-going`，单个任务报错不影响其余任务，最后给出成功/失败汇总；
  想一遇错就停加 `--fail-fast`。

---

## 常用参数

| 参数 | 含义 |
| --- | --- |
| `--only` | 任务选择：`all` / `figures` / `tables` / 具体短名（见 `--list`） |
| `--list` | 列出所有任务后退出 |
| `--proc-root` | 预处理数据目录（默认 `Data/proc_Data/pelger_cn_adjusted`） |
| `--output-root` | 结果输出目录（默认 `Result/pelger_cn_adjusted`） |
| `--years` | 指定年份，如 `--years 2016 2017` |
| `--max-stocks` | 只用前 N 只股票（smoke test） |
| `--jump-a` / `--k-max` / `--gamma` / `--g-fn` | 论文方法参数（影响数值，会进入缓存键） |
| `--workers` / `--paper-workers` / `--rolling-workers` | 并行 worker 数 |
| `--memory-budget-gb` | 自适应内存预算 |
| `--heartbeat-sec` | 心跳间隔秒数 |
| `--fail-fast` / `--no-heartbeat` | 容错与心跳开关 |
| `--restart` | 丢弃兼容 checkpoint 重跑上游 pipeline |

> 改变 `--years / --max-stocks / --jump-a / --k-max / --gamma / --g-fn / --proc-root`
> 会改变缓存键，从而自动失效旧缓存、触发重算；纯性能参数（worker / 心跳）不影响缓存。

---

## 图表清单

**图（figcode/，对应论文 Figure 1-15）**

| 短名 | 图 | 说明 |
| --- | --- | --- |
| fig1 / fig2 | Figure 1 / 2 | HF 因子个数（非平衡 / 平衡面板的扰动特征值比折线） |
| fig3 / fig4 / fig5 | Figure 3 / 4 / 5 | 代理 / 连续 PCA / 月频 PCA 因子组合权重热图 |
| fig6 / fig7 | Figure 6 / 7 | 载荷时间变化（全部 GC / 前 4 个主因子 GC） |
| fig8 / fig9 | Figure 8 / 9 | 随时间变化的组合权重 / 解释方差 |
| fig10 / fig11 | Figure 10 / 11 | 因子结构时间变化分解 / 连续因子结构分解 |
| fig12 / fig13 | Figure 12 / 13 | 预期日内与隔夜收益 / 因子累计收益 |
| fig14 / fig15 | Figure 14 / 15 | 行业 / size-value 组合资产定价（**需外部数据，当前占位图**） |

**表（tablecode/，对应论文 Table I-V + 配套）**

| 短名 | 表 | 说明 |
| --- | --- | --- |
| table_i | Table I | 连续 / 跳跃收益汇总统计 |
| table_ii | Table II | 平衡 / 非平衡面板因子空间广义相关性 |
| table_iii | Table III | 行业 / FFC 因子 GC（**需外部数据，当前占位说明**） |
| table_iv | Table IV | 时间变化分解汇总 |
| table_v | Table V | 日内 / 隔夜 / 日度夏普 |
| table_fc | — | 扰动特征值比诊断表（Figure 1-2 的数值底座） |
| table_w | — | 连续 / 代理 / 月频因子权重表（Figure 3-5 的底座） |
| table_fr | — | 因子收益摘要（Figure 12-13 的底座） |
| table_cov | — | 复现覆盖度报告 |

> 外部数据缺口（Table III、Figure 14/15）按原脚本策略输出**明确占位**，
> 保持论文编号完整、不静默缺失。要补全请提供相应测试资产 / 因子 CSV，
> 并接入 `engine.load_test_asset_csv` / `engine.load_external_factor_csv`。

---

## 与原脚本的等价性 / 一键全量

- 计算逻辑（`core/engine.py`）与 `allcode_Need.py` 数学等价。
- 如果你想用原脚本式的"一次性全量复现"（含 checkpoint 续跑等），`engine.py` 内仍保留
  `run_cn_replication / export_replication_outputs / export_all_paper_figures`，可直接调用。
- 本拆分版的 `main.py` 是更适合**维护调试**的入口：可单跑、可分组、有心跳、有分段日志。

## 依赖

`numpy`、`pandas`、`scipy`、`matplotlib`（与原项目一致）。
