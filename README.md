# USR 论文复现代码库

本项目用于复现 Pelger (2020)《Understanding Systematic Risk: A High-Frequency Approach》在中国 A 股 5 分钟频率数据上的核心分析流程。仓库的重点不是封装一个通用量化库，而是把论文主线实验整理成可以直接扫描本地数据、构造面板、提取高频因子、做稳健性检验并导出结果的一套脚本。

当前仓库最重要的入口是 [`Code/allcode_Need.py`](Code/allcode_Need.py)。它负责从本地 `data.bz2` 数据出发，完成样本扫描、收益构造、跳跃分解、PCA 因子提取、Sharpe 计算、滚动稳定性分析、99% 覆盖样本稳健性比较和结果导出。

> 当前仓库数据快照示例  
> 以下数字来自当前仓库里的 `Data/EXTRA_STOCK_A` 本地数据扫描结果，只是示例，不是代码写死的常量：
> - 股票数量约 2891 只
> - 全局样本区间为 `2013-01-04` 到 `2016-12-30`
> - 严格平衡样本 `strict_balanced` 为 180 只
> - 99% 覆盖样本 `near_balanced_99` 为 465 只
>
> 仓库根目录下的 `replication_output_cache_check/` 还保留了一份样例输出，方便对照运行后会生成哪些文件。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `Code/allcode_Need.py` | 项目主入口，一站式复现脚本，负责主分析流程和结果导出 |
| `Code/getApidb.py` | 从 AmazingData 接口批量拉取 A 股 5 分钟 K 线，支持 worker、失败重试、跳过已下载文件 |
| `Code/convert_bz2_to_csv.py` | 将 pandas pickle 格式的 `.bz2` 文件转换为 CSV，便于人工检查数据 |
| `requirements.txt` | 公开可安装依赖清单，适用于 `allcode_Need.py` 和 `convert_bz2_to_csv.py` |
| `Data/EXTRA_STOCK_A/` | 当前仓库内的本地数据目录，按股票代码分文件夹存放 `data.bz2` |
| `.hf_cache/` | 本地缓存目录，保存扫描结果、逐股票收益缓存、面板缓存等中间产物 |
| `replication_output_cache_check/` | 一份现成的样例输出目录，可用于对照结果文件结构 |

## 环境与依赖

推荐环境：

- Python 3.10
- Windows / PowerShell 可以直接运行当前仓库内的命令示例

`allcode_Need.py` 依赖：

- `numpy`
- `pandas`
- `scipy`
- `matplotlib` 可选，仅在需要导出 PNG 图时使用

基础安装方式：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前覆盖：

- `Code/allcode_Need.py` 的公开运行时依赖
- `Code/convert_bz2_to_csv.py` 所需依赖

### `requirements.txt` 说明

- 这是“公开可安装依赖清单”，不是当前机器的完整环境导出文件。
- 本项目不追求环境逐字节复现，因此这里不锁死精确版本，而是使用“宽松下限”策略。
- `matplotlib` 在代码里属于可选绘图依赖，但本次仍被纳入 `requirements.txt`，目的是让默认安装即可支持完整输出和 PNG 图形。
- `requirements.txt` 不包含 `Code/getApidb.py` 的专有数据接口依赖。

`getApidb.py` 额外依赖：

- `AmazingData`
- `api_AmazingData_professional`

这两个依赖是外部专有数据接口，不是本仓库自带内容，也不是标准 `pip` 依赖。因此：

- 你可以直接使用 `allcode_Need.py` 处理本地已有的 `data.bz2`
- 如果要重新下载原始数据，再考虑配置 `getApidb.py` 所需环境
- 即使已经执行 `pip install -r requirements.txt`，也仍然不能直接运行 `getApidb.py`

## 数据组织与路径约定

### 目录结构

`allcode_Need.py` 期望的 `data_root` 目录结构如下：

```text
<data_root>/
├─ 000001.SZ/
│  └─ data.bz2
├─ 000002.SZ/
│  └─ data.bz2
└─ ...
```

在当前仓库里，推荐直接使用：

```text
Data/EXTRA_STOCK_A
```

也就是说，单只股票文件的典型路径是：

```text
Data/EXTRA_STOCK_A/000001.SZ/data.bz2
```

### 文件格式

这里的 `data.bz2` 不是普通文本压缩包，而是用 pandas 保存的压缩 pickle 文件。`allcode_Need.py` 和 `convert_bz2_to_csv.py` 都是通过 `pd.read_pickle(..., compression="bz2")` 读取它。

每个 `data.bz2` 至少要包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `code` | 股票代码 |
| `kline_time` | 5 分钟 K 线时间戳 |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额 |

脚本在扫描阶段会对数据做基础清洗，包括：

- 解析 `kline_time`
- 将 OHLCV 等字段转成数值
- 按时间排序
- 对重复时间戳去重
- 检查单日是否恰好有 48 根 5 分钟 bar
- 检查时间网格是否匹配 A 股日内交易时段
- 检查 `open/high/low/close` 是否都大于 0

### 路径注意事项

这一点非常重要：

- `Code/allcode_Need.py` 代码里的默认 `data_root` 是 `Code/EXTRA_STOCK_A`
- 当前仓库实际的数据目录是 `Data/EXTRA_STOCK_A`

所以 README 里的所有运行示例都会显式写：

```bash
--data-root Data/EXTRA_STOCK_A
```

如果你使用 `getApidb.py` 重新下载数据，建议直接把输出放到 `Data/` 下，例如：

```bash
python Code/getApidb.py --output-root Data
```

这样最终目录会更接近 `allcode_Need.py` 所需结构；如果你的数据不在这个位置，也可以在运行 `allcode_Need.py` 时显式指定 `--data-root`。

## `allcode_Need.py` 详解

### 脚本定位

[`Code/allcode_Need.py`](Code/allcode_Need.py) 是本仓库的核心脚本。它既可以作为命令行程序直接运行，也可以被当作模块导入后调用高层函数。

它的工作内容包括：

- 扫描全市场 `.bz2` 数据并统计样本覆盖情况
- 构造高频收益面板 `HFPanel`
- 用 TOD 阈值法拆分连续收益与跳跃收益
- 用 PCA 提取高频因子、连续因子、跳跃因子
- 估计因子数量
- 计算日内、隔夜、日度 Sharpe
- 计算滚动稳定性指标
- 在 `near_balanced_99` 样本上做稳健性比较
- 导出 CSV、JSON 和可选 PNG 图

### 主流程

脚本主流程可以概括为：

```text
scan_cn_bz2_universe
  -> build_cn_hf_panel
  -> PelgerPipeline.run_full
  -> rolling_* / run_near_balanced_robustness
  -> export_replication_outputs
```

对应含义如下：

1. `scan_cn_bz2_universe`  
   扫描所有 `data.bz2`，识别有效交易日、无效交易日、样本覆盖率、疑似除权除息风险，并生成全市场样本摘要。

2. `build_cn_hf_panel`  
   根据样本筛选规则和年份范围，构造高频收益面板 `HFPanel`，包含：
   - 日内收益矩阵 `R_intra`
   - 隔夜收益矩阵 `R_night`
   - 交易日索引 `day_ids`
   - 股票列表 `tickers`
   - 日期列表 `dates`

3. `PelgerPipeline.run_full`  
   这是论文主流程的封装，内部依次执行：
   - `detect_jumps`：跳跃分解
   - `step2_determine_K`：确定因子个数
   - `step3_extract_factors`：提取高频 / 连续 / 跳跃因子
   - `step4_asset_pricing`：计算组合因子与 Sharpe

4. `rolling_gc_vs_global` / `rolling_explained_variation` / `run_near_balanced_robustness`  
   在主流程结果之上，进一步计算滚动稳定性指标与稳健性比较。

5. `export_replication_outputs`  
   将结果统一写到输出目录，形成 CSV、JSON 和可选 PNG 图。

### 样本口径与收益口径

#### `strict_balanced`

这是论文主结果默认口径。

- 要求在目标年份范围内完整覆盖目标交易日
- 对于按年筛选时，要求 `valid_days == total_days` 且 `observed_days == total_days`
- 样本更“干净”，但股票数通常更少

#### `near_balanced_99`

这是稳健性分析口径。

- 允许样本覆盖率约为 99% 以上
- 仍要求观测天数与有效天数一致，即不接受“观测了但当天网格不完整”的情况
- 在稳健性分析中，脚本会对这类非完全平衡样本使用 pairwise-covariance PCA，并做 PSD 修正

#### `open_close`

这是默认的日内收益构造方式。

- 每根 5 分钟 bar 的日内收益定义为 `log(close / open)`
- 单日总收益等于 48 根 bar 的日内收益之和再加隔夜收益

#### `close_close`

这是备选收益构造方式。

- 第 1 根 bar 用 `log(close_1 / open_1)`
- 从第 2 根开始，使用相邻 bar 的收盘价构造 `log(close_t / close_{t-1})`
- 适合在你希望把日内收益更明确地写成“收盘到收盘”链式收益时使用

无论使用哪种口径，隔夜收益都来自：

```text
log(当日首个 open / 前一交易日最后一个 close)
```

### 关键对象 / 函数速览

| 名称 | 作用 |
| --- | --- |
| `scan_cn_bz2_universe` | 扫描全市场数据，形成样本清单与覆盖统计 |
| `build_cn_hf_panel` | 按样本口径和年份构造 `HFPanel` |
| `HFPanel` | 高频面板容器，保存 `R_intra`、`R_night`、`day_ids`、`tickers`、`dates` 等核心数据 |
| `detect_jumps` | 用 TOD 阈值法把日内收益拆成连续部分和跳跃部分 |
| `PCAResult` | 存放 PCA 结果，包括载荷、因子、特征值等 |
| `PelgerPipeline` | 论文主流程封装，串起跳跃分解、因子数估计、PCA、Sharpe 计算 |
| `run_near_balanced_robustness` | 对 99% 覆盖样本做稳健性比较 |
| `ReplicationResult` | 保存完整运行后的结果对象和导出文件信息 |
| `export_replication_outputs` | 将主流程结果导出为表格、摘要和图片 |
| `run_cn_replication` | 一站式高层入口，最适合作为脚本或模块调用的主函数 |

如果你是从代码导入使用，建议优先理解下面这几个对外入口：

- `scan_cn_bz2_universe`
- `build_cn_hf_panel`
- `PelgerPipeline`
- `run_near_balanced_robustness`
- `run_cn_replication`

### CLI 参数说明

以下参数来自 `Code/allcode_Need.py` 当前实现。

#### 输入输出参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-root` | `Code/EXTRA_STOCK_A` | 输入数据根目录。当前仓库请显式传 `Data/EXTRA_STOCK_A` |
| `--output-root` | `Code/replication_output` | 结果输出目录 |
| `--cache-root` | `Code/.hf_cache/pelger_cn` | 缓存目录 |

#### 缓存控制参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--no-cache` | 关闭后为 `False` | 禁用缓存 |
| `--refresh-cache` | `False` | 刷新全部缓存 |
| `--refresh-symbol-cache` | `False` | 刷新逐股票收益缓存 |

#### 执行模式参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--scan-only` | `False` | 只扫描样本，不执行论文主流程 |
| `--no-robustness` | `False` | 跳过 `near_balanced_99` 样本稳健性分析 |
| `--no-plots` | `False` | 不输出 PNG 图形 |

#### 样本控制参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sample-mode` | `strict_balanced` | 主流程样本口径，可选 `strict_balanced` / `near_balanced_99` |
| `--return-mode` | `open_close` | 日内收益构造方式，可选 `open_close` / `close_close` |
| `--years` | 全部年份 | 只运行指定年份，例如 `--years 2015 2016` |
| `--max-stocks` | 不限制 | 仅取前若干只股票，适合 smoke test |

#### 模型参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--jump-a` | `3.0` | 跳跃阈值倍数 |
| `--k-max` | `10` | 扰动特征值比率搜索上限 |
| `--gamma` | `0.08` | 扰动特征值比率阈值 |

### 推荐运行命令

以下命令都假设你在仓库根目录执行。

#### 1. 只扫描样本

适合先确认数据是否能被正确识别、当前样本覆盖情况如何。

```bash
python Code/allcode_Need.py --data-root Data/EXTRA_STOCK_A --output-root replication_output_scan --cache-root .hf_cache/pelger_cn --scan-only
```

#### 2. 小样本 smoke test

适合先跑通流程，减少等待时间。

```bash
python Code/allcode_Need.py --data-root Data/EXTRA_STOCK_A --output-root replication_output_smoke --cache-root .hf_cache/pelger_cn --years 2016 --max-stocks 10 --no-robustness --no-plots
```

#### 3. 按当前仓库数据做完整运行

```bash
python Code/allcode_Need.py --data-root Data/EXTRA_STOCK_A --output-root replication_output --cache-root .hf_cache/pelger_cn
```

说明：

- 这里显式指定了 `--output-root` 和 `--cache-root`，是为了把输出和缓存都集中放在仓库根目录，避免默认写到 `Code/` 下
- 如果你想只跑部分年份，可以额外加 `--years`
- 如果你第一次跑完整流程较慢，后续重复运行通常会因为缓存而明显加速

### 输出文件说明

完整运行后，输出目录中通常会看到以下文件：

| 文件名 | 说明 |
| --- | --- |
| `universe_scan.csv` | 全市场样本扫描明细，包含每只股票的有效天数、无效天数、覆盖率、疑似隔夜异常等信息 |
| `universe_summary.json` | 全市场样本摘要，例如全局日期范围、每年交易日数量、平衡样本数量等 |
| `main_summary.json` | 主流程摘要，包含样本报告、跳跃统计、因子个数和 Sharpe |
| `jump_stats.csv` | 跳跃分解核心统计量 |
| `factor_counts.csv` | `K_hf_hat`、`K_cont_hat`、`K_jump_hat` |
| `sharpes.csv` | 日内、隔夜、日度 Sharpe |
| `main_sample_symbols.csv` | 主流程最终使用的股票列表 |
| `rolling_gc.csv` | 滚动 generalized correlation 指标 |
| `rolling_explained_variation.csv` | 滚动解释方差结果 |
| `robustness_yearly_gc.csv` | `near_balanced_99` 样本的稳健性比较结果 |
| `corp_action_risk.csv` | 样本内疑似除权除息风险提示 |

如果没有加 `--no-plots`，并且本地可用 `matplotlib`，通常还会生成：

- `rolling_gc.png`
- `rolling_explained_variation.png`

你可以直接参考仓库里的 `replication_output_cache_check/`，查看这些文件在实际运行后长什么样。

### 常见问题 / 注意事项

1. 默认数据路径和当前仓库不一致  
   `allcode_Need.py` 默认读取 `Code/EXTRA_STOCK_A`，但当前仓库数据在 `Data/EXTRA_STOCK_A`。务必显式传 `--data-root Data/EXTRA_STOCK_A`。

2. 第一次跑会比较慢，缓存是正常设计的一部分  
   脚本会缓存：
   - 样本扫描结果
   - 逐股票收益
   - 面板构造结果  
   因此重复运行通常会快很多。

3. 当前结果没有做复权处理  
   README 开头旧 TODO 里提到的复权问题依然存在。也就是说，隔夜大跳变里可能混入除权除息等公司行为影响。

4. `corp_action_risk.csv` 是提示，不是自动修正  
   脚本会统计异常大的隔夜收益，帮助识别潜在公司行为风险，但不会自动下载复权因子，也不会自动调整价格。

5. `replication_output_cache_check/` 是样例结果目录，不是默认输出目录  
   正常运行时，输出位置取决于你传入的 `--output-root`。如果不传，则默认写到 `Code/replication_output`。

6. 历史缓存或样例摘要里的绝对路径可能来自旧环境  
   例如某些 JSON 里可能保留老机器上的绝对路径。这类路径主要用于记录历史运行上下文，不代表你当前机器上的真实路径设置。

7. `.bz2` 必须是 pandas pickle，不是普通压缩文本  
   如果你拿一个 CSV 再手工压成 `.bz2`，脚本是读不出来的。

## 其他脚本

### `getApidb.py`

[`Code/getApidb.py`](Code/getApidb.py) 用来批量导出 AmazingData 的 A 股 5 分钟 K 线数据。当前 `TYPE_CONFIG` 里启用的是 `EXTRA_STOCK_A`。

这个脚本的特点：

- 支持 parent / worker 分离运行
- 支持分批导出
- 支持失败批次重试
- 支持大批次自动拆分
- 支持跳过已存在的非空 `data.bz2`
- 会记录事件日志和失败股票列表

如果你想把数据直接放到当前仓库推荐的位置，可以使用：

```bash
python Code/getApidb.py --output-root Data --batch-size 50 --skip-existing
```

运行后，目标结构会接近：

```text
Data/
└─ EXTRA_STOCK_A/
   └─ <symbol>/
      └─ data.bz2
```

注意：

- 该脚本依赖外部专有数据接口
- 没有配置 AmazingData 环境时，它不能开箱即用
- 即使已经执行 `pip install -r requirements.txt`，也还需要额外配置专有 SDK / API 环境

### `convert_bz2_to_csv.py`

[`Code/convert_bz2_to_csv.py`](Code/convert_bz2_to_csv.py) 用来把 pandas pickle 格式的 `.bz2` 文件导出为 CSV，适合抽样检查单只股票的数据内容。

使用方式：

```bash
python Code/convert_bz2_to_csv.py <path>
```

其中 `<path>` 可以是：

- 单个 `.bz2` 文件
- 一个目录

但要注意目录模式的行为：

- 它只会处理该目录下当前层级的 `.bz2` 文件
- 不会递归遍历子目录

例如，下面这种写法适合转换单只股票目录中的 `data.bz2`：

```bash
python Code/convert_bz2_to_csv.py Data/EXTRA_STOCK_A/000001.SZ
```

## 已知限制

- 当前项目主要围绕 `allcode_Need.py` 的论文复现流程组织，不是通用研究框架或量化平台。
- 当前本地样本快照主要覆盖 `2013-01-04` 到 `2016-12-30`；如果本地数据变化，README 里的样本数量示例也会变化。
- 价格当前未做复权处理，隔夜异常中可能混入除权除息等公司行为影响。
- 脚本中为外部因子和测试资产预留了接口，但不会自动下载任何外部因子、Fama-French 因子或测试资产数据。
- `getApidb.py` 依赖外部专有数据源，不是一个任何机器都能直接运行的数据下载脚本。
