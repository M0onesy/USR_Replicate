# Understanding Systematic Risk 高频复现项目

本仓库用于复现 Pelger (2020) 的高频系统性风险框架，并用中国 A 股 5 分钟 K 线做数据适配。

## 工作流

1. 先运行 `Code/preprocess_cn_data.py`，读取原始 `Data/kline_Data/EXTRA_STOCK_A` 和 `Data/fact_Data/backward_factor.csv`，生成固定口径的预处理面板到 `Data/proc_Data/pelger_cn_adjusted`。
2. 再运行 `Code/allcode_Need.py`，只读取 `Data/proc_Data/pelger_cn_adjusted`，输出论文复现所需的表、图和诊断文件到 `Result/pelger_cn_adjusted`。

## 数据目录

- `Data/kline_Data/EXTRA_STOCK_A`：原始 A 股 5 分钟 K 线，当前共有 `5799` 个股票目录。
- `Data/fact_Data/backward_factor.csv`：后复权因子。
- `Data/proc_Data/pelger_cn_adjusted`：预处理产物。
- `Result/pelger_cn_adjusted`：复现结果。

## 面板定义

- `R_daily`：前一交易日收盘到当日收盘的日度总收益。
- `R_intra`：当日开盘到当日收盘的日内收益，形状 `(D, N)`。
- `R_night`：前一交易日收盘到当日开盘的隔夜收益，形状 `(D, N)`。
- `R_5min_full`：5 分钟收盘接续收益，形状 `(D*48, N)`。
- `strict_balanced/full`：全区间零缺失股票。
- `strict_balanced/year_YYYY`：当年零缺失股票。

## 运行命令

```bash
python Code/preprocess_cn_data.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\pelger_cn_adjusted
```

```bash
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted
```

`--return-mode` 仅保留兼容，不再影响收益定义。正式运行不要加 `--no-plots`，否则不会输出图片。

## 导出 CSV

```bash
python Code/export_panel_csv.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\panel_csv
```

## 输出结构

- `tables`：论文表格和诊断表。
- `figures`：论文图，编号 `Figure_01` 到 `Figure_15`。
- `diagnostics`：样本摘要、覆盖率、复权风险和导出状态。

## 说明

- 当前只保留 `strict_balanced`，不再生成或使用旧的非平衡稳健性面板。
- 如果重新替换原始数据或复权因子，请加 `--refresh` 重新跑预处理。
