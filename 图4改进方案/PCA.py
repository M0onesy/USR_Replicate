#导入必要的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 读取股票日度收益率数据
stock_returns = pd.read_csv('full_daily.csv', index_col=0, parse_dates=True)
print(f"个股数据维度: {stock_returns.shape}")   # (3156, 115)


# 2. 标准化收益率矩阵（每列减去均值，除以标准差）
scaler = StandardScaler()
stock_returns_scaled = scaler.fit_transform(stock_returns)   # numpy 数组
# 转回 DataFrame 保持索引
stock_returns_scaled = pd.DataFrame(stock_returns_scaled,
                                    index=stock_returns.index,
                                    columns=stock_returns.columns)

# 3. 计算样本相关矩阵
corr_matrix = np.cov(stock_returns_scaled.T)   # shape (115, 115)

# 4. 特征分解
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)   # 升序
eigenvalues = eigenvalues[::-1]              # 降序
eigenvectors = eigenvectors[:, ::-1]         # 对应特征向量
print(eigenvalues)

# 5. 扰动特征值比确定主成分个数 K
gamma = 0.25
g = np.median(eigenvalues) * 0.15
perturbed = eigenvalues + g
ER = perturbed[:-1] / perturbed[1:]          # 比值序列，长度 N-1
# 找出所有满足 ER_k > 1+gamma 的索引，取最大索引+1
idx = np.where(ER > 1 + gamma)[0]
if len(idx) > 0:
    K = idx[-1] + 1
else:
    K = 1
print(f"扰动特征值比法确定因子个数 K = {K}")

# 绘制 ER 图
plt.figure(figsize=(10,6))
plt.plot(range(1, len(ER)+1), ER, marker='o')
plt.axhline(y=1+gamma, color='r', linestyle='--', label=f'Threshold = {1+gamma}')
plt.xlabel('因子序号 k')
plt.ylabel('扰动特征值比 ER_k')
plt.title('扰动特征值比确定因子个数')
plt.legend()
plt.grid(True)
plt.savefig('ER_plot.png', dpi=300, bbox_inches='tight')

# 绘制碎石图
plt.figure(figsize=(10,6))
comp = range(1, len(eigenvalues)+1)
plt.plot(comp, eigenvalues, marker='o', linestyle='-', color='green')
plt.xlabel('主成分序号')
plt.ylabel('特征值')
plt.title('碎石图')
plt.grid(True)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='特征值=1')
plt.legend()
plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')

# 6. 提取前 K 个主成分的载荷和因子收益率
loadings = eigenvectors[:, :K]     # shape (115, K)
# 因子收益率 = 标准化数据 × 载荷
F = stock_returns_scaled.values @ loadings     # shape (3156, K)
factor_returns = pd.DataFrame(F, index=stock_returns.index,
                              columns=[f'PC{i+1}' for i in range(K)])
print("因子收益率矩阵维度:", factor_returns.shape)
print(factor_returns)
print("载荷矩阵维度:", loadings.shape)
loadings_data=pd.DataFrame(loadings,index=stock_returns.columns,columns=[f'PC{i+1}' for i in range(K)])
print(loadings_data)
# 保存结果
factor_returns.to_csv('factor_returns.csv', index=True, header=True)
np.save('loadings.npy', loadings)

F_N = stock_returns.values @ loadings    # 用未标准化的数据计算的因子收益率
factor_returns_N = pd.DataFrame(F_N, index=stock_returns.index,
                              columns=[f'PC{i+1}' for i in range(K)])
print("未标准化因子收益率矩阵维度:", factor_returns_N.shape)
print(factor_returns_N)
factor_returns_N.to_csv('factor_returns_N.csv', index=True, header=True)

# 7. 绘制载荷矩阵热力图 (N×K)
K = factor_returns.shape[1]
top_n = 15
loadings_top = loadings[:top_n, :]   # shape (15, K)
index_names = [loadings_data.index[:top_n]]
column_names = [f'PC{i+1}' for i in range(K)]
df_loadings = pd.DataFrame(loadings_top, index=index_names, columns=column_names)
plt.figure(figsize=(10, 6))
sns.heatmap(df_loadings, annot=True, fmt='.2f',cmap='RdBu_r', center=0,linewidths=0.5, linecolor='gray',
            xticklabels=[f'PC{i+1}' for i in range(K)],
            yticklabels=False, cbar_kws={'label': '载荷值'})
plt.title(f'因子载荷矩阵 (N={loadings.shape[0]}只股票, K={K}个主成分)')
plt.xlabel('主成分')
plt.ylabel(f'前{top_n}只股票')
plt.tight_layout()
plt.savefig('loadings.png', dpi=300, bbox_inches='tight')



# 股票行业信息（含宏观行业）
stock_info = pd.read_csv('stock_marco_industry.csv',encoding='gbk')
# 只保留需要的列，假设有 ts_code 和 marco_industry
stock_info = stock_info[['ts_code', 'marco_industry']]

# 2. 合并载荷与行业
merged = loadings_data.reset_index().rename(columns={'index': 'ts_code'})
merged = merged.merge(stock_info, on='ts_code', how='left')
# 删除缺失行业的股票（如果有）
merged = merged.dropna(subset=['marco_industry'])

# 3. 按宏观行业排序
merged = merged.sort_values('marco_industry').reset_index(drop=True)
print(merged)

# 记录每个行业的起始和结束索引（用于画分隔线）
industry_boundaries = []
for industry, group in merged.groupby('marco_industry'):
    start = group.index[0]
    end = group.index[-1]
    industry_boundaries.append((start, end, industry))

# 4. 获取行业唯一列表并分配颜色
industries = merged['marco_industry'].unique()
print(f"分成了{len(industries)}个行业：",industries)
colors = plt.cm.tab10(np.linspace(0, 1, len(industries)))
industry_color = dict(zip(industries, colors))

# 5. 绘制5个子图（纵向排列）
fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)  # 共享x轴
pcs = [f'PC{i+1}' for i in range(5)]
for idx, pc in enumerate(pcs):
    ax = axes[idx]
    y = merged[pc].values
    x = np.arange(len(y))
    bar_width = 0.7
    # 按行业绘制柱状图
    for industry, group in merged.groupby('marco_industry'):
        idx_industry = group.index
        ax.bar(idx_industry, group[pc],width=bar_width,color=industry_color[industry], alpha=0.8,edgecolor='black',
               linewidth=0.5,
               label=industry)
    # 添加行业分隔线（垂直虚线）
    for start, end, industry in industry_boundaries:
        ax.axvline(x=start-0.5, color='r', linestyle='--', linewidth=0.8, alpha=0.7)
    # 最后一个边界后加一根
    last_end = industry_boundaries[-1][1]
    ax.axvline(x=last_end+0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_ylabel('Factor Loadings')
    ax.set_title(f'{pc}')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    # 获取当前PC数据的最大绝对值，并留一点边距
    max_val = np.max(np.abs(y))
    ax.set_ylim(-max_val * 1.1, max_val * 1.1)
    # 只在第一个子图显示图例，避免重复
    if idx == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[-1].set_xlabel('Stocks sorted by macro industry')
plt.tight_layout()
plt.savefig('loadings_by_industry.png', dpi=300, bbox_inches='tight')


