import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# 加载数据
df = pd.read_csv('okGas.csv')

# 基本信息查看
print(df.info())
print(df.head())

# 绘制核密度曲线，每10个特征绘制在一个图上
num_features = df.shape[1] - 1  # 不包括标签列
features_per_chart = 10
num_charts = num_features // features_per_chart + (0 if num_features % features_per_chart == 0 else 1)

for i in range(num_charts):
    plt.figure(figsize=(15, 3))
    start_idx = i * features_per_chart
    end_idx = min(start_idx + features_per_chart, num_features)
    for column in df.columns[start_idx:end_idx]:
        sns.kdeplot(df[column], fill=True)  # 使用 fill=True 替代 shade=True
    plt.title(f'Feature Density Plots (Features {start_idx} to {end_idx - 1})')
    plt.legend(labels=df.columns[start_idx:end_idx])
    plt.show()
# 相关系数热力图
corr_matrix = df.corr()

# 找出相关性最高的20个特征
abs_corr_matrix = corr_matrix.abs().unstack()
sorted_pairs = abs_corr_matrix.sort_values(kind="quicksort", ascending=False)
top_pairs = sorted_pairs[sorted_pairs != 1][:40]  # 排除自身比较，取前40个结果(每对特征出现2次)

# 提取这些特征的名称
top_features = set([pair[0] for pair in top_pairs.index])
top_features = list(top_features)[:20]  # 限制只取前20个

# 只选取相关性高的特征绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, fmt=".2f")
plt.title('Top Feature Correlation Heatmap')
plt.show()


# 删除高相关性特征
threshold = 0.8  # 设置相关系数的阈值
columns_to_drop = set()  # 存储要删除的列名

# 对于相关系数矩阵中每一对特征，如果相关性高于阈值，则删除一个特征
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            columns_to_drop.add(colname)

# 删除这些列
df_reduced = df.drop(columns=columns_to_drop)
print(df_reduced.info())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression

# 假设DataFrame df 包含特征1到128以及目标列tar

# 准备数据
X = df.loc[:, '1':'128']  # 选择特征列
y = df['tar']             # 选择目标列tar

# 执行ANOVA F-test
f_values, p_values = f_regression(X, y)

# 将特征名和对应的F值放入DataFrame中
features_f_values = pd.DataFrame({
    'Feature': X.columns,
    'F_value': f_values,
    'P_value': p_values
})

# 根据F值对结果进行降序排序，并选择前25个
top_25_features = features_f_values.sort_values(by='F_value', ascending=False).head(25)

# 画出前25个最重要特征的柱状图
plt.figure(figsize=(10, 8))
plt.barh(top_25_features['Feature'], top_25_features['F_value'], color='skyblue')
plt.xlabel('F-value')
plt.ylabel('Feature')
plt.title('Top 25 Features based on ANOVA F-test')
plt.gca().invert_yaxis()  # 反转y轴，以便最高的条形位于顶部
plt.show()