
import pandas as pd

# 加载CSV文件
df = pd.read_csv('./datas/cancer/okDatas.csv')
info_dict = {'Column Name': df.columns, 'Non-Null Count': df.count(), 'Dtype': df.dtypes}
info_table = pd.DataFrame.from_dict(info_dict).reset_index(drop=True)

# 输出info的三线表
print(info_table)

# 显示前几行数据以检查列名和数据
print(df.head())

# 检查是否存在缺失值
missing_values = df.isnull().sum()
print(missing_values)

import matplotlib.pyplot as plt
import seaborn as sns



# 核密度曲线图的函数
def plot_kde(df, features, num_per_row):
    fig, axes = plt.subplots(len(features) // num_per_row, num_per_row, figsize=(14, 4 * (len(features) // num_per_row)))
    axes = axes.flatten()  # 展平轴数组以便于循环
    for i, feature in enumerate(features):
        sns.kdeplot(data=df, x=feature, ax=axes[i], shade=True)
        axes[i].set_title(feature)
    plt.tight_layout()
    plt.show()



# 绘制每五个特征的核密度曲线图
features = df.columns[1:]  # 排除目标列
for i in range(0, len(features), 5):
    plot_kde(df, features[i:i+5], 5)



# 为不同类型的特征值（均值、误差、最大值）绘制热力图的函数
def plot_heatmap(df, value_type):
    # 筛选包含特定类型值的列（例如均值、误差、最大值）
    feature_cols = [col for col in df.columns if value_type in col]
    corr = df[feature_cols].corr()  # 计算相关性矩阵

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title(f"Heatmap of {value_type} features")
    plt.show()



# 分别为均值、误差、最大值绘制热力图
plot_heatmap(df, 'mean')
plot_heatmap(df, 'se')
plot_heatmap(df, 'worst')


from scipy.stats import f_oneway

# 方差分析（ANOVA）
def anova_test(df, target, features):
    groups = df.groupby(target)
    for feature in features:
        anova_data = [group[feature].values for name, group in groups]
        f_stat, p_val = f_oneway(*anova_data)
        print(f"Feature: {feature} - F-statistic: {f_stat}, P-value: {p_val}")

# 执行方差分析
anova_test(df, 'diagnosis', features)

# 相关系数矩阵
corr_matrix = df[features].corr()

# 输出相关系数矩阵
print(corr_matrix)




# 计算方差
variances = df.iloc[:, 2:].var()

# 每10个特征一组，绘制独立的柱状图
num_plots = len(variances) // 10 + 1  # 计算需要绘制的图的数量

for i in range(num_plots):
    start_index = i * 10
    end_index = min((i + 1) * 10, len(variances))
    variances_subset = variances[start_index:end_index]

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制柱状图
    variances_subset.plot(kind='bar', ax=ax)
    ax.set_xlabel('Features')
    ax.set_ylabel('Variance')
    ax.set_title(f'Variance Analysis - Group {i+1}')
    ax.set_xticklabels(variances_subset.index, rotation=45)

    plt.tight_layout()
    plt.show()


