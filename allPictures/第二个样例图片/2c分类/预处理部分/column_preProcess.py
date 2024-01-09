
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

df = pd.read_csv('column_3C.csv')

# 数据归一化函数
def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# 对每个特征列进行归一化，除了最后一列目标变量
for col in df.columns[:-1]:
    df[col] = min_max_normalize(df[col])

class_counts_before_mapping = df["tar"].value_counts()

# 绘制柱状图
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts_before_mapping.index, y=class_counts_before_mapping.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# 映射目标变量
tar_dict = {"NO": 0, "AB": 1,"SL":2}
df["tar"] = df["tar"].map(tar_dict)  # 确保这里的"Class"是目标变量的正确列名



# 删除原有的class列（如果存在）
if 'class' in df.columns:
    df = df.drop(columns=['class'])

# 保存处理后的数据到新CSV文件
df.to_csv("ok3C.csv")

# 创建信息表并打印
info_dict = {'Column Name': df.columns, 'Non-Null Count': df.count(), 'Dtype': df.dtypes}
info_table = pd.DataFrame.from_dict(info_dict).reset_index(drop=True)
print(info_table)

# 显示前几行数据以检查列名和数据
print(df.head())

# 检查是否存在缺失值
missing_values = df.isnull().sum()
print(missing_values)

# 核密度曲线图的函数
def plot_kde(df, features, num_per_row):
    fig, axes = plt.subplots(len(features) // num_per_row, num_per_row, figsize=(14, 4 * (len(features) // num_per_row)))
    axes = axes.flatten()  # 展平轴数组以便于循环
    for i, feature in enumerate(features):
        sns.kdeplot(data=df, x=feature, ax=axes[i], fill=True)
        axes[i].set_title(feature)
    plt.tight_layout()
    plt.show()



# 绘制每三个特征的核密度曲线图
features = df.columns[:-1]  # 排除目标列
for i in range(0, len(features), 3):
    plot_kde(df, features[i:i+3], 3)

# 方差分析（ANOVA）
def anova_test(df, target, features):
    groups = df.groupby(target)
    for feature in features:
        anova_data = [group[feature].values for name, group in groups]
        f_stat, p_val = f_oneway(*anova_data)
        print(f"Feature: {feature} - F-statistic: {f_stat}, P-value: {p_val}")


# 执行方差分析
anova_test(df, 'tar', features)  # 使用'tar'作为目标变量

# 相关系数矩阵
corr_matrix = df[features].corr()

# 绘制特征之间的相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap of Feature Correlations")
plt.show()

# 计算方差
variances = df.iloc[:, :-1].var()

# 每6个特征一组，绘制独立的柱状图
num_plots = len(variances) // 6 + (len(variances) % 6 > 0)

for i in range(num_plots):
    start_index = i * 6
    end_index = min((i + 1) * 6, len(variances))
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
