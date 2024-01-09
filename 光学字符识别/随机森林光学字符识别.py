import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.datasets import load_digits  # 导入数字数据集
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 加载数据集
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

'''
feature_names = data.feature_names

# 数据预处理：删除相关性高的特征
df = pd.DataFrame(X, columns=feature_names)
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
df_reduced = df.drop(to_drop, axis=1)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(df_reduced, y, test_size=0.3, random_state=42)
'''
# 创建随机森林分类器实例并训练
'''
data = pd.read_csv('okGas.csv')

# 分离特征和标签
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)
'''

data = load_digits ()
print (data)

# 提取特征数据和标签数据
feature = data.data
target = data.target

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split (feature, target, test_size=0.2, random_state=2021)
X=data.data

# 创建随机森林分类器实例并训练
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)




# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    rf, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.title('Learning curve for RandomForestClassifier')
plt.xlabel('Training size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

# 验证曲线（以 n_estimators 参数为例）
param_range = np.arange(10, 200, 20)
train_scores, test_scores = validation_curve(
    rf, X_train, y_train, param_name="n_estimators", param_range=param_range,
    cv=3, scoring="accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)
plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
plt.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
plt.title('Validation curve for RandomForestClassifier')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

# ROC 和 AUC
'''
y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
'''

y_pred = rf.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# 将评估指标组成一个列表和一个对应的名称列表
metrics = [accuracy, precision, recall, f1]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# 绘制柱状图
plt.bar(metric_names, metrics, color=['blue', 'green', 'orange', 'red'])

# 添加柱状图上的数值显示
for index, value in enumerate(metrics):
    plt.text(index, value, f"{value:.4f}", ha='center', va='bottom', color='black')

# 设置坐标轴标签和图表标题
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Evaluation Metrics for RandomForestClassifier')

# 显示图表
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 假设 rf 是你的随机森林模型，X 是你的特征矩阵
# 特征重要性


# 假设 rf 是你的随机森林模型，X 是包含特征名称的 DataFrame
importances = rf.feature_importances_
feature_names = X.columns

# 按照特征重要性排序，最重要的特征排在前面
sorted_indices = np.argsort(importances)[::-1]

# 每次显示的特征数量
features_per_chart = 35

'''
# 定义一个函数来绘制特征重要性
def plot_feature_importances(start_index, end_index):
    # 限制索引不超过特征数量
    end_index = min(end_index, len(feature_names))

    # 创建图表
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature importances by RandomForest (Features {start_index + 1}-{end_index})")
    plt.barh(range(end_index - start_index), importances[sorted_indices][start_index:end_index], color='lightblue',
             align='center')
    plt.yticks(range(end_index - start_index), [feature_names[i] for i in sorted_indices[start_index:end_index]])
    plt.gca().invert_yaxis()
    plt.xlabel("Relative Importance")
    plt.show()


# 通过多个块来显示所有特征的重要性
total_features = len(feature_names)
for start_index in range(0, total_features, features_per_chart):
    end_index = start_index + features_per_chart
    plot_feature_importances(start_index, end_index)
'''