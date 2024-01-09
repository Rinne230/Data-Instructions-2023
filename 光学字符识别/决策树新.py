import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯
from sklearn.datasets import load_digits  # 导入数字数据集
from sklearn.model_selection import train_test_split  # 用于对数据集的拆分
from sklearn import metrics

# 读取样本数据集
data = load_digits ()
print (data)

# 提取特征数据和标签数据
feature = data.data
target = data.target

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split (feature, target, test_size=0.2, random_state=2021)
X=data.data

'''
data = pd.read_csv('okgas.csv')
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]



# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
'''


# 决策树模型
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 交叉验证
cv = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

# 交叉验证曲线
plt.figure(figsize=(10, 5))
plt.title("Cross Validation Scores")
plt.plot(range(1, len(cross_val_scores) + 1), cross_val_scores, marker='o')
for i, score in enumerate(cross_val_scores):
    plt.text(i + 1, score, f"{score:.2f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.xticks(range(1, len(cross_val_scores) + 1))
plt.show()





# 预测和评估指标
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 clf 是你的分类器，X_test 是测试集的特征，y_test 是测试集的真实标签
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# 获取宏平均的评价指标（忽略最后的 'support'）
metrics = {k: v for k, v in report['macro avg'].items() if k != 'support'}

# 分类模型评估指标柱状图
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title('Classification Report')

# 在每个条形上方添加文本标签
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),  # 格式化为两位小数
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                textcoords='offset points')

plt.ylim(0, 1)  # 确保有足够空间在顶部显示数值
plt.show()

# ROC曲线图和AUC
'''
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
'''
# 混淆矩阵的热力图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.tree import plot_tree

unique_classes = np.unique(y_test)
class_names = [str(cls) for cls in unique_classes]

# 绘制决策树，可以考虑调整fontsize参数以改变文字大小
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True,
          feature_names=['Feature' + str(i) for i in range(X.shape[1])],
          class_names=class_names,
          rounded=True,
          proportion=True,
          fontsize=6,  # 调整文字大小
          max_depth=5)  # 限制树的深度，只显示前几层
plt.title("Decision Tree Visualization")
plt.show()

