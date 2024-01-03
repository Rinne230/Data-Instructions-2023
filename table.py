import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 混淆矩阵数据
confusion_matrix = np.array([[125, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 132, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 130, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 129, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 121, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 116, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 128, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 124, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 131, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 121]])

# 绘制混淆矩阵的可视化
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

# 图表标题和标签
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# 显示图表
plt.show()
