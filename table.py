import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 混淆矩阵数据
confusion_matrix = np.array([[32,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                             [0, 20, 1, 0, 0, 0, 0, 1, 8, 0],
                             [0, 1, 18, 0, 0, 0, 0, 0, 12, 0],
                             [0, 1, 0, 30, 0, 2, 0, 2, 7, 0],
                             [0, 0, 0, 0, 32, 0, 1, 4, 0, 0],
                             [0, 0, 0, 1, 0, 33, 0, 3, 1, 0],
                             [0, 0, 1, 0, 0, 0, 43, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 33, 0, 0],
                             [0, 1, 1, 1, 0, 1, 0, 0, 36, 0],
                             [1, 1, 0, 0, 1, 1, 0, 4, 6, 18]])

# 绘制混淆矩阵的可视化
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

# 图表标题和标签
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# 显示图表
plt.show()
