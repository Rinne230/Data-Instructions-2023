import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle

# 加载数据
data = load_digits()

# 检查数据中是否有缺失值
# 因为load_digits返回的特征是整数，我们可以检查是否有缺失的整数值
if np.any(np.isnan(data.data)):
    print("数据集包含缺失值")
else:
    print("数据集没有缺失值")

# 打乱数据顺序
X, y = shuffle(data.data, data.target, random_state=0)  # random_state是种子值，用于复现结果
