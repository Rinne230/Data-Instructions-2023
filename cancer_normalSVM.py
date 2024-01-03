import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd

class SMO:
    def __init__(self, X, y, C, kernel, tol, max_passes=10):
        self.X = X  # 样本特征 m*n m个样本 n个特征
        self.y = y  # 样本标签 m*1
        self.C = C  # 惩罚因子, 用于控制松弛变量的影响
        self.kernel = kernel  # 核函数
        self.tol = tol  # 容忍度
        self.max_passes = max_passes  # 最大迭代次数
        self.m, self.n = X.shape
        self.alpha = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(self.n)

    # 计算核函数
    def K(self, i, j):
        if self.kernel == 'linear':
            return np.dot(self.X[i].T, self.X[j])
        elif self.kernel == 'rbf':
            gamma = 0.5
            return np.exp(-gamma * np.linalg.norm(self.X[i] - self.X[j]) ** 2)

        else:
            raise ValueError('Invalid kernel specified')

    def predict(self, X):
        pred = np.zeros_like(X[:, 0])
        pred = np.dot(X_test, self.w) + self.b
        return np.sign(pred)

    def train(self):
        """
        训练模型
        :return:
        """
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.m):
                # 计算E_i, E_i = f(x_i) - y_i, f(x_i) = w^T * x_i + b
                # 计算误差E_i
                E_i = 0
                for ii in range(self.m):
                    E_i += self.alpha[ii] * self.y[ii] * self.K(ii, i)
                E_i += self.b - self.y[i]
                # 检验样本x_i是否满足KKT条件
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # 随机选择样本x_j
                    j = np.random.choice(list(range(i)) + list(range(i + 1, self.m)), size=1)[0]
                    # 计算E_j, E_j = f(x_j) - y_j, f(x_j) = w^T * x_j + b
                    # E_j用于检验样本x_j是否满足KKT条件
                    E_j = 0
                    for jj in range(self.m):
                        E_j += self.alpha[jj] * self.y[jj] * self.K(jj, j)
                    E_j += self.b - self.y[j]

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    # L和H用于将alpha[j]调整到[0, C]之间
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    # 如果L == H，则不需要更新alpha[j]
                    if L == H:
                        continue

                    # eta: alpha[j]的最优修改量
                    eta = 2 * self.K(i, j) - self.K(i, i) - self.K(j, j)
                    # 如果eta >= 0, 则不需要更新alpha[j]
                    if eta >= 0:
                        continue

                    # 更新alpha[j]
                    self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta
                    # 根据取值范围修剪alpha[j]
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # 检查alpha[j]是否只有轻微改变，如果是则退出for循环
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha[i]
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # 更新b1和b2
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K(i, i) \
                         - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K(i, j)
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K(i, j) \
                         - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K(j, j)

                    # 根据b1和b2更新b
                    if 0 < self.alpha[i] and self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # 提取支持向量和对应的参数
        idx = self.alpha > 0  # 支持向量的索引
        # SVs = X[idx]
        selected_idx = np.where(idx)[0]
        SVs = X[selected_idx]
        SV_labels = y[selected_idx]
        SV_alphas = self.alpha[selected_idx]

        # 计算权重向量和截距
        self.w = np.sum(SV_alphas[:, None] * SV_labels[:, None] * SVs, axis=0)
        self.b = np.mean(SV_labels - np.dot(SVs, self.w))
        print("w", self.w)
        print("b", self.b)

    def score(self, X, y):
        predict = self.predict(X)
        print("predict", predict)
        print("target", y)
        return np.mean(predict == y)



data = pd.read_csv("./datas/cancer/okDatas.csv")
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
y[y != 0] = -1
y[y == 0] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svc = SMO(X, y, C = 1, kernel= 'linear', tol= 0.001)
svc.train()
breast_cancer_test_pred_svm = svc.predict(X_test)
print(metrics.classification_report(y_test, breast_cancer_test_pred_svm))
print(metrics.confusion_matrix(y_test, breast_cancer_test_pred_svm))
print(metrics.accuracy_score(y_test, breast_cancer_test_pred_svm))