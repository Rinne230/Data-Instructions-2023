import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seedVal = 114514
np.random.seed(seedVal)

class navieBayes:
    def __init__(self):
        pass
    
    def fit(self, X, y): # 训练
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.mean = {} # 均值
        self.var = {} # 方差
        self.prior = {} # 先验频率
        for cl in self.classes:
            self.prior[cl] = np.sum(self.y == cl) / len(self.y)
            clX = self.X[self.y == cl]
            self.mean[cl] = np.mean(clX, axis = 0)
            self.var[cl] = np.var(clX, axis = 0)

    def predict(self, X): # 预测
        posterior = []
        for cl in self.classes:
            logprior = np.log(self.prior[cl])
            maxLike = np.sum(np.log(self.prob_density(X, cl)), axis = 1)
            predictClass = logprior + maxLike
            posterior.append(predictClass)
        return self.classes[np.argmax(posterior, axis = 0)]
    
    def prob_density(self, X, cl): # 概率密度
        mean = self.mean[cl]
        var = self.var[cl]
        numerator = np.exp(-(X - mean)**2 / (2 * var))
        denomiator = np.sqrt(2 * np.pi * var)
        return numerator / denomiator

    def score(self, y_pred, y): # 准确度
        acc = (y_pred == y).sum() / len(y)
        return acc 

