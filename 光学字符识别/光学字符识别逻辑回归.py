import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # 导入数字数据集

class MulticlassLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.history = []

    def preprocess_data(self, X):
        for i in range(X.shape[1]):
            if np.any(X[:, i] > 5):
                X[:, i] = self.z_score_normalize(X[:, i])
        return X

    def z_score_normalize(self, column):
        mean = np.mean(column)
        std = np.std(column)
        return (column - mean) / std

    def softmax(self, z):
        z = np.clip(z, -20, 20)
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, h, y):
        epsilon = 1e-7
        h = np.clip(h, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y * np.log(h), axis=1))

    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        X_train = np.insert(X_train, 0, 1, axis=1)
        self.weights = np.zeros((X_train.shape[1], len(np.unique(y_train))))

        y_one_hot = pd.get_dummies(y_train).values

        for i in range(self.epochs):
            z = np.dot(X_train, self.weights)
            h = self.softmax(z)
            gradient = self.gradient_descent(X_train, h, y_one_hot)
            self.weights -= self.lr * gradient
            accuracy = accuracy_score(y_train, np.argmax(h, axis=1))
            loss = self.loss(h, y_one_hot)
            # 添加到历史记录中
            self.history.append((accuracy, loss))

    def predict_proba(self, X):
        X = self.preprocess_data(X)
        X = np.insert(X, 0, 1, axis=1)
        return self.softmax(np.dot(X, self.weights))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, X_test, y_test, verbose=0):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        if verbose:
            print(f'Accuracy: {accuracy}')
            print(f'Confusion Matrix:\n{conf_matrix}')
            print(f'Classification Report:\n{class_report}')
        return accuracy, conf_matrix, class_report

    def cross_validate(self, X, y, cv=5, verbose=0):
        kfold = StratifiedKFold(n_splits=cv)
        accuracies = []
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.fit(X_train, y_train)
            accuracy, _, _ = self.evaluate(X_test, y_test, verbose=verbose)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        if verbose:
            print(f'Mean CV Accuracy: {mean_accuracy}')
        return mean_accuracy

    def plot_roc_curve(self, X_test, y_test):
        y_score = self.predict_proba(X_test)
        y_test_bin = pd.get_dummies(y_test).values

        # Compute ROC curve and ROC area for each class
        n_classes = y_test_bin.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()



# 使用MulticlassLogisticRegression类

data = pd.read_csv('okDatas.csv')




y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values






# 选择最佳的学习率和epoch数
learning_rates = [0.001, 0.01, 0.1]
epochs_options = [100, 500, 1000]

best_accuracy = 0
best_lr = 0
best_epoch = 0

for lr in learning_rates:
    for epoch in epochs_options:
        model = MulticlassLogisticRegression(lr=lr, epochs=epoch)
        mean_accuracy = model.cross_validate(X, y, cv=5, verbose=0)  # Set verbose to 0 to prevent printing
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_lr = lr
            best_epoch = epoch

print(f'Best LR: {best_lr}, Best Epoch: {best_epoch}, Best Accuracy: {best_accuracy}')


best_epoch=20
# 使用最佳参数训练模型
model = MulticlassLogisticRegression(lr=best_lr, epochs=best_epoch)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
model.fit(X_train, y_train)
accuracy, conf_matrix, class_report = model.evaluate(X_test, y_test, verbose=1)  # Set verbose to 1 to print metrics

print("混淆矩阵",conf_matrix)


import matplotlib.pyplot as plt
model.plot_roc_curve(X_test, y_test)
accuracies = [history[0] for history in model.history]
losses = [history[1] for history in model.history]

# 绘制曲线图
plt.plot(range(1, best_epoch+1), accuracies, label='Accuracy')
plt.plot(range(1, best_epoch+1), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Model Iteration Process')
plt.legend()
plt.show()



# 绘制评估指标的柱状图
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Evaluation Metrics')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
class_labels = np.unique(y_test)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Heatmap')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()