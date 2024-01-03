import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class MulticlassLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

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

    def predict_proba(self, X):
        X = self.preprocess_data(X)
        X = np.insert(X, 0, 1, axis=1)
        return self.softmax(np.dot(X, self.weights))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Classification Report:\n{class_report}')
        return accuracy, conf_matrix, class_report

    def cross_validate(self, X, y, cv=5):
        kfold = StratifiedKFold(n_splits=cv)
        accuracies = []
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.fit(X_train, y_train)
            accuracy, _, _ = self.evaluate(X_test, y_test)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        print(f'Mean CV Accuracy: {mean_accuracy}')
        return mean_accuracy
