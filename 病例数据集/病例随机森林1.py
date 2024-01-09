import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=float("inf"), min_gain=1e-7):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.root = None

    def fit(self, X, y):
        self.features = range(X.shape[1])
        self.feature_importances_ = np.zeros(len(self.features))
        self.root = self._grow_tree(X, y)
        self.feature_importances_ /= self.feature_importances_.sum()  # 归一化特征重要性

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(value=predicted_class)

        if depth < self.max_depth and len(y) >= self.min_samples_split and len(set(y)) > 1:
            idx, thr, gain = self._best_split(X, y)
            if gain is not None and gain > self.min_gain:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth+1)
                node.right = self._grow_tree(X_right, y_right, depth+1)
        return node

    def _best_split(self, X, y):
        best_gain = 0
        best_idx, best_thr = None, None
        current_uncertainty = self._entropy(y)
        n_features = X.shape[1]

        for idx in self.features:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(set(y))
            num_right = Counter(classes)

            for i in range(1, len(y)):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                if thresholds[i] == thresholds[i - 1]:
                    continue

                p_left = float(i) / len(y)
                p_right = 1 - p_left
                gain = current_uncertainty - p_left * self._entropy(list(num_left)) - p_right * self._entropy(list(num_right.values()))

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

                    # 更新特征重要性
                    self.feature_importances_[idx] += gain

        return best_idx, best_thr, best_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while not node.is_leaf_node():
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def prune(self, X, y):
        # 使用后剪枝简化决策树
        self._prune_node(self.root, X, y)

    def _prune_node(self, node, X, y):
        if node.is_leaf_node():
            return

        indices_left = X[:, node.feature] < node.threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        if not node.left.is_leaf_node():
            self._prune_node(node.left, X_left, y_left)
        if not node.right.is_leaf_node():
            self._prune_node(node.right, X_right, y_right)

        # 尝试将当前节点合并成叶节点
        if node.left.is_leaf_node() and node.right.is_leaf_node():
            error_before_prune = self._error(node, X, y)
            error_after_prune = self._error(node, X, y, True)

            if error_after_prune <= error_before_prune:
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.value = Counter(y).most_common(1)[0][0]

    def _error(self, node, X, y, pretend=False):
        if not pretend and node.is_leaf_node():
            return np.sum(node.value != y)

        if pretend or node.is_leaf_node():
            predicted_value = Counter(y).most_common(1)[0][0]
            return np.sum(predicted_value != y)

        else:
            indices_left = X[:, node.feature] < node.threshold
            left_error = self._error(node.left, X[indices_left], y[indices_left])
            right_error = self._error(node.right, X[~indices_left], y[~indices_left])
            return left_error + right_error

    def precision(self, X, y):
        y_pred = self.predict(X)
        return precision_score(y, y_pred, average='macro')

    def recall(self, X, y):
        y_pred = self.predict(X)
        return recall_score(y, y_pred, average='macro')

    def f1(self, X, y):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro')

    def get_confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone


class RandomForestClassifier:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=float("inf"), max_features=None, bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.trees = []
        self.feature_importances_ = np.zeros(X.shape[1])
        np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(len(X), len(X))
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y

            # Subsample features
            max_features = self.max_features
            if max_features is None:
                max_features = int(np.sqrt(X.shape[1]))
            features_indices = np.random.choice(X.shape[1], max_features, replace=False)
            tree.features = features_indices

            X_sample_subset = X_sample[:, features_indices]
            tree.fit(X_sample_subset, y_sample)
            self.trees.append(tree)

            # Accumulate feature importances
            self._accumulate_feature_importances(tree)

        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators

    def _accumulate_feature_importances(self, tree):
        node = tree.root
        stack = [node]
        while stack:
            node = stack.pop()
            if not node.is_leaf_node():
                self.feature_importances_[node.feature] += 1
                stack.append(node.left)
                stack.append(node.right)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X[:, tree.features]) for tree in self.trees])
        tree_preds_majority = np.squeeze(stats.mode(tree_preds, axis=0)[0])
        return tree_preds_majority

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_feature_importances(self):
        return self.feature_importances_


# 使用随机森林
if __name__ == "__main__":
    # 假设csv文件名为"data.csv"，第一列是类别，其余列是特征
    df = pd.read_csv("data.csv")
    X = df.drop(columns=["类别"]).values
    y = df["类别"].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 初始化随机森林分类器
    rf_clf = RandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', random_state=42)

    # 拟合模型
    rf_clf.fit(X_train, y_train)

    # 在测试集上评估模型
    print("模型评分:", rf_clf.score(X_test, y_test))

    # 计算特征重要性
    feature_importances = rf_clf.get_feature_importances()
    print("特征重要性:", feature_importances)
