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
    def __init__(self, min_samples_split=2, max_depth=float("inf"), min_gain=1e-9, criterion='entropy', class_weights=None):
        self.min_samples_split = min_samples_split  # 最小样本拆分数
        self.max_depth = max_depth  # 最大深度
        self.min_gain = min_gain  # 最小信息增益
        self.criterion = criterion  # 用于选择最佳分割的准则
        self.class_weights = class_weights  # 类别权重字典
        self.root = None  # 决策树根节点

    def _most_common_class(self, y):
        # 使用 numpy 的 bincount 函数找到最频繁的类别
        counts = np.bincount(y)
        most_common_class = np.argmax(counts)
        return most_common_class

    def _gini(self, y):
        # Calculate the Gini impurity weighted by class weights
        hist = np.bincount(y, minlength=len(self.class_weights))  # 统计每个类别标签的出现次数
        total = np.sum(hist)  # 计算总实例数
        weighted_gini = 1  # 初始化加权基尼不纯度为1
        #print(self.class_weights)
        for cls in np.unique(y):  # 遍历每个唯一类别标签
            p = hist[cls] / total  # 计算实例属于类别cls的概率
            weight = self.class_weights.get(cls, 1)  # 获取类别cls的权重，如果未指定则默认为1
            weighted_gini -= weight * (p ** 2)  # 计算类别cls对加权基尼不纯度的贡献
            #print("标签{},权重{}".format(cls,weight))


        return weighted_gini  # 返回加权基尼不纯度

    def _grow_tree(self, X, y, depth=0):
        # 创建节点并获取最佳分割点
        best_idx, best_thr, best_gain = self._best_split(X, y)

        # 如果无法再分割或达到最大深度，将当前节点设为叶节点
        if best_gain == 0 or depth >= self.max_depth:
            return DecisionTreeNode(value=self._most_common_class(y))

        # 根据最佳分割点分割数据集
        left_mask = X[:, best_idx] <= best_thr
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # 递归构建左右子树
        left_child = self._grow_tree(left_X, left_y, depth + 1)
        right_child = self._grow_tree(right_X, right_y, depth + 1)

        # 创建当前节点
        return DecisionTreeNode(feature=best_idx, threshold=best_thr, left=left_child, right=right_child)

    def fit(self, X, y):
        self.features = range(X.shape[1])  # 存储所有特征的索引
        self.feature_importances_ = np.zeros(len(self.features))  # 初始化特征重要性数组
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(unique_classes,class_counts)
        # 计算与类频率成反比的类权重
        total_samples = len(y)
        self.class_weights = {unique_classes[i]: total_samples / (len(unique_classes) * class_counts[i]) for i in
                              range(len(unique_classes))}

        self.root = self._grow_tree(X, y)  # 构建决策树
        length=len(self.class_weights)
        for i in range(length):
            self.class_weights[i]=self.class_weights[i] ** 2
        self.feature_importances_ /= self.feature_importances_.sum()  # 归一化特征重要性

    def _grow_tree(self, X, y, depth=0):
        # 找到最佳的特征和阈值来分割数据
        best_idx, best_thr, best_gain = self._best_split(X, y)

        # 如果无法再分割或达到最大深度，创建一个叶节点
        if best_gain == 0 or depth >= self.max_depth:
            return DecisionTreeNode(value=self._most_common_class(y))

        # 根据最佳特征和阈值分割数据集
        left_mask = X[:, best_idx] <= best_thr
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # 递归构建左右子树
        left_child = self._grow_tree(left_X, left_y, depth + 1)
        right_child = self._grow_tree(right_X, right_y, depth + 1)

        # 创建当前节点，使用最佳特征和阈值
        return DecisionTreeNode(feature=best_idx, threshold=best_thr, left=left_child, right=right_child)

    def _best_split(self, X, y):
        best_gain = 0  # 最佳信息增益
        best_idx, best_thr = None, None  # 最佳特征索引和阈值

        n_features = X.shape[1]  # 特征数量
        if self.criterion == 'gini':
            current_uncertainty = self._gini(y)  # 当前的不确定性（基尼指数）
        else:
            current_uncertainty = self._entropy(y)  # 当前的不确定性（熵）

        num_classes = np.unique(y)  # 类别的取值

        for idx in self.features:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))  # 根据特征值对数据进行排序
            num_left = [0] * len(set(y))  # 左子节点中每个类别的样本数量
            num_right = Counter(classes)  # 右子节点中每个类别的样本数量

            for i in range(1, len(y)):  # 可能的分割位置
                c = classes[i - 1]
                class_idx = np.where(num_classes == c)[0][0]  # 获取类别 c 的索引
                num_left[class_idx] += 1
                num_right[c] -= 1
                if thresholds[i] == thresholds[i - 1]:
                    continue

                p_left = float(i) / len(y)  # 左子节点的样本比例
                p_right = 1 - p_left  # 右子节点的样本比例
                gain = current_uncertainty - p_left * self._entropy(list(num_left)) - p_right * self._entropy(
                    list(num_right.values()))  # 计算信息增益

                if gain > best_gain:  # 更新最佳信息增益和对应的特征索引、阈值
                    best_gain = gain
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

                    # 更新特征重要性
                    self.feature_importances_[idx] += gain

        return best_idx, best_thr, best_gain

    def _entropy(self, y):
        hist = np.bincount(y, minlength=len(self.class_weights))  # 类别频数
        total = np.sum(hist)  # 总样本数量
        weighted_entropy = 0  # 加权熵
        for cls in np.unique(y):
            p = hist[cls] / total  # 类别占比
            weight = self.class_weights.get(cls, 1)  # 类权重
            if p > 0:
                weighted_entropy -= weight * p * np.log2(p)  # 计算加权熵
        return weighted_entropy

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while not node.is_leaf_node():  # 循环直到到达叶节点
            if inputs[node.feature] < node.threshold:
                node = node.left  # 根据特征值判断向左子树移动
            else:
                node = node.right  # 根据特征值判断向右子树移动
        return node.value  # 返回叶节点的预测类别

    def score(self, X, y):
        y_pred = self.predict(X)  # 预测类别
        return np.mean(y_pred == y)  # 返回准确率

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
        y_pred = self.predict(X)  # 预测类别
        return precision_score(y, y_pred, average='macro')  # 返回精确率

    def recall(self, X, y):
        y_pred = self.predict(X)  # 预测类别
        return recall_score(y, y_pred, average='macro')  # 返回召回率

    def f1(self, X, y):
        y_pred = self.predict(X)  # 预测类别
        return f1_score(y, y_pred, average='macro')  # 返回F1值

    def get_confusion_matrix(self, X, y):
        y_pred = self.predict(X)  # 预测类别
        return confusion_matrix(y, y_pred)  # 返回混淆矩阵

# 使用决策树
if __name__ == "__main__":
    data = pd.read_csv('./datas/cancer/okDatas.csv')

    #  Assuming the last column is the label
    X = data.iloc[:, 1:]  # Features
    y = data.iloc[:, 0]  # Target

    # Calculate the correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95 (this threshold can be changed)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    # Drop highly correlated features
    X_reduced = X.drop(columns=to_drop)

    # If there are still more than 10 features, select the first 10
    # You might want to implement a more sophisticated feature selection method here
    if X_reduced.shape[1] > 10:
        X_reduced = X_reduced.iloc[:, :10]






    X = data.iloc[:, 1:10].values
    y = data.iloc[:, 0].values  # Ensure y contains only the class labels
    classweight = [1, 6]
    # Initialize the decision tree classifier
    classifier = DecisionTreeClassifier(
        min_samples_split=10,
        max_depth=30,

        class_weights=classweight
    )

    # Fit the model
    classifier.fit(X, y)

    # Evaluate the model on the training set
    print("Model score:", classifier.score(X, y))

    # Prune the tree
    classifier.prune(X, y)
    print("Model evaluation", classifier.get_confusion_matrix(X, y))
    # Re-evaluate the model
    print("Post-pruning model score:", classifier.score(X, y))
    metrics = ["Model score", "Post-pruning model score"]
    scores = [classifier.score(X, y), classifier.score(X, y)]
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import seaborn as sns
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.metrics import roc_curve, auc
    # Create the bar chart
    plt.bar(metrics, scores)
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.title("Model Evaluation")
    plt.show()
    confusion_matrix = classifier.get_confusion_matrix(X, y)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    plt.barh(range(len(classifier.feature_importances_)), classifier.feature_importances_)
    plt.yticks(range(len(classifier.feature_importances_)),
               ['Feature ' + str(i) for i in range(len(classifier.feature_importances_))])
    plt.xlabel('Feature Importance')
    plt.show()