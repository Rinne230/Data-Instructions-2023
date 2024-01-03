import numpy as np
import geatpy as ea
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt  # 导入画图包的plt方法
import seaborn as sns  # 热力图
import matplotlib  # 导入画图工具

class gaSVM(ea.Problem):
    def __init__(self, PoolType):
        name = 'gaSVM'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）    
        varTypes = [0, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [2 ** (-8)] * Dim  # 决策变量下界
        ub = [2 ** 8] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        ori_data = pd.read_csv("./datas/cancer/wisc_bc_data.csv")
        ori_data = ori_data.drop(columns='id')
        print(ori_data.head())
        # min_max标准化
        def min_max_normalize(x):
            return (x - x.min())/(x.max() - x.min())
        
        # 数据预处理
        def preSolve(data):
            print(data.diagnosis.value_counts())
            print(data.diagnosis.value_counts()/len(data))
            dignosis_dict = {"B":0,"M":1}
            data["diagnosis"] = data["diagnosis"].map(dignosis_dict)
            for col in data.columns[1:31]:
                data[col] = min_max_normalize(data[col])
            data.iloc[:,1:].describe()
            data.to_csv("./datas/cancer/wisc_aft.csv", index = False)
        preSolve(ori_data)

        ori_data = pd.read_csv("./datas/cancer/wisc_aft.csv")

        y = ori_data['diagnosis']
        ori_data = ori_data.drop(columns='diagnosis')
        X = ori_data
        breast_cancer_minmax_train, breast_cancer_minmax_test,\
        breast_cancer_train_labels, breast_cancer_test_labels \
        = model_selection.train_test_split(X, y, test_size=0.7, random_state = 42)
        print("验证集训练集差异性：")
        print(breast_cancer_train_labels.value_counts()/len(breast_cancer_train_labels))
        print(breast_cancer_test_labels.value_counts()/len(breast_cancer_test_labels))
        # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1)

        frames_tmp = [breast_cancer_minmax_test, breast_cancer_test_labels]
        df_data_tmp = pd.concat(frames_tmp, axis=1)  # 进行数据列合并
        df_data_tmp.to_csv('./datas/cancer/data_test.csv', index=False)

        self.data = np.array(breast_cancer_minmax_train)  # 训练集的特征数据（归一化）
        self.dataTarget = np.array(breast_cancer_train_labels)
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)), [Vars] * pop.sizes, [self.data] * pop.sizes, [self.dataTarget] * pop.sizes))
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())

    def test(self, C, G):  # 代入优化后的C、Gamma对测试集进行检验
        data_test = pd.read_csv("./datas/cancer/data_test.csv")
        X_test = data_test.drop(columns="diagnosis")
        Y_test = data_test["diagnosis"]
        data_test = np.array(X_test)  # 测试集的特征数
        dataTarget_test = np.array(Y_test)  # 测试集的标签数据
        svc = svm.SVC(kernel='rbf', C=C, gamma=G).fit(self.data, self.dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
        dataTarget_predict = svc.predict(X_test)  # 采用训练好的分类器对象对测试集数据进行预测
        print("测试集数据分类正确率 = %s%%" % (len(np.where(dataTarget_predict == dataTarget_test)[0]) / len(dataTarget_test) * 100))
        print("验证集查准率: {:.2f}%".format(precision_score(Y_test, dataTarget_predict) * 100))  # 打印验证集查准率
        print("验证集查全率: {:.2f}%".format(recall_score(Y_test, dataTarget_predict) * 100))  # 打印验证集查全率
        print("验证集F1值: {:.2f}%".format(f1_score(Y_test, dataTarget_predict) * 100))  # 打印验证集F1值


def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]
    C = Vars[i, 0]
    G = Vars[i, 1]
    svc = svm.SVC(C=C, kernel='rbf', gamma=G).fit(data, dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
    scores = cross_val_score(svc, data, dataTarget, cv=30)  # 计算交叉验证的得分
    ObjV_i = [scores.mean()]  # 把交叉验证的平均得分作为目标函数值
    return ObjV_i
