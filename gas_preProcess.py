import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 定义一个函数来解析数据行
def parse_line(line):
    parts = line.split()  # 按空格分割
    # 提取标签
    label = int(parts[0].split(';')[0])
    # 提取特征，转换为 (特征编号, 特征值) 对
    features = [part.split(':') for part in parts[1:]]
    # 将特征转换为一个字典，其中键为特征编号，值为特征值
    features = {int(k): float(v) for k, v in features}
    return label, features

# 用于存储解析后的数据的列表
labels = []
feature_list = []

# 循环读取所有的数据文件
for i in range(1, 11):
    file_name = f'./datas/gas/batch{i}.dat'
    with open(file_name, 'r') as file:
        for line in file:
            label, features = parse_line(line.strip())
            labels.append(label)
            feature_list.append(features)

# 从字典创建DataFrame
df = pd.DataFrame(feature_list)

# 填充缺失值（如果有的话）
df.fillna(0, inplace=True)
df['tar'] = labels
df.to_csv("./datas/gas/dataset.csv", index = False)