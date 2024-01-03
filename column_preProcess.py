import pandas as pd

# 加载CSV文件
df = pd.read_csv('./datas/vertebral/column_2C.csv')
def min_max_normalize(x):
    return (x - x.min())/(x.max() - x.min())
for col in df.columns[:-1]:
    df[col] = min_max_normalize(df[col])
tar_dict = {"NO":0,"AB":1}
df["tar"] = df["tar"].map(tar_dict)
df.iloc[:,1:].describe()
df.to_csv("./datas/vertebral/ok2C.csv")
info_dict = {'Column Name': df.columns, 'Non-Null Count': df.count(), 'Dtype': df.dtypes}
info_table = pd.DataFrame.from_dict(info_dict).reset_index(drop=True)


df = pd.read_csv('./datas/vertebral/column_3C.csv')
def min_max_normalize(x):
    return (x - x.min())/(x.max() - x.min())
for col in df.columns[:-1]:
    df[col] = min_max_normalize(df[col])
tar_dict = {"NO":0,"DH":1,"SL":2}
df["tar"] = df["tar"].map(tar_dict)
df.iloc[:,1:].describe()
df.to_csv("./datas/vertebral/ok3C.csv")
info_dict = {'Column Name': df.columns, 'Non-Null Count': df.count(), 'Dtype': df.dtypes}
info_table = pd.DataFrame.from_dict(info_dict).reset_index(drop=True)
