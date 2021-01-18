
import numpy as np
import pandas as pd


"""
重构综合预测模型。使用tensorflow2.0。
"""


# 数据预处理，读入数据-训练集
def read_data(data_dir):
    data1 = pd.read_excel(data_dir, header=None)
    print("****************************")
    X1 = np.array(data1.loc[0:9999, 0:3])
    label = data1.loc[0:9999, 4]
    Y1 = []
    for i in label:
        temp = [0, 0, 0, 0, 0]
        temp[i] = 1
        Y1.append(temp)
    Y1 = np.array(Y1)
    return X1, Y1


# 数据预处理，读入数据-测试集
def read_data_v(data_dir):
    data1 = pd.read_excel(data_dir, header=None)
    print("****************************")
    X1 = np.array(data1.loc[0:3999, 0:3])
    label = data1.loc[0:3999, 4]
    Y1 = []
    for i in label:
        temp = [0, 0, 0, 0, 0]
        temp[i] = 1
        Y1.append(temp)
    Y1 = np.array(Y1)
    return X1, Y1

# 输入数据
