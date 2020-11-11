# coding=utf-8
import tensorflow as tf
import numpy as np
import scipy.io as scio
# 加载训练数据，输入为X1，输出为Y1；X1为8维的URL信息数据，Y1为2维数据，[1，0]代表钓鱼网址[0，1]代表正常网址
def read_data(path):
    data = scio.loadmat(path)
    return data['X'],np.array([i.tolist().index(1) for i in data['Y']])
x_test, y_test = read_data(r'./datasets/test_data_716.mat')

print(x_test[5],y_test[5])
print(x_test[6],y_test[6])
print(x_test[7],y_test[7])
print(x_test[8],y_test[8])