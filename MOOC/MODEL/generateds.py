import numpy as np
import matplotlib.pyplot as plt

seed = 2

def generateds():
    # 基于seed产生随机数
    rdm = np.random.RandomState(seed)

    # 随机数返回300行2列矩阵 表示32组 体积和重量 座位输入数据集
    X = rdm.randn(300, 2)

    # 从X这个300行2列矩阵中 取出一行 判断如果平方和小于2 给Y赋值1 否则Y=0
    # 作为输入数据集标签（正确答案）
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]

    # 遍历Y中的背个元素 1渎职 red 其余赋值 blue 这样可视化显示 人可以直观区分
    Y_c = [['red' if y else 'blue'] for y in Y_]

    # 对数据集X和标签Y进行shape整理 第一个元素位 -1 表示 随第二个参数计算得到 第二个参数表示多少列 把X整理位n行2列，把Y整理成n行1列
    X = np.vstack(X).reshape(-1,2)
    Y_ = np.vstack(Y_).reshape(-1,1)

    return X,Y_,Y_c


# 用plt.scatter画出数据集X各行中第0列元素和第1列元素的点 即各行的(x0,x1),用各行Y_c对应的值表示颜色
# plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
# plt.show()