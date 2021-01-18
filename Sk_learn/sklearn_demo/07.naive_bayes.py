'''
    朴素贝叶斯分类：朴素贝叶斯分类是一种依据统计概率理论而实现的一种分类方式。
        观察这组数据：
            天气情况         穿衣风格    约女朋友        ==>    心情
            0（晴天）    0（休闲）    0（约了）    ==>    0（高兴）
            0            1（风骚）    1（没约）    ==>    0
            1（多云）    1           0            ==>    0
            0            2（破旧）    1           ==>    1（郁闷）
            2（下雨）    2            0            ==>    0
            ...            ...            ...            ==>    ...
            0            1            0            ==>    ？

        通过上述训练样本如何预测：晴天、穿着休闲、没有约女朋友时的心情？可以整理相同特征值的样本，计算属于某类别的概率即可。
        但是如果在样本空间没有完全匹配的数据该如何预测？

    贝叶斯定理：P(A|B)=P(B|A)P(A)/P(B) <== P(A, B) = P(A) P(B|A) = P(B) P(A|B)
        例如：
            假设一个学校里有60%男生和40%女生.女生穿裤子的人数和穿裙子的人数相等,所有男生穿裤子.
            一个人在远处随机看到了一个穿裤子的学生.那么这个学生是女生的概率是多少?
                P(女) = 0.4
                P(裤子|女) = 0.5
                P(裤子) = 0.6 + 0.2 = 0.8
                P(女|裤子) = P(裤子|女) * P(女) / P(裤子) = 0.5 * 0.4 / 0.8 = 0.25

        根据贝叶斯定理，如何预测：晴天、穿着休闲、没有约女朋友时的心情？
            P(晴天,休闲,没约,高兴)
            = P(晴天|休闲,没约,高兴) P(休闲,没约,高兴)
            = P(晴天|休闲,没约,高兴) P(休闲|没约,高兴) P(没约,高兴)
            = P(晴天|休闲,没约,高兴) P(休闲|没约,高兴) P(没约|高兴)P(高兴)
            （ 朴素：条件独立，特征值之间没有因果关系）
            = P(晴天|高兴) P(休闲|高兴) P(没约|高兴)P(高兴)
        由此可得，统计总样本空间中晴天、穿着休闲、没有约女朋友时高兴的概率，
        与晴天、穿着休闲、没有约女朋友时不高兴的概率，择其大者为最终结果。

    高斯贝叶斯分类器相关API：
            import sklearn.naive_bayes as nb
            # 创建高斯分布朴素贝叶斯分类器
            model = nb.GaussianNB()
            model.fit(x, y)
            result = model.predict(samples)

    案例：multiple1.txt
'''
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb

data = np.loadtxt('./ml_data/multiple1.txt', delimiter=',', unpack=False, dtype='f8')
print(data.shape)
x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 训练NB模型，完成分类业务
model = nb.GaussianNB()
model.fit(x, y)

# 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
test_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
test_y = model.predict(test_x)
grid_z = test_y.reshape(grid_x.shape)

# 画图
mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], s=80, c=y, cmap='jet', label='Samples')

mp.legend()
mp.show()


