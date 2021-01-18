import sklearn.tree as st
import pandas as pd
import numpy as np

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
x_train, y_train = read_data(r'C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\MOOC\TF21\999TEST\0.xinan_sync\my_datasets\train.xlsx')
x_test, y_test = read_data_v(r'C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\MOOC\TF21\999TEST\0.xinan_sync\my_datasets\val.xlsx')
# ------------------------------------------

# 创建决策树回归器模型  决策树的最大深度为4
model = st.DecisionTreeRegressor(max_depth=4)
# 训练模型
# train_x： 二维数组样本数据
# train_y： 训练集中对应每行样本的结果
model.fit(x_train, y_train)
# 测试模型

'''
3	0.4	0	3	3
3	0.2	1	1	1
2	0.9	1	5	2
4	0.01	1	1	1
2	0.19	0	5	2
3	0.4	0	3	3
3	0.2	1	1	1
2	0.9	1	5	2
4	0.01	1	1	1
2	0.19	0	5	2
'''
pred_test_y = model.predict(np.array([[3,0.4,0,3],[3,0.2,1,1],[2,0.9,1,5],[4,0.01,1,1],[2,0.19,0,5],[3,0.4,0,3]]))

print(np.argmax(pred_test_y,axis=1))

print(model.score(x_test, y_test))