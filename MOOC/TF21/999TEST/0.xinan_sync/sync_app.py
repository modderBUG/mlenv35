import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import pandas as pd
from PIL import Image
import numpy as np

# 自定义

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


"""
用于预测sync模型的app，输入预测值，输出预测结果
"""

# 要使用模型 先还原模型
checkpoint_path = r"./ckpt/sync/sync.ckpt"

# 模型超参数
lr = 5e-3  # 学习率
d1_size = 15  # 第一层神经元
d2_size = 10  # 第二层神经元
d3_size = 5  # 第三层神经元，为输出神经元


# 模型类,三层神经网络，全部使用sigmoid激活。
class SyncModel(Model):
    def __init__(self):
        super(SyncModel, self).__init__()
        self.d1 = Dense(d1_size, activation='sigmoid')
        self.d2 = Dense(d2_size, activation='sigmoid')
        self.d3 = Dense(d3_size, activation='sigmoid')
        # self.a1 = Activation()

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        y = self.d3(x)
        return y


model = SyncModel()

# 加载模型
model.load_weights(checkpoint_path)

test_x, test_y = read_data_v(
    r'C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\MOOC\TF21\999TEST\0.xinan_sync\my_datasets\val.xlsx')

# 输入预测数据
img_arr = np.array([[3, 0.4, 0, 3], [3, 0.2, 1, 1], [2, 0.9, 1, 5], [4, 0.01, 1, 1], [2, 0.19, 0, 5], [3, 0.4, 0, 3]])  # 1,2,0
x_predict = img_arr

import time
start_time = time.time()
# 预测

result = model.predict(x_predict)
# result2 = model.predict(np.array(test_x))

end_time = time.time()
print(end_time-start_time)
result = tf.argmax(result,1)

print('\n')
tf.print(result)
