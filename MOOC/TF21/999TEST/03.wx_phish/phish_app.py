import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from PIL import Image
import numpy as np

"""
用于预测sync模型的app，输入预测值，输出预测结果
"""

# 要使用模型 先还原模型
checkpoint_path =  r"./ckpt/phish/phish.ckpt"

# 模型超参数
lr = 0.01  # 学习率
d1_size = 10  # 第一层神经元
d2_size = 5  # 第二层神经元
d3_size = 2  # 第三层神经元，为输出神经元


# 模型类,三层神经网络，全部使用sigmoid激活。
class PhishModel(Model):
    def __init__(self):
        super(PhishModel, self).__init__()
        self.d1 = Dense(d1_size, activation='sigmoid')
        self.d2 = Dense(d2_size, activation='sigmoid')
        self.d3 = Dense(d3_size, activation='sigmoid')
        # self.a1 = Activation()

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        y = self.d3(x)
        return y


# 创建模型
model = PhishModel()

# 加载模型
model.load_weights(checkpoint_path)

# 输入预测数据
img_arr = np.array([[31,  1, 1, 3, 0, 0, 1,  1], [22, 1, 1,  2,  1,  1,  1,  1], [35,  0,  0, 6, 1,  1,  0,  1]])  # 1,1,0
x_predict = img_arr

# 预测
result = model.predict(x_predict)

# 输出预测结果
for i in result:
    i = i.tolist()
    print(i.index(max(i)))
# pred = tf.argmax(result,axis=1)
print('\n')
tf.print(result)
