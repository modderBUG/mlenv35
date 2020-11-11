import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

# from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Dense,Flatten
# from tensorflow.keras import Model

from keras import Model

"""
LeNet网络搭建示例，以上依赖可以选择，推荐在2.1下运行
"""

# 上节内容 读取数据,这部分代码不改变。
np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class LeNet5(Model):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.c1 = Conv2D(filters=6,kernel_size=(5,5),activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2,2),strides=2)
        self.c2 = Conv2D(filters=16,kernel_size=(5,5),activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120,activation='sigmoid')
        self.f2 = Dense(84,activation='sigmoid')
        self.f3 = Dense(10,activation='softmax')

    def call(self,x):
        x= self.c1(x)
        x= self.p1(x)

        x= self.c2(x)
        x= self.p2(x)

        x= self.flatten(x)
        x= self.f1(x)
        x= self.f2(x)
        y= self.f3(x)
        return y








