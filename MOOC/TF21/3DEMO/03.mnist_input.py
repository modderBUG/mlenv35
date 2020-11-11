import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
# from keras.layers import Dense
# from keras import Model
import numpy as np
from  sklearn import datasets
import matplotlib.pyplot as plt

# mnist 读入数据
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 可视化训练输入特征的第一个元素
plt.imshow(x_train[0],cmap='gray') #灰度图
plt.show()

# 打印第一个元素
print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(y_train.shape)