
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
# Sequential 按顺序构成模型
from keras.models import Sequential
# Densen全连接层
from keras.layers import Dense,Activation

from keras.optimizers import SGD

"""
在九天平台运行正常  mnist
"""
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape)

print(y_train.shape)

# (60000,28,28) -> (60000,784)
x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test =x_test.reshape(x_test.shape[0],-1)/255.0

# 换one hot 格式
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model = Sequential([
    Dense(units=10,input_dim=784,bias_initializer='one',activation='softmax')
])

sgd = SGD(lr=0.2)

model.compile(
    optimizer=sgd,
    loss='mse',
    metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=32,epochs=10)

loss,accuracy =model.evaluate(x_test,y_test)

print(loss,accuracy)