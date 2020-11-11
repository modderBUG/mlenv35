import keras
import numpy as np
import matplotlib.pyplot as plt
# Sequential 按顺序构成模型
from keras.models import Sequential
# Densen全连接层
from keras.layers import Dense,Activation

from keras.optimizers import SGD

"""
在九天平台运行正常
"""

x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0,0.02,x_data.shape)
y_data  = np.square(x_data)+noise

# plt.scatter(x_data,y_data)
# plt.show()

# 构建一个顺序模型
model = Sequential()
# 在模型中添加一个全连接层
# 1-10-1
model.add(Dense(units=10,input_dim=1,activation='relu'))
# model.add(Activation('tanh'))
model.add(Dense(units=1,activation='relu'))
# model.add(Activation('tanh'))
# sgd : Stochastic gradient descent，随机梯度下降
# mse : Mean Squared ERROR ，均方误差
sgd =SGD(lr=0.3)
model.compile(optimizer=sgd,loss='mse')


for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data,y_data)
    if step % 500 ==0:
        print('cost',cost)

#
y_pred = model.predict(x_data)

plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()