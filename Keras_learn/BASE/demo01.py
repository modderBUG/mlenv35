import keras
import numpy as np
import matplotlib.pyplot as plt
# Sequential 按顺序构成模型
from keras.models import Sequential
# Densen全连接层
from keras.layers import Dense

x_data = np.random.rand(100)    #gen 100 points
noise = np.random.normal(0,0.01,x_data.shape)
y_data  = x_data*0.1+0.2+noise

# plt.scatter(x_data,y_data)
# plt.show()

# 构建一个顺序模型
model = Sequential()
# 在模型中添加一个全连接层
model.add(Dense(units=1,input_dim=1))
# sgd : Stochastic gradient descent，随机梯度下降
# mse : Mean Squared ERROR ，均方误差
model.compile(optimizer='sgd',loss='mse')


for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data,y_data)
    if step % 500 ==0:
        print('cost',cost)

# print weight , bias
W,b=model.layers[0].get_weights()
print(W,b)

#
y_pred = model.predict(x_data)

plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()