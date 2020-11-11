import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam

"""
在九天平台运行正常  mnist cnn 
"""
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

# (60000,28,28) -> (60000,784)
x_train = x_train.reshape(-1,28,28,-1)/255.0
x_test =x_test.reshape(-1,28,28,-1)/255.0

# 换one hot 格式
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# 定义顺序模型
model = Sequential()

# 第一个卷积层
# input_shape 输入平面
# filter 卷积核/滤波器个数
# kernel_size 卷积窗口大小
# strides 步长
# padding same/valid
# activation 激活函数
model.add(Convolution2D(
    input_shape=(28,28,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))

# 池化层 - 1
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
))


# 卷积层-2
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))

# 池化层 - 2
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))

# 把第二个池化层输出扁平化为1维
model.add(Flatten())

# FC - 1
model.add(Dense(1024,activation='relu'))

# Dropout
model.add(Dropout(0.5))

# FC - 2
model.add(Dense(10,activation='softmax'))

# 定义优化器
adam = Adam(lr = 1e-4)

# 定义优化器，loss function,训练过程中计算准确率
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10)

# 评估模型
loss,accuracy=model.evaluate(x_test,y_test)

print('test loss',loss)
print('test accuracy',accuracy)
