import numpy as np
import keras.datasets.mnist as mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Input
from keras.models import Model
from keras.optimizers import Adam

"""
在九天平台运行正常  mnist 函数模型
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


# 定义序贯模型
model = Sequential()

# 第一个卷积层
# input_shape 输入平面
# filter 卷积核/滤波器个数
# kernel_size 卷积窗口大小
# strides 步长
# padding same/valid
# activation 激活函数
model.add(Conv2D(
    input_shape=(28,28,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))

# 第一个池化层
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
))

# 第二个卷积层
model.add(Conv2D(64,5,strides=1,padding='same',activation='relu'))

# 第二池化层
model.add(MaxPooling2D(2,2,'same'))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

# 函数式模型
inputs = Input(shape=(28,28,1))
x= Conv2D(filters=32,kernel_size=5,padding='same',activation='relu')(inputs)
x = MaxPooling2D(pool_size=2)(x)
x=Conv2D(filters=64,kernel_size=5,padding='same',activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Flatten()(x)
x=Dense(1024,activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)
model = Model(inputs=inputs,outputs =predictions)


# train
adam = Adam(lr =1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy')
model.fit(x_train,y_train,batch_size=64,epochs=10)

loss,accuracy = model.evaluate(x_test,y_test)

print('test loss',loss)
print('test accuracy',accuracy)