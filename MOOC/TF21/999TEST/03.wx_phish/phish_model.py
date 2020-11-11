import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.io as scio
"""
重构url预测模型。使用tensorflow2.1。
"""

# 读取训练集和测试集的函数,y标签由[0 1][1 0]....
def read_data(path):
    data = scio.loadmat(path)
    return data['X'],data['Y']

# 输入数据
x_train, y_train = read_data(r'./datasets/train_data_716.mat')
x_test, y_test = read_data(r'./datasets/test_data_716.mat')

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

# 编译模型
model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['categorical_accuracy'])

# -------------模型保存----------------------
checkpoint_path = "./ckpt/phish/phish.ckpt"  # 保存模型的路径
if os.path.exists(checkpoint_path + '.index'):
    print("load the model")
    model.load_weights(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 喂入模型
history = model.fit(x_train, y_train, batch_size=32, epochs=2000,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
# 执行模型
model.summary()

# -------------绘制误差图像-----------------------------
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training And Validation Accuracy')
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training And Validation loss')
plt.legend()
plt.show()
# ---------------------end---------------------------------
