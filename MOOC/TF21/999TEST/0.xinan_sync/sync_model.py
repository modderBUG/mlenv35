import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""
重构综合预测模型。使用tensorflow2.0。
"""


# 数据预处理，读入数据-训练集
def read_data(data_dir):
    data1 = pd.read_excel(data_dir, header=None)
    print("****************************")
    X1 = np.array(data1.loc[0:9999, 0:3])
    label = data1.loc[0:9999, 4]
    Y1 = []
    for i in label:
        temp = [0, 0, 0, 0, 0]
        temp[i] = 1
        Y1.append(temp)
    Y1 = np.array(Y1)
    return X1, Y1


# 数据预处理，读入数据-测试集
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


# 输入数据
x_train, y_train = read_data(r'./my_datasets/train.xlsx')
x_test, y_test = read_data_v(r'./my_datasets/val.xlsx')

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


# 创建模型
model = SyncModel()

# 编译模型
model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
              loss = tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# -------------模型保存----------------------
checkpoint_path = "./ckpt/sync/sync.ckpt"  # 保存模型的路径
if os.path.exists(checkpoint_path + '.index'):
    print("load the model")
    model.load_weights(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 喂入模型
history = model.fit(x_train, y_train, batch_size=64, epochs=500,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
# 执行模型
model.summary()

# -------------绘制误差图像-----------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
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
