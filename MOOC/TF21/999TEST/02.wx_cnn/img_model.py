import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout,GlobalMaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,MobileNetV2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""
重构图像模型。使用tensorflow2.0。
"""

# 训练集生成 读取指定目录文件。
# 定义路径：
train_dir = r'D:\Data\classify\train'
val_dir = r'D:\Data\classify\test'
# 训练集生成，使用内置函数。
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_dir,   #'/home/zyb/easyimage/data/train',
                                                 target_size=(224,224),
                                                 # color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

val_iter = val_gen.flow_from_directory(val_dir,   #'/home/zyb/easyimage/data/val',
                                        class_mode='categorical',
                                        target_size=(224,224),
                                        batch_size=32)


# 模型部分
# 模型超参数
lr = 1e-4  # 学习率
f1_size ,drop_rate1= 1024,0.5  # 第一层神经元,丢弃率
f2_size,drop_rate2 = 1024,0.3  # 第二层神经元,丢弃率
f3_size,drop_rate3 = 512,0.2  # 第三层神经元,丢弃率
f4_size = 3  # 第四层神经元，为输出神经元。分类


# 模型类,迁移学习，先使用MobileNetV2然后接全局平均池化->全连接1->全连接2->全连接3->全连接4(输出层)。
class ImgModel(Model):
    def __init__(self):
        super(ImgModel, self).__init__()
        self.imgNet = MobileNetV2(weights='imagenet',include_top=False)

        self.g1 = GlobalAveragePooling2D()

        self.f1 = Dense(f1_size, activation='relu')
        self.d1 = Dropout(drop_rate1)

        self.f2 = Dense(f2_size, activation='relu')
        self.d2 = Dropout(drop_rate2)

        self.f3 = Dense(f3_size,activation='relu')
        self.d3 = Dropout(drop_rate3)
        self.f4 = Dense(f4_size, activation='softmax')

    def call(self, x):
        x = self.imgNet(x)
        x = self.g1(x)
        x = self.f1(x)
        x = self.d1(x)

        x = self.f2(x)
        x = self.d2(x)

        x = self.f3(x)
        x = self.d3(x)
        y = self.f4(x)

        return y


# 创建模型
model = ImgModel()



# 编译模型
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr),   # 使用RMSprop算法。
              loss='binary_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# -------------模型保存----------------------
checkpoint_path = "./ckpt/img/img.ckpt"  # 保存模型的路径
if os.path.exists(checkpoint_path + '.index'):
    print("load the model")
    model.load_weights(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 喂入模型
# 获取生成的数据
step_size_train = train_generator.n//train_generator.batch_size
step_size_val = val_iter.n//val_iter.batch_size

history = model.fit_generator(generator=train_generator,
                    validation_data=val_iter,
                    validation_steps=step_size_val,
                    steps_per_epoch=step_size_train,
                    epochs=100,
                    callbacks=[cp_callback])
# 执行模型
model.summary()

# -------------绘制误差图像-----------------------------
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
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
