import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout,GlobalMaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,MobileNetV2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""
用于图像预测的app，加载模型输入预测值，输出预测结果。右键直接运行查看
"""

# 要使用模型 先还原模型
# 模型部分
# 模型超参数
checkpoint_path = r"./ckpt/img/img.ckpt"

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

# 模型加载
model.load_weights(checkpoint_path)


# 预处理，准备输入模型
def my_test(path):

    pic_path=path
    X = my_proess_pics(pic_path)
    prediction_label = np.argmax(model.predict(X), axis=1)

    print(prediction_label)


def my_proess_pics(pics_path):

    img_path = os.listdir(pics_path)
    need_pred=[]

    for index in img_path:
        img = image.load_img(os.path.join(pics_path,index), target_size=(224, 224))
        print(os.path.join(pics_path,index))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        need_pred.append(x)
    X = np.concatenate([i for i in need_pred])
    return X

img_arr =r'D:\Data\classify\app' #process_file_my(content_text, word_to_id)
my_test(img_arr)

# x_predict = img_arr
# result = model.predict(x_predict)
#
# # 输出结果
# for i in result:
#     i = i.tolist()
#     print(i.index(max(i)))
# # pred = tf.argmax(result,axis=1)
# print('\n')
# tf.print(result)
