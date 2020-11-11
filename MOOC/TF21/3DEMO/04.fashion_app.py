import  tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from PIL import Image
import numpy as np

# 预测图像路径
# img_path = '/root/datas/Keras_train_data/test_data/4.png'
img_path = r'C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\MOOC\TF21\3DEMO\img\9f510fb30f2442a7a303a9cbdc43ad4bd113024a.jpg'

# 要使用模型 先还原模型
ckpt_path = "./ckpt/02/fashion.ckpt"
model  = tf.keras.models.Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.load_weights(ckpt_path)

# preNum = int(input("input the number of test pic:"))


img = Image.open(img_path)
img = img.resize((28,28),Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))



img_arr = img_arr/255.0
x_predict = img_arr[tf.newaxis,...]
result = model.predict(x_predict)
print(result)
# pred = tf.argmax(result,axis=1)
print('\n')
# tf.print(pred)