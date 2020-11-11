import  tensorflow as tf
from  keras.layers import Dense,Flatten
from keras import Model
from PIL import Image
import numpy as np

# 预测图像路径
img_path = ''

# 要使用模型 先还原模型
ckpt_path = "ckpt/mnist.ckpt"
model  = tf.keras.models.Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.load_weights(ckpt_path)

# preNum = int(input("input the number of test pic:"))


img = Image.open(img_path)
img = img.resize((28,28),Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

for i in range(28):
    for j in range(28):
        if img_arr[i][j]<200:
            img_arr[i][j]=255
        else:
            img_arr[i][j] = 0

img_arr = img_arr/255.0
x_predict = img_arr[tf.newaxis,...]
result = model.predict(x_predict)
pred = tf.argmax(result,axis=1)
print('\n')
tf.print(pred)