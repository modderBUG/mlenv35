import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout,LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error


maotai = pd.read_csv('./SH600519.csv')
train_set = maotai.iloc[0:2426-300,2:3].values
test_set = maotai.iloc[2426-300:,2:3].values

# 归一化
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)
test_set = sc.transform(test_set)

x_train=[]
y_train=[]

x_test = []
y_test = []

for i in range(60,len(train_set_scaled)):
    x_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],60,1))

for i in range(60,len(test_set)):
    x_test.append(test_set[i-60:i,0])
    y_test.append(test_set[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    LSTM(80,return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

checkpoint_path = "./ckpt/05/rnn.ckpt"

if os.path.exists(checkpoint_path + '.index'):
    print("load the model")
    model.load_weights(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                 monitor='val_loss')
history = model.fit(x_train, y_train, batch_size=32, epochs=50,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()


# 绘制误差图像

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training And Validation loss')
plt.legend()
plt.show()

predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price  =sc.inverse_transform(test_set[60:])
plt.plot(real_stock_price,color = 'red',label = 'MaoTai Stock Price')
plt.plot(predicted_stock_price,color = 'blue',label = 'Predict MaoTai Stock Price')
plt.legend()
plt.show()

import math
mse = mean_squared_error(predicted_stock_price,real_stock_price)
rmse = math.sqrt(mean_squared_error(predicted_stock_price,real_stock_price))
mae = mean_absolute_error(predicted_stock_price,real_stock_price)

print("均方误差 %.6f"%mse)
print("均方根误差 %.6f"%rmse)
print("平均绝对误差 %.6f"%mae)