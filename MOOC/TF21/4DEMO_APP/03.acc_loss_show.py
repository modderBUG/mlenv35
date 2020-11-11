import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
# from keras.layers import Dense,Flatten
# from keras import Model
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import os

# mnist 读入数据
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 断点续训练
ckpt_path = "ckpt/mnist.ckpt"
if os.path.exists(ckpt_path+'.index'):
    print("load the model")
    model.load_weights(ckpt_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)


history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[cp_callback]
)

model.summary()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label = 'Training Accuracy')
plt.plot(val_acc,label = 'Validation Accuracy')
plt.title('Training And Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label = 'Training loss')
plt.plot(val_loss,label = 'Validation loss')
plt.title('Training And Validation loss')
plt.legend()