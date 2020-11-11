import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN,Embedding
import matplotlib.pyplot as plt
import os

input_word = 'abcde'
w_to_id = {'a': 0,
           'b': 1,
           'c': 2,
           'd': 3,
           'e': 4}


x_train = [w_to_id['a'],
           w_to_id['b'],
           w_to_id['c'],
           w_to_id['d'],
           w_to_id['e']]

y_train = [w_to_id['b'],
           w_to_id['c'],
           w_to_id['d'],
           w_to_id['e'],
           w_to_id['a']]


np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train,(len(x_train),1,))
y_train = np.array(y_train)


model = tf.keras.Sequential([
    Embedding(5,2),
    SimpleRNN(3),
    Dense(5,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path = "./ckpt/03/1pre1.ckpt"

if os.path.exists(checkpoint_path + '.index'):
    print("load the model")
    model.load_weights(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                 monitor='loss')
history = model.fit(x_train, y_train, batch_size=32, epochs=100,
                    callbacks=[cp_callback])

model.summary()

# 绘制误差图像
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training And Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.title('Training And Validation loss')
plt.legend()

preNum = int(input("input Num?:"))

for i in range(preNum):
    alphabet1 = input("input abcde:")
    alphabet = [id_to_onehot[w_to_id[alphabet1]]]

    alphabet = np.reshape(alphabet,(1,1,5))
    result = model.predict([alphabet])
    pred = tf.argmax(result,axis=1)
    pred=int(pred)
    tf.print(alphabet1+'->'+input_word[pred])