import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

plt.imshow(x_train[0])
plt.show()

print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(y_train.shape)