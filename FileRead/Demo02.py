import tensorflow as tf
import os


class Cifar(object):
    def __init__(self):
        # init
        self.height = 32
        self.width = 32
        self.channels = 3

        # bytes count
        self.image_bytes = self.height*self.width*self.channels
        self.label_bytes = 1
        self.all_bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self,file_list):
        # 1.
        file_queue = tf.train.string_input_producer(file_list)

        # 2.
        reader = tf.FixedLengthRecordReader(self.all_bytes)

        # k - v
        key,value = reader.read(reader)

        # decode
        decoded = tf.decode_raw(value,tf.unit8)
        # slice
        label = tf.slice(decoded,[0],[self.label_bytes])
        image = tf.slice(decoded,[self.label_bytes],[self.image_bytes])

        image_reshaped = tf.reshape(image,shape=[self.channels,self.height,self.width])
        image_transposed = tf.transpose(image_reshaped,[1,2,0])
        image_cast = tf.cast(image_transposed,tf.float32)

        with tf.Session() as f:
            pass

