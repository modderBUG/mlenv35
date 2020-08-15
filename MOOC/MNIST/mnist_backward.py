import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data ,mnist

import MOOC.MNIST.mnist_forward as mnist_forward
import os


BATCH_SIZE = 200 # 一次喂入多少组数据到神经网络
LEAENING_RATE_BASE = 0.001 # 最初学习率
LEARNING_RATE_DECAY = 0.999# 学习率衰减
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'

def backward(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])

    y = mnist_forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))


    learning_rate = tf.train.exponential_decay(LEAENING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name="train")

    saver  = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,stop = sess.run([train_op,loss,global_step],feed_dict={x:xs,y:ys})

            if i % 1000 ==0:
                print('步长:{},误差:{}'.format(i,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    pass

if __name__ == '__main__':
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)