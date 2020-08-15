import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import MOOC.MODEL.generateds as generateds
import MOOC.MODEL.forward as forward

STEPS = 40000
BATCH_SIZE = 30 # 一次喂入多少组数据到神经网络
LEAENING_RATE_BASE = 0.001 # 最初学习率
LEARNING_RATE_DECAY = 0.999# 学习率衰减
REGULARIZER = 0.01

def backward():
    x= tf.placeholder(tf.float32,shape=(None,2))
    y_= tf.placeholder(tf.float32,shape=(None,1))

    X,Y_,Y_c = generateds.generateds()
    y = forward.forward(x,REGULARIZER)

    global_step = tf.Variable(0,trainable=False)

    learning_rate = tf.train.exponential_decay(LEAENING_RATE_BASE,
                                               global_step,
                                               300/BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)
    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse+tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法 ： 包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
                print("第{}步，loss是：{}\n".format(i, loss_v))

        # xx在-3到3之间以步长位0.01,yy在-3到3之间以步长0.01，生成二维网络坐标点
        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]

        # 将xx ,yy 拉直 并合并成一个2列矩阵 得到一个网络坐标点集合
        grid = np.c_[xx.ravel(), yy.ravel()]

        # 将网络坐标点喂给神经网络 probs为输出
        probs = sess.run(y, feed_dict={x: grid})

        # probs 的shape 调成xx的样子
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


def backward2():
    x= tf.placeholder(tf.float32,shape=(None,2))
    y_= tf.placeholder(tf.float32,shape=(None,1))

    X,Y_,Y_c = generateds.generateds()
    y = forward.forward(x,REGULARIZER)

    global_step = tf.Variable(0,trainable=False)

    learning_rate = tf.train.exponential_decay(LEAENING_RATE_BASE,
                                               global_step,
                                               300/BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)
    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse+tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法 ： 包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        showPlot(sess,y,x,X,Y_c)
        plt.ion()

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
                print("第{}步，loss是：{}\n".format(i, loss_v))

                showPlot(sess,y,x,X,Y_c)
                plt.pause(0.1)

        showPlot(sess, y, x, X, Y_c)
        plt.show()

def showPlot(sess,y,x,X,Y_c):
    # xx在-3到3之间以步长位0.01,yy在-3到3之间以步长0.01，生成二维网络坐标点
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]

    # 将xx ,yy 拉直 并合并成一个2列矩阵 得到一个网络坐标点集合
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 将网络坐标点喂给神经网络 probs为输出
    probs = sess.run(y, feed_dict={x: grid})

    # probs 的shape 调成xx的样子
    probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])

if __name__ == '__main__':
    backward2()