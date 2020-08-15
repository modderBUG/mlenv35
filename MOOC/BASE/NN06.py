'''
正则化防止过拟合
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
seed = 2

# 基于seed产生随机数
rdm = np.random.RandomState(seed)


def nn01():
    # 随机数返回300行2列矩阵 表示32组 体积和重量 座位输入数据集
    X = rdm.rand(300, 2)

    # 从X这个300行2列矩阵中 取出一行 判断如果平方和小于2 给Y赋值1 否则Y=0
    # 作为输入数据集标签（正确答案）
    Y_ = [[int(x0*x0 + x1*x1 < 2)] for (x0, x1) in X]

    # 遍历Y中的背个元素 1渎职 red 其余赋值 blue 这样可视化显示 人可以直观区分
    Y_c = [['red' if y else 'blue'] for y in Y_]

    # 对数据集X和标签Y进行shape整理 第一个元素位 -1 表示 随第二个参数计算得到 第二个参数表示多少列 把X整理位n行2列，把Y整理成n行1列
    X = np.vstack(X).reshape(-1,2)
    Y_ = np.vstack(Y_).reshape(-1,1)
    print("aaaaa")

    print(X)
    print(Y_)
    print(Y_c)

    # 用plt.scatter画出数据集X各行中第0列元素和第1列元素的点 即各行的(x0,x1),用各行Y_c对应的值表示颜色
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.show()

    #定义神经网络 输入 参数 输出定义向前传播过程
    def get_weight(shape,regularizer):
        w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

    def get_bias(shape):
        b = tf.Variable(tf.constant(0.01,shape=shape))
        return b

    x= tf.placeholder(tf.float32,shape=(None,2))
    y_= tf.placeholder(tf.float32,shape=(None,1))

    w1 = get_weight([2,1],0.01)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

    w2 = get_weight([11,1],0.01)
    b2 = get_bias([1])
    y = tf.matmul(y1,w2)+b2     # 输出层不激活

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_totle = loss_mse +tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法 ： 不含正则化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 40000
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 300
            end = start +BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
            if i % 2000 ==0:
                loss_mse_v = sess.run(loss_mse,feed_dict={x:X,y_:Y_})
                print("第{}步，loss是：{}\n".format(i, loss_mse_v))

        # xx在-3到3之间以步长位0.01,yy在-3到3之间以步长0.01，生成二维网络坐标点
        xx,yy = np.mgrid[-3:3:0.01,-3:3:0.01]

        # 将xx ,yy 拉直 并合并成一个2列矩阵 得到一个网络坐标点集合
        grid =np.c_[xx.ravel(),yy.ravel()]

        # 将网络坐标点喂给神经网络 probs为输出
        probs = sess.run(y,feed_dict={x:grid})

        # probs 的shape 调成xx的样子
        probs = probs.reshape(xx.shape)
        print("\nw1:\n",sess.run(w1))
        print("\nb1:\n",sess.run(b1))
        print("\nw2:\n",sess.run(w2))
        print("\nb2:\n",sess.run(b2))

    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

    # 定义反向传播方法 包含正则化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_totle)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 40000
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
        print("\nw1:\n", sess.run(w1))
        print("\nb1:\n", sess.run(b1))
        print("\nw2:\n", sess.run(w2))
        print("\nb2:\n", sess.run(b2))
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

if __name__ == '__main__':
    nn01()
