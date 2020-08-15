# 导入模块，生成模拟数据集
# 反向传播和损失函数
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 23455

# 基于seed产生随机数
rdm = np.random.RandomState(seed)

# 随机数返回32行2列矩阵 表示32组 体积和重量 座位输入数据集
X = rdm.rand(32,2)

# 均方误差
def nn01():
    # 从X这个32行2列矩阵中 取出一行 判断如果和小于1 给Y赋值1 否则Y=0
    # 作为输入数据集标签（正确答案）
    Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
    print("X:\n{}\n     Y:\n{}\n".format(X, Y))

    # 1.定义神经网络输入、参数、输出，定义向前传播过程。
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    # 定义输入和参数
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    # 定义向前传播过程
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    # 2.定义损失函数及反向传播方法
    loss = tf.reduce_mean(tf.square(y-y_))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(0.001).minimize(loss)
    # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


    # 3.生成会话 训练steps轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 输出目前（未经测试）的参数取值
        print("w1:\n{}\n   w2:\n{}\n".format(sess.run(w1),sess.run(w2)))

        # 训练模型
        STEPS = 3000
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 32
            end = start +BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
            if i%500 == 0:
                total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
                print("第{}步，loss是：{}\n".format(i,total_loss))

        print('w1 is:\n',sess.run(w1))

# loss_mse Mean Squared Error 均方误差
# 预测多或预测少影响一样
def nn02():

    Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

    # 1.定义神经网络输入、参数、输出，定义向前传播过程。
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    # 定义输入和参数
    w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
    y = tf.matmul(x,w1)

    # 2.定义损失函数及反向传播方法
    loss_mse = tf.reduce_mean(tf.square(y_-y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
    # train_step = tf.train.MomentumOptimizer(0.001).minimize(loss)
    # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


    # 3.生成会话 训练steps轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 输出目前（未经测试）的参数取值
        # print("w1:\n{}\n   w2:\n{}\n".format(sess.run(w1),sess.run(w2)))

        # 训练模型
        STEPS = 30000
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 32
            end = start +BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
            if i%500 == 0:
                # total_loss = sess.run(loss_mse,feed_dict={x:X,y_:Y_})
                # print("第{}步，loss是：{}\n".format(i,total_loss))
                print("第{}步，w1是：{}\n".format(i,sess.run(w1)))

        print('w1 is:\n',sess.run(w1))

# 自定义损失函数
# 分段函数 预测y多了损失成本 否则损失利润了
def nn03():
    COST = 9
    PROFIT = 1

    # 从X这个32行2列矩阵中 取出一行 判断如果和小于1 给Y赋值1 否则Y=0
    # 作为输入数据集标签（正确答案）
    Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
    print("X:\n{}\n     Y:\n{}\n".format(X, Y))

    # 1.定义神经网络输入、参数、输出，定义向前传播过程。
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    # 定义输入和参数
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # 定义向前传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 2.定义损失函数及反向传播方法
    loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(0.001).minimize(loss)
    # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 3.生成会话 训练steps轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 输出目前（未经测试）的参数取值
        print("w1:\n{}\n   w2:\n{}\n".format(sess.run(w1), sess.run(w2)))

        # 训练模型
        STEPS = 3000
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 32
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 500 == 0:
                total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
                print("第{}步，loss是：{}\n".format(i, total_loss))

        print('w1 is:\n', sess.run(w1))

if __name__ == '__main__':
    # nn01()
    nn02()
    # nn03()