# 设损失函数 loss = (w+1)^2，另w初始是常数5.反向传播就是求最优化w，即求最小loss对应的w值
# 使用指数衰减的学习率，在迭代初期得到较高下降速度，可以在较小的训练轮数下取得更有收敛度
'''
学习率
'''

import tensorflow as tf
import numpy as np

LEAENING_RATE_BASE = 0.1 # 最初学习率
LEARNING_RATE_DECAY = 0.99# 学习率衰减
LEARNING_RATE_STEP = 1# 为多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本/BATCH_SIZE

# 反向传播和损失函数
def nn01():

    # 运行了几轮BATCH_SIZE的计数器，出示给0，设为不被训练
    global_step = tf.Variable(0,trainable=False)

    # 定义指数下降学习率
    learning_rate = tf.train.exponential_decay(LEAENING_RATE_BASE,
                                               global_step,
                                               LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    # 定义待优化参数 出示给10
    w = tf.Variable(tf.constant(5,dtype=tf.float32))

    # 定义损失函数loss
    loss = tf.square(w+1)

    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 生成会话 训练40轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(40):
            sess.run(train_step)
            learning_rate_val = sess.run(learning_rate)
            global_step_val = sess.run(global_step)
            w_val = sess.run(w)
            loss_val = sess.run(loss)
            print("第{}步，loss是：{}，learning_rate_val:{}，global_step_val:{}\n".format(i, loss_val,learning_rate_val,global_step_val))


if __name__ == '__main__':
    nn01()
    # nn02()
    # nn03()