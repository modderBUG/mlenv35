# 设损失函数 loss = (w+1)^2，另w初始是常数5.反向传播就是求最优化w，即求最小loss对应的w值
import tensorflow as tf
import numpy as np
# 反向传播和损失函数
def nn01():
    # 定义待优化参数w初始值5
    w = tf.Variable(tf.constant(5,dtype=tf.float32))

    # 定义损失函数loss
    loss = tf.square(w+1)

    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 生成会话 训练40轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(40):
            sess.run(train_step)
            w_val = sess.run(w)
            loss_val = sess.run(loss)
            print("第{}步，loss是：{}\n".format(i, loss_val))

if __name__ == '__main__':
    nn01()
    # nn02()
    # nn03()