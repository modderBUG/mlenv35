'''
滑动平滑（影子值）
'''

import tensorflow as tf
import numpy as np

def nn01():
    # 1.定义变量及滑动平均类
    # 定义一个32位浮点变量，初始值0.0 这个代码不断更新w1参数 优化w1参数 华东平均做了个w1的影子
    w1 = tf.Variable(0,dtype=tf.float32)

    # 定义num_updates(NN的迭代轮数),初始值位0,不可被优化(训练)，这个参数不训练
    global_step = tf.Variable(0,trainable=False)

    # 实例化滑动平均类 给删减率位0.99 当前轮数global_step
    MOVING_AVERAGE_DECAY = 0.99
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    # ema.apply后的括号里是更新列表 每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值
    # 在实际应用中会使用tf.trainable_variables() 自动将所有带训练的参数汇总为列表
    ema_op =ema.apply(tf.trainable_variables())

    # 2.查看不同迭代中变量取值变化

    with tf.Session() as sess:
        # 初始化
        init_op =tf.global_variables_initializer()
        sess.run(init_op)
        # 用 ema.average(w1) 获取w1滑动平均值（要求运行多个节点，作为列表中的元素列出，卸载sess.run中）
        print(sess.run([w1,ema.average(w1)]))

        # 参数w1赋值为1
        sess.run(tf.assign(w1,1))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

        # 更新step 和 w1 值 模拟出100轮迭代后 参数w1位10
        sess.run(tf.assign(global_step,100))
        sess.run(tf.assign(w1,10))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

        # 每次sess.run会更新一次w1 滑动平均
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

if __name__ == '__main__':
    nn01()
