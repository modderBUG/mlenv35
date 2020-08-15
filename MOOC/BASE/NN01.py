# 两层简单神经网络（全连接）
import tensorflow as tf



def nn01():
    # 定义输入和参数
    x = tf.constant([[0.7,0.5]])
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    # 定义向前传播过程
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    # 用会话计算结果
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('y in 01 is:\n',sess.run(y))

def nn02():
    # 定义输入和参数
    # x = tf.constant([[0.7,0.5]])
    # 用 placeholder 实现输入定义 ,在sess.run中喂一组数据
    x = tf.placeholder(tf.float32,shape=(1,2))
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    # 定义向前传播过程
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    # 用会话计算结果
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('y in 02 is:\n',sess.run(y,feed_dict={x:[[0.7,0.5]]}))

def nn03():
    # 定义输入和参数
    # x = tf.constant([[0.7,0.5]])
    # 用 placeholder 实现输入定义 ,在sess.run中喂一组数据
    x = tf.placeholder(tf.float32,shape=(None,2))
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    # 定义向前传播过程
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    # 用会话计算结果
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('y in 03 is:\n',sess.run(y,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
        print("w1:\n",w1)
        print("w2:\n",w2)

if __name__ == '__main__':
    nn01()
    nn02()
    nn03()
