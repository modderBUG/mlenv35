import tensorflow as tf

def linear_regrssion():
    '''
    自实现线性回归
    :return:
    '''
    # 1.准备数据
    X = tf.random_normal(shape=[100,1])
    y_true = tf.matmul(X,[[0.8]])+0.7

    # 2.构造模型
    # 定义模型参数用变量
    weight = tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
    y_predict = tf.matmul(X,weight) + bias

    # 3.构造损失函数
    error = tf.reduce_mean(tf.square(y_predict-y_true))


    # 4.优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(error)

    # 显式初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)
        # 查看初始化模型参数之后的值
        print("{}--{}---{}".format(weight.eval(),bias.eval(),error.eval()))
        # 开始训练
        for i in range(100):
            sess.run(optimizer)
            print("{}--{}---{}".format(weight.eval(),bias.eval(),error.eval()))

        tf.summary.FileWriter("./tmp/summary",graph=sess.graph)

    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    linear_regrssion()