import tensorflow as tf
import os

def picture_read(file_list):
    '''
    狗图片读取
    :param file_list:
    :return:
    '''

    # 1.构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2.读取与解码
    # 读取阶段
    reader = tf.WholeFileReader()
    # key 文件名,value 编码形式
    key,value = reader.read(file_queue)
    print("{'{0}':'{1}'}".format(key,value))

    #解码阶段
    image = tf.image.decode_jpeg(value)
    print('image:\n',image)

    # 图片形状、类型修改
    image_resized = tf.image.resize_images(image,[200,200])
    print("image_resized:\n",image_resized)

    # 静态形状修改
    image_resized.set_shape(shape=[200,200,3])
    print("image_resized:\n",image_resized)

    # 3.批处理
    image_batch = tf.train.batch([image_resized],batch_size=100,num_threads=1,capacity=100)
    print("image_batch:\n",image_batch)

    # 开启会话
    with tf.Session() as sess:
        # 开启线程
        # 线程协调员
        coord = tf.train.Coordinator()
        treads = tf.train.start_queue_runners(sess,coord)

        key_new,value_new,image_new,image_resized_new =sess.run([key,value,image,image_resized])
        print("key_new\n",key_new)
        print("value_new\n",value_new)
        print("image_new\n",image_new)
        print("image_resized_new\n",image_resized_new)

        # 回收线程
        coord.request_stop()
        coord.join(treads)

    return None

if __name__ == '__main__':

    filename = os.listdir('./dog')

    file_list = [os.path.join('./dog/',file) for file in filename]

    print(file_list)
    picture_read(file_list)


