'''
输入手写 制作数据集
'''
import os

import tensorflow as tf
import numpy as np
import PIL.Image as Image
import MOOC.MNIST.mnist_forward as mnist_forward
import MOOC.MNIST.mnist_backward as mnist_backward


def restore_model(testPicArr):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x,None)
        preValue = tf.argmax(y,1)

        varible_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        varibles_to_restore = varible_averages.variables_to_restore()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt =  tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue =sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold =50

    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 -im_arr[i][j]
            if(im_arr[i][j]<threshold):
                im_arr[i][j] = 0
            else:
                im_arr[j][j] =255
    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1.0/255.0)

    return img_ready


def application():
    testNum = eval(input("Input the number of test picture:"))
    for i in range(testNum):
        testPic = input("the path of test picture:")
        while os.path.exists(testPic) == False:
            testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("the prediction number is:",preValue)


def main():
    application()


if __name__ == '__main__':
    main()