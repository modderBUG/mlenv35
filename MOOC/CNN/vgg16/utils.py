from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示zh
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示正负


def load_image(path):
    fig = plt.figure("Center and Resize")
    img = io.imread(path)
    img = img / 255.0

    print(path)

    ax0 = fig.add_subplot(131)
    ax0.set_xlabel(u'Original Picture')
    ax0.imshow(img)

    short_edge = min(img.shape[:2])
    y = (img.shape[0] - short_edge) // 2
    x = (img.shape[1] - short_edge) // 2
    crop_img = img[y:y + short_edge, x:x + short_edge]

    axl = fig.add_subplot(132)
    axl.set_xlabel(u"Center Picture")
    axl.imshow(crop_img)

    re_img = transform.resize(crop_img, (224, 224))

    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u"Resize Picture")
    ax2.imshow(re_img)

    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready


def percent(value):
    return '%.2f%%' % (value * 100)
