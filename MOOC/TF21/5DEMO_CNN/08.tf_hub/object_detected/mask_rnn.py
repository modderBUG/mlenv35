import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# def load_image_local(image_path, image_size=(512, 512), preserve_aspect_ratio=True):
#     """Loads and preprocesses images."""
#     # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
#     img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
#     if img.max() > 1.0:
#         img = img / 255.
#     if len(img.shape) == 3:
#         img = tf.stack([img, img, img], axis=-1)
#     img = crop_center(img)
#     img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
#     return img

image_tensor = img = plt.imread("262-1Z9112320032.jpg").astype(np.float32)[np.newaxis, ...]

# Apply image detector on a single image.
detector = hub.load("https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")
detector_output = detector(image_tensor)
class_ids = detector_output["detection_classes"]

