import tensorflow as tf
import os





with tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
    for i in range(100):
        image = None
        label = None
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(tf.train.BytesList([image])),
            "label": tf.train.Feature(tf.train.Int64List(label))
        }))
        writer.write(example.SerializeToString())

def read_tfrecords(self):
    file_queue = tf.train.string_input_producer(["cifar10.tfrecords"])

    reader = tf.TFRecordReader()
    key,value = reader.read(reader)

    feature = tf.parse_single_example(value,features={
        "image":tf.FixedLenFeature([],tf.string),
        "label":tf.FixedLenFeature([],tf.int64)
    })
    image = feature["image"]
    label = feature["label"]

    # codeing
    image_decoded = tf.decode_raw(image,tf.unit8)
    image_reshaped = tf.reshape(image_decoded,[height,width,channel])

    # deel ...
    image_batch,label_batch = tf.train.batch([image_reshaped,label],100,2,100)

    # session ...


