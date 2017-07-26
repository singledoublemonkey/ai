Tensorflow提供了一种统一的数据封装格式，即TFRecord，可以统一不同的源数据格式

TFRecord中的数据是通过tf.train.Example Protocol Buffer格式存储

TFRecord中包含了属性名到值的字典.

属性名是字符串，属性值可以是
- 字符串（BytesList）
- 实数列表（FloatList）
- 整数列表（IntList）


```
# -*- coding: utf-8 -*-
'''
    将手写数字转换为TFRecord
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("../mnist/", dtype=tf.uint8, one_hot=True)

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = "./model/output.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)

for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(tf.train.Features({
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())

writer.close()
```
