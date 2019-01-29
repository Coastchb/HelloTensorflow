# -*- coding:utf-8 -*- 
# @Time		:2018/11/21 3:27 PM
# @Author	:Coast Cao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

### the example features are 2-dimensional, fixed-length

image = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])

example = tf.train.Example(
    features = tf.train.Features(
        feature = {
            # "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))    # OK
            "row1": tf.train.Feature(float_list=tf.train.FloatList(value=image[0])),
            "row2": tf.train.Feature(float_list=tf.train.FloatList(value=image[1]))
        }
    )
)
image_tfr = "tfrecords/image.tfr"

with tf.io.TFRecordWriter(image_tfr) as f:
    f.write(example.SerializeToString())


### Read TFRecord
def parse_example(example_proto):
    example = {
        "row1": tf.FixedLenFeature([3],tf.float32, default_value=[0.0,0.0,0.0]),
        "row2": tf.FixedLenFeature([3], tf.float32, default_value=[0.0, 0.0, 0.0])
    }
    return tf.parse_single_example(example_proto, example)

dataset = tf.data.TFRecordDataset(image_tfr)
print(dataset)
dataset = dataset.map(parse_example)
print(dataset)
iter = dataset.make_one_shot_iterator()
next_element = iter.get_next()
print(next_element)
print(next_element['row1'][1])
input = []
input.append(next_element['row1'])
input.append(next_element['row2'])

W = tf.constant([[1.0],[2.0],[3.0]])
output = tf.matmul(input, W)

sess = tf.Session()
print(sess.run(output))