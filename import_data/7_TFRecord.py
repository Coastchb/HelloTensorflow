# -*- coding:utf-8 -*- 
# @Time		:2018/11/20 3:24 PM
# @Author	:Coast Cao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

### the example features are 1-dimensional, fixed-length

#################################
### create and write TFRecord ###
#################################

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

n_observations = int(1e4)
Feature0 = np.random.choice([False, True], size=n_observations)
Feature1 = [np.random.bytes(10) for i in range(n_observations)]
Feature2 = np.random.randint(-10000, 10000, size=n_observations)
Feature3 = np.random.randn(n_observations)

def create_example(features):
    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                'F0': _int64_feature(features[0]),
                'F1': _bytes_feature(features[1]),
                'F2': _int64_feature(features[2]),
                'F3': _float_feature(features[3])
            }
        )
    )
    return example

# print(create_example([Feature0[0], Feature1[0], Feature2[0], Feature3[0]]))

tfr_file = "tfrecords/1.tfr"

writer = tf.io.TFRecordWriter(tfr_file)

for i in range(n_observations):
    example = create_example([Feature0[i], Feature1[i], Feature2[i], Feature3[i]])
    writer.write(example.SerializeToString())
writer.close()

#################################
###         read TFRecord     ###
#################################

def parse_example(example_proto):
    features = {
        'F0': tf.FixedLenFeature([],tf.int64,default_value=0),      # the key must be F0
        'F1': tf.FixedLenFeature([],tf.string,default_value='0'),
        'F2': tf.FixedLenFeature([],tf.int64,default_value=0),
        'F3': tf.FixedLenFeature([],tf.float32,default_value=0.0)
    }
    return tf.parse_single_example(example_proto, features)

dataset = tf.data.TFRecordDataset(tfr_file)

dataset = dataset.map(parse_example)
dataset = dataset.batch(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()
print(sess.run(next_element))