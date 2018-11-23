# -*- coding:utf-8 -*- 
# @Time		:2018/11/21 4:16 PM
# @Author	:Coast Cao


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from functools import partial

### the example features are sequences
# Question 1

utterance1 = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
utterance2 = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])

def create_sequence_example(features):
    example = tf.train.SequenceExample(
        feature_lists = tf.train.FeatureLists(
            feature_list = {
                "feat": tf.train.FeatureList(
                    feature = [tf.train.Feature(
                        float_list = tf.train.FloatList(value=value)
                    ) for value in features]
                )
            }
        )
    )
    return example

utt_tfr = "tfrecords/utt.tfr"
with tf.io.TFRecordWriter(utt_tfr) as f:
    f.write(create_sequence_example(utterance1).SerializeToString())
    f.write(create_sequence_example(utterance2).SerializeToString())


### Read TFRecord
def parse_example(example_proto):
    #print(type(example_proto))
    example = {
        "feat": tf.FixedLenSequenceFeature(shape=[3], dtype=tf.float32),
    }
    return tf.parse_single_sequence_example(example_proto, sequence_features=example)

dataset = tf.data.TFRecordDataset(utt_tfr)
datasets = dataset.map(parse_example)
dataset = tf.data.Dataset.zip(datasets)
# Question 1
# dataset = dataset.batch(2) # Error.  Then how to get batch?
iter = dataset.make_one_shot_iterator()
next_element = iter.get_next()
sess = tf.Session()

print(sess.run(next_element))

W = tf.ones(shape=[3,1])
output = tf.matmul(next_element[1]['feat'],W)
print(sess.run(output))
