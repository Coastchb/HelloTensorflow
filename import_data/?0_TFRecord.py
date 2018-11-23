# -*- coding:utf-8 -*-
# @Time		:2018/11/20 2:50 PM
# @Author	:Coast Cao
import tensorflow as tf
import numpy as np
import array

features = []
labels = []
for i in range(100):
    features.append(i)
    labels.append(i%2)

bytes = array.array('b') #np.array(features).tostring()
bytes.fromlist(features)
print(bytes.tostring())
print(bytes.tostring().tolist())


'''
example = tf.train.Example(
    features = tf.train.Features(
        feature = {
            'feats':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(features).tostring()])),
            'labs':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(labels).tostring()]))
        }
    )
)
with tf.io.TFRecordWriter("tfrecords/0.tfr") as writer:
    writer.write(example.SerializeToString())

dataset = tf.data.TFRecordDataset("tfrecords/0.tfr")
dataset = dataset.batch(20)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
ne = sess.run(next_element)
print(np.array(ne))
#print(ne[1])
'''