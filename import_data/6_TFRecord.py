# -*- coding:utf-8 -*- 
# @Time		:2018/11/20 2:50 PM
# @Author	:Coast Cao
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

print(_bytes_feature(tf.compat.as_bytes('test_string')))
print(_bytes_feature(bytes('test_bytes','utf-8')))
print(_bytes_feature(tf.compat.as_bytes(np.array([1,2,3]).tostring())))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))


