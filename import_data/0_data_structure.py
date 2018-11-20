# -*- coding:utf-8 -*- 
# @Time		:2018/11/6 6:34 PM
# @Author	:Coast Cao

import tensorflow as tf


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4]));
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]),tf.random_uniform([4,10], maxval=10, dtype=tf.float32)));
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print(dataset1.output_types)
print(dataset1.output_shapes)
print(dataset2.output_types)
print(dataset2.output_shapes)
print(dataset3.output_types)
print(dataset3.output_shapes)

dataset4 = tf.data.Dataset.from_tensor_slices({"a":tf.random_uniform([4]), "b": tf.random_uniform([4,10], maxval=10, dtype=tf.float32)})
print(dataset4.output_types)
print(dataset4.output_shapes)
print(dataset4.output_classes)