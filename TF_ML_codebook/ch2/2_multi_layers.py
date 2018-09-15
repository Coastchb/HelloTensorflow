# -*- coding:utf-8 -*- 
# @Time		:2018/9/15 下午11:32
# @Author	:Coast Cao

import tensorflow as tf
import numpy as np

x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)

my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')

def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix);
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    tmp1 = tf.matmul(A, input_matrix_squeezed)
    tmp2 = tf.add(tmp1, b)
    return tf.sigmoid(tmp2)

with tf.name_scope("Custom_layer") as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

with tf.Session() as sess:
    print(sess.run(custom_layer1, feed_dict={x_data:x_val}))

