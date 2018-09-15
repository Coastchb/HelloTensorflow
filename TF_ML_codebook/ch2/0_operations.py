# -*- coding:utf-8 -*- 
# @Time		:2018/9/15 下午10:59
# @Author	:Coast Cao

import tensorflow as tf
import numpy as np

x_vals = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.0)
product = tf.multiply(x_data, m_const)

with tf.Session() as sess:
    for x_val in x_vals:
        print(sess.run(product, feed_dict = {x_data: x_val}))