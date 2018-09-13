# -*- coding:utf-8 -*- 
# @Time		:2018/9/13
# @Author	:

import tensorflow as tf

mat_a = tf.random_uniform((2,3), minval=1, maxval=3);
mat_b = tf.random_normal((3,4), mean=0.0, stddev=1.0);

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer());
    print(sess.run(tf.matmul(mat_a, mat_b)));
    print(sess.run(tf.transpose(mat_a)));