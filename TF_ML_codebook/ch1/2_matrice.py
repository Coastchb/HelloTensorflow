# -*- coding:utf-8 -*- 
# @Time		:2018/9/13
# @Author	:

import tensorflow as tf

mat_a = tf.random_uniform((3,3), minval=1, maxval=3);
mat_b = tf.random_normal((3,3), mean=0.0, stddev=1.0);
mat_d = tf.matrix_diag([1.0,1.0,1.0]);

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer());
    print(sess.run([tf.matmul(mat_a, mat_b), tf.multiply(mat_a, mat_b), mat_a * mat_b])); # tf.multiply is equal to *, performs element-wise multiplication
                                                                                            # if operators are tensor and a scalar value, use +,-,*,/;
                                                                                            # if operators are two tensors, use tf.add,tf.sub,tf.multiply,tf,div
    print(sess.run([tf.div(mat_a, mat_b), mat_a/mat_b]));  # tf.div is equal to /
    print(sess.run(tf.transpose(mat_a)));
    print(sess.run(tf.matrix_determinant(mat_a)));
    print(sess.run(tf.matrix_inverse(mat_a)));
    print(sess.run(tf.cholesky(mat_d)));
    print(sess.run([tf.self_adjoint_eigvals(mat_a), tf.self_adjoint_eig(mat_a)]));

