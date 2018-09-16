# -*- coding:utf-8 -*- 
#@Time: 18-9-16 上午9:57
#@Author: Coast Cao

import tensorflow as tf
import numpy as np

x_vals = np.random.normal(1, 0.1, 100)
targets = np.repeat(10, 100)

x_data = tf.placeholder(tf.float32)
t_data = tf.placeholder(tf.float32)
A = tf.Variable(1., dtype=tf.float32)
output = tf.multiply(A, x_data)
loss = tf.nn.l2_loss(output-t_data)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02,);
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for x_val,t_val in zip(x_vals,targets):
        loss_val,_ = sess.run([loss, train_op], feed_dict={x_data:x_val, t_data:t_val})
        print("for %d, loss=%.3f, A=%.3f" % (x_val, loss_val, sess.run(A)))
        print("for %d, loss=%.3f, A=%.3f" % (x_val, loss_val, sess.run(A))) # A is variable not ramdom value
                                                                            # so its value will not chage
                                                                            # in different run, unless its value
                                                                            # is updated

    print("final A=%.3f" % sess.run(A))