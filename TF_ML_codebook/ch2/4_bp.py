# -*- coding:utf-8 -*-
#@Time: 18-9-16 上午9:57
#@Author: Coast Cao

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

# the rael data (training samples)
x_vals = np.random.normal(1, 0.1, 100)
targets = np.repeat(10, 100)

# placeholder and variables
x_data = tf.placeholder(tf.float32)
t_data = tf.placeholder(tf.float32)
A = tf.Variable(1., dtype=tf.float32)

# the model
output = tf.multiply(A, x_data)

# the loss and optimization function
loss = tf.nn.l2_loss(output-t_data)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02,);
train_op = optimizer.minimize(loss)

# do training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for x_val,t_val in zip(x_vals,targets):
        loss_val,_ = sess.run([loss, train_op], feed_dict={x_data:x_val, t_data:t_val})
        print("for %d, loss=%.3f, A=%.3f" % (x_val, loss_val, sess.run(A)))
        print("for %d, loss=%.3f, A=%.3f" % (x_val, loss_val, sess.run(A))) # A is a variable not ramdom value
                                                                            # so its value will not chage
                                                                            # in different run, unless its value
                                                                            # is updated

    print("final A=%.3f" % sess.run(A))

ops.reset_default_graph()

# the rael data (training samples)
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
t_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

# placeholder and variables
x = tf.placeholder(shape=[1], dtype=tf.float32, name="x")
t = tf.placeholder(shape=[1], dtype=tf.float32, name="t")
A = tf.Variable(tf.random_normal(mean=10, shape=[1]), name="A")

# the model
y = tf.add(x, A, name="add")

# loss and optimization function
y_expanded = tf.expand_dims(y, 0)
t_expanded = tf.expand_dims(t, 0)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=t_expanded, logits=y_expanded)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

# do training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4000):
        rand_index = np.random.choice(100);
        sess.run(train_op, feed_dict={x: [x_vals[rand_index]], t: [t_vals[rand_index]]})

    print("In classification task,final A=%.3f" % sess.run(A))

