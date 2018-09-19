# -*- coding:utf-8 -*- 
# @Time		:2018/9/19 上午9:03
# @Author	:Coast Cao

import tensorflow as tf
import numpy as np

iter_num = 100;
batch_size = 20;

x_vals = np.random.normal(2, 0.1, 500)
t_vals = np.repeat(0.75, 500)
batch_num = int(100 / batch_size)
samples = [[[[x_vals[j]] for j in range(i*batch_size,(i+1)*batch_size)], [[t_vals[j]] for j in range(i*batch_size,(i+1)*batch_size)]]
           for i in range(batch_num)]

def create_model(activation, xp, yp):
    W = tf.Variable(tf.random_normal(shape=[1,1],mean=10., stddev=0.1, dtype=tf.float32))
    output = activation(tf.matmul(W, tf.transpose(xp)))
    loss = tf.reduce_mean(tf.square(tf.subtract(tf.transpose(output), yp)))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return loss, train_op

xp = tf.placeholder(tf.float32, [None,1])
yp = tf.placeholder(tf.float32, [None,1])

re_loss, re_train = create_model(tf.nn.relu, xp, yp)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(iter_num):
    np.random.shuffle(samples)
    for j in range(batch_num):
        sess.run(re_train, feed_dict={xp:samples[j][0], yp: samples[j][1]})


