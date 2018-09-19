# -*- coding:utf-8 -*- 
# @Time		:2018/9/19 上午9:03
# @Author	:Coast Cao

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

iter_num = 100;
batch_size = 20;
tf.set_random_seed(5)
np.random.seed(42)

x_vals = np.random.normal(2, 0.1, 500)
t_vals = np.repeat(0.75, 500)
batch_num = int(500 / batch_size)
samples = [[[[x_vals[j]] for j in range(i*batch_size,(i+1)*batch_size)], [[t_vals[j]] for j in range(i*batch_size,(i+1)*batch_size)]]
           for i in range(batch_num)]

def create_model(activation, xp, yp):
    W = tf.Variable(tf.random_normal(shape=[1,1]),name="weight")
    b = tf.Variable(tf.random_uniform(shape=[1,1]))
    output = activation(tf.add(tf.matmul(W, tf.transpose(xp)), b))
    loss = tf.reduce_mean(tf.square(tf.subtract(tf.transpose(output), yp)))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return W, output, loss, train_op

xp = tf.placeholder(tf.float32, [None,1])
yp = tf.placeholder(tf.float32, [None,1])

with tf.variable_scope("relu") as scope:
  re_W, re_output, re_loss, re_train = create_model(tf.nn.relu, xp, yp)
with tf.variable_scope("sigmoid") as scope:
  sg_W, sg_output, sg_loss, sg_train = create_model(tf.sigmoid, xp, yp)

sess = tf.Session()
summary_writer = tf.summary.FileWriter('tensorboard',
                                      tf.get_default_graph())
if not os.path.exists('tensorboard'):
  os.makedirs('tensorboard')

sess.run(tf.global_variables_initializer())

re_Ws = []
sg_Ws = []
re_outputs = []
sg_outputs = []
re_losses = []
sg_losses = []

for i in range(iter_num):
    np.random.shuffle(samples)
    for j in range(batch_num):
        [[s_w]], s_output, s_loss, _ = sess.run([sg_W, sg_output, sg_loss, sg_train],
                                                feed_dict={xp: samples[j][0], yp: samples[j][1]})
        [[r_w]], r_output, r_loss, _ = sess.run([re_W, re_output, re_loss, re_train],
                                                feed_dict={xp:samples[j][0], yp: samples[j][1]})
        re_Ws.append(r_w)
        sg_Ws.append(s_w)
        re_outputs.append(np.mean(r_output))
        sg_outputs.append(np.mean(s_output))
        re_losses.append(r_loss)
        sg_losses.append(s_loss)
        print("re_w: %.3f" % r_w)
        print("sg_w: %.3f" % s_w)

plt.title("weights")
plt.xlabel("iter")
plt.ylabel("weight value")
plt.plot(re_Ws, 'r--', label="relu weights")
plt.plot(sg_Ws, 'b-', label="sigmoid weights")
plt.legend(loc='upper right')
plt.show()

plt.title("loss")
plt.xlabel("iter")
plt.ylabel("loss value")
plt.plot(re_losses, 'r--', label="relu loss")
plt.plot(sg_losses, 'b-', label="sigmoid loss")
plt.legend(loc='upper right')
plt.show()

plt.title("output")
plt.xlabel("iter")
plt.ylabel("output value")
plt.plot(re_outputs, 'r--', label="relu output")
plt.plot(sg_outputs, 'b-', label="sigmoid output")
plt.ylim([0.0, 2.0] )
plt.legend(loc='upper right')
plt.show()