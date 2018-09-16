# -*- coding:utf-8 -*- 
# @Time		:2018/9/16 上午8:30
# @Author	:Coast Cao

import tensorflow as tf
import matplotlib.pyplot as plt

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

l2_loss_val = tf.square(target - x_vals)
l1_loss_val = tf.abs(target - x_vals)

with tf.Session() as sess:
    x_array = sess.run(x_vals)
    l2_loss_out = sess.run(l2_loss_val)
    l1_loss_out = sess.run(l1_loss_val)

plt.plot(x_array, l2_loss_out, 'b-', label="L2 Loss")
plt.plot(x_array, l1_loss_out, 'r--', label="L1 Loss")

plt.ylim(-0.2, 0.4)
plt.legend(loc="lower right", prop={'size':11})

plt.show()