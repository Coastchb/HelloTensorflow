# -*- coding:utf-8 -*- 
# @Time		:2018/9/24 上午8:19
# @Author	:Coast Cao

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed = 0
batch_size = 20
num_iter = 1000
input_dim = 20
output_dim = 1
num_samples = 500

np.random.seed(seed)
tf.set_random_seed(seed)

x_vals = np.random.normal(2.0, 1.0, (num_samples, input_dim))
t_vals = np.repeat(0.75, num_samples).reshape((num_samples,1))

raw_data = tf.placeholder(tf.float32, shape=(None,input_dim))
x_data = tf.expand_dims(raw_data, -1)
t_data = tf.placeholder(tf.float32, shape=(None,1))
fc_w = tf.Variable(tf.random_normal((10,1)))
fc_b = tf.Variable(tf.random_normal(()))
filter_fun = tf.Variable(tf.random_normal(shape=(2,1,1)))

conv_output = tf.nn.conv1d(x_data, filters=filter_fun, stride=2, padding='SAME', data_format='NWC')
#pool_output = tf.nn.max_pool(conv_output, )
fc_input = tf.squeeze(conv_output)
fc_output = tf.nn.sigmoid(tf.add(tf.matmul(fc_input,fc_w),fc_b))
loss = tf.reduce_mean(tf.square(tf.subtract(fc_output, t_data)))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_losses = []
x_test_input = np.random.normal(2.0, 1.0, (100, input_dim))

for i in range(num_iter):
    train_indices = np.random.choice(num_samples, batch_size)
    train_loss, _ = sess.run((loss,train_op), feed_dict={raw_data:x_vals[train_indices],t_data:t_vals[train_indices]})
    train_losses.append(train_loss)

test_pre = sess.run(fc_output,feed_dict={raw_data:x_vals[train_indices]})

plt.title('Loss')
plt.xlabel('iter')
plt.ylabel('loss value')
plt.plot(train_losses, 'b-', label="training loss")
plt.legend(loc='upper right')
plt.show()

plt.hist(test_pre)
plt.show()



