# -*- coding:utf-8 -*- 
#@Time: 18-9-16 下午1:58
#@Author: Coast Cao

import tensorflow as tf
import numpy as np

# define batch size, etc.
batch_size = 20
iter_num = 500

# the rael data (training samples)
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
t_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
batch_num = int(100 / batch_size)
samples = [[[[x_vals[j]] for j in range(i*batch_size,(i+1)*batch_size)], [[t_vals[j]] for j in range(i*batch_size,(i+1)*batch_size)]]
           for i in range(batch_num)]

# placeholder and variables
x = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="x")
t = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="t")
A = tf.Variable(tf.random_normal(mean=10, shape=[1]), name="A")

# the model
y = tf.add(x, A, name="add")

# loss and optimization function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

# initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# do training
for i in range(iter_num):
  np.random.shuffle(samples)
  for j in range(batch_num):
    batch_loss, _ = sess.run([loss, train_op], feed_dict={x: samples[j][0], t: samples[j][1]})
    #print(batch_loss.shape)
    # #print(batch_loss)

print("In classification task,final A=%.3f" % sess.run(A))

# evaluation
test_x_vals = np.concatenate((np.random.normal(-1, 1, 100), np.random.normal(3, 1, 100)))
test_x_vals = np.reshape(test_x_vals, (200,1), -1)
test_t_vals = np.concatenate((np.repeat(0., 100), np.repeat(1., 100)))
test_t_vals = np.reshape(test_t_vals, (200,1), -1)

predict = tf.round(tf.nn.sigmoid(y))
ret = tf.equal(predict, t)
acc = tf.reduce_mean(tf.cast(ret, tf.float32))

print("A=%.3f" % sess.run(A))
train_acc = sess.run(acc, feed_dict={x: np.reshape(x_vals, (100,1), -1), t:np.reshape(t_vals, (100,1), -1)})
test_acc = sess.run(acc, feed_dict={x: test_x_vals, t:test_t_vals});
print("train acc: %.3f%%" % (train_acc*100))
print("test acc: %.2f%%" % (test_acc*100))


