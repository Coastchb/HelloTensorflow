# -*- coding:utf-8 -*- 
# @Time		:2018/9/21 上午8:07
# @Author	:Coast

import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt

seed = 2
batch_size = 50
num_iter = 500

tf.set_random_seed(seed)
np.random.seed(seed)

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([[x[3]] for x in iris.data])
len_all = len(x_vals)
train_indices = np.random.choice(len_all, round(len_all*0.8), replace=False)
test_indices = np.array(list(set(range(len_all)) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

x_vals_train = normalize_cols(x_vals_train)
x_vals_test = normalize_cols(x_vals_test)

x_data = tf.placeholder(dtype=tf.float32,shape=(None,3),name="input")
y_data = tf.placeholder(dtype=tf.float32,shape=(None,1),name="output")

W1 = tf.Variable(tf.random_normal(mean=0,stddev=1,shape=(1,3)), name="W1")
b1 = tf.Variable(tf.random_normal(mean=0,stddev=1,shape=(1,)), name="b1")

output = tf.nn.relu(tf.add(tf.matmul(W1,tf.transpose(x_data)), b1))
loss = tf.reduce_mean(tf.square(tf.subtract(tf.transpose(output),y_data)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(num_iter):
    iter_train_indices = np.random.choice(len(train_indices), batch_size, replace=False)
    sess.run(train_op, feed_dict={x_data:x_vals_train[iter_train_indices],y_data:y_vals_train[iter_train_indices]})





