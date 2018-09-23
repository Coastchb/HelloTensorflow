# -*- coding:utf-8 -*- 
# @Time		:2018/9/21 上午8:07
# @Author	:Coast

import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt

seed = 0
batch_size = 50
num_iter = 500
num_hidden_layers = 2
num_hidden_unit = 5

tf.set_random_seed(seed)
np.random.seed(seed)

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([[x[3]] for x in iris.data])
len_all = len(x_vals)
train_indices = np.random.choice(len_all, round(len_all*0.8), replace=False)
test_indices = np.array(list(set(range(len_all)) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
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

def create_FC(input, n_input, n_output):
    W = tf.Variable(tf.random_normal(shape=(n_input,n_output)), name="W")
    b = tf.Variable(tf.random_normal(shape=(n_output,), name="b"))
    output = tf.nn.relu(tf.add(tf.matmul(input, W), b))
    return output

cur_output = x_data

for i in range(num_hidden_layers):
    num_input = 3 if i==0 else num_hidden_unit
    num_output = 1 if i==(num_hidden_layers-1) else num_hidden_unit

    with tf.variable_scope("layer%d" % (i+1)) as scope:
        cur_output = create_FC(cur_output, num_input, num_output)



loss = tf.reduce_mean(tf.square(tf.subtract(cur_output,y_data)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_loss, test_loss = [],[]

for i in range(num_iter):
    iter_train_indices = np.random.choice(len(train_indices), batch_size, replace=False)
    iter_loss, _ = sess.run([loss, train_op], feed_dict={x_data:x_vals_train[iter_train_indices],y_data:y_vals_train[iter_train_indices]})
    train_loss.append(iter_loss)

    iter_test_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_data:y_vals_test})
    test_loss.append(iter_test_loss)

plt.title("loss during training")
plt.xlabel("iter")
plt.ylabel("loss value")
plt.plot(train_loss, 'r-', label='train loss')
plt.plot(test_loss, 'b--', label="test loss")
plt.legend(loc='upper right')
plt.show()





