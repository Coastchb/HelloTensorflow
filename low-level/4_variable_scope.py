import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

'''
with tf.compat.v1.variable_scope("scope_1") as scope:
    out1 = tf.compat.v1.layers.dense(inputs=np.arange(12).reshape((2, 6)).astype(np.float32), units=1, name='dense1')
print(tf.compat.v1.trainable_variables())
'''

with tf.compat.v1.variable_scope("scopre_2") as scope:
    dense2 = tf.keras.layers.Dense(units=1, name='dense2', trainable=False)
    out2 = dense2(inputs=np.arange(12).reshape((2, 6)).astype(np.float32))
    dense3 = tf.compat.v1.layers.Dense(units=1, name='dense3', trainable=False)
    out3 = dense3(out2)
print('tf.compat.v1.trainable_variables():{0}'.format(tf.compat.v1.trainable_variables()))
print('tf.compat.v1.global_variables():{0}'.format(tf.compat.v1.global_variables()))
for var in tf.compat.v1.trainable_variables():
    tf.print(var, [var], var.name)
    #print(tf.compat.v1.get_variable(var))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    #sess.run()

all_vars = tf.compat.v1.trainable_variables()
var2val = dict([(x.name, x) for x in all_vars])
print(tf.global_variables())

'''
# 创建一个非常简单的神经网络，它有两层
x = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32)
layer1 = tf.compat.v1.layers.dense(x, 5, activation=tf.compat.v1.nn.sigmoid, name=LAYER_1_NAME)
layer2 = tf.compat.v1.layers.dense(layer1, 2, activation=tf.compat.v1.nn.sigmoid, name=LAYER_2_NAME)
print(tf.compat.v1.trainable_variables())

loss = tf.reduce_mean((layer2 - x) ** 2)
optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    x_values = np.random.normal(0, 1, (5000, 2))  # 生成用于输入的随机数
    for step in range(1000):
        _, loss_value = sess.run([optimizer, loss], feed_dict={x: x_values})
        if step % 100 == 0:
            print("step: %d, loss: %f" % (step, loss_value))
'''
