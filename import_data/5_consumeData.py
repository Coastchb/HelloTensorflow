# -*- coding:utf-8 -*- 
# @Time		:2018/11/12 8:12 PM
# @Author	:Coast Cao

import tensorflow as tf

dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()

sess.run(iterator.initializer)

result = tf.add(next_element, next_element)

while True:
    try:
        print(sess.run(result))
    except tf.errors.OutOfRangeError:
        print("reach the end of the dataset")
        break

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4,100])))
dataset3 = tf.data.Dataset.zip((dataset1,dataset2))

iterator3 = dataset3.make_initializable_iterator()

x, (y, z) = iterator3.get_next()

sess.run(iterator3.initializer)

while True:
    try:
        xx, yy, zz = sess.run((x,y,z))
        print(xx)
        print(yy)
        print(zz)
    except tf.errors.OutOfRangeError:
        print("reach the end of dataset")
        break
