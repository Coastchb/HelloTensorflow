# -*- coding:utf-8 -*- 
# @Time		:2018/11/8 8:46 PM
# @Author	:Coast Cao
import tensorflow as tf

max_value = tf.placeholder(dtype=tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer, feed_dict = {max_value : 10})
for i in range(10):
    value = sess.run(next_element)
    print(value)
    assert value == i

sess.run(iterator.initializer, feed_dict = {max_value : 100})
for i in range(10):
    value = sess.run(next_element)
    assert value == i