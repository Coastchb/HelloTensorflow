# -*- coding:utf-8 -*- 
# @Time		:2018/11/8 8:36 PM
# @Author	:Coast Cao
import tensorflow as tf

N = 100
dataset = tf.data.Dataset.range(N)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()

for i in range(N):
    value = sess.run(next_element)
    print(value)
    assert i == value