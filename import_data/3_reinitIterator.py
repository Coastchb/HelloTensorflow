# -*- coding:utf-8 -*- 
# @Time		:2018/11/8 8:54 PM
# @Author	:Coast Cao
import tensorflow as tf

training_dataset =tf.data.Dataset.range(100).map( lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)

next_element = iterator.get_next()

train_op = iterator.make_initializer(training_dataset)
validation_op = iterator.make_initializer(validation_dataset)

sess = tf.Session()
for _ in range(20):
    sess.run(train_op)
    for _ in range(100):
        sess.run(next_element)

    sess.run(validation_op)
    for _ in range(50):
        sess.run(next_element)
