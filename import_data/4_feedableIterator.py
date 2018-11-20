# -*- coding:utf-8 -*- 
# @Time		:2018/11/12 7:38 PM
# @Author	:Coast Cao

import tensorflow as tf

training_dataset =tf.data.Dataset.range(100).map( lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

handle = tf.placeholder(tf.string, [])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)

next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

sess = tf.Session()
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

while True:
    for _ in range(200):
        sess.run(next_element, feed_dict={handle:training_handle})

    sess.run(validation_iterator.initializer)
    for _ in range(50):
        sess.run(next_element, feed_dict={handle:validation_handle})
