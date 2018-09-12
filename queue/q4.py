# -*- coding:utf-8 -*- 
# @Time		:2018/8/17 12:16 PM
# @Author	:Coast Cao

import tensorflow as tf

sess = tf.Session()
q = tf.FIFOQueue(1000, 'float32')
counter = tf.Variable(0.0)
add_op = tf.assign_add(counter, tf.constant(1.0))
enqueueData_op = q.enqueue(counter)

qr = tf.train.QueueRunner(q, enqueue_ops=[add_op, enqueueData_op]*2)
sess.run(tf.global_variables_initializer())

coordinator = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coordinator, start=True)

for i in range(10):
  print(sess.run(q.dequeue()))

coordinator.request_stop()
coordinator.join(enqueue_threads)