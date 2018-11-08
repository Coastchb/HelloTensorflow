# -*- coding:utf-8 -*- 
# @Time		:2018/8/17 11:39 AM
# @Author	:Coast Cao

import tensorflow as tf

with tf.Session() as sess:
  q = tf.FIFOQueue(1000, 'float32')
  counter = tf.Variable(0.0)
  add_op = tf.assign_add(counter, tf.constant(1.0))
  enqueueData_op = q.enqueue(counter)

  qr = tf.train.QueueRunner(q, enqueue_ops = [add_op, enqueueData_op]*2)
  sess.run(tf.global_variables_initializer())
  enqueue_threads = qr.create_threads(sess, start=True)

  for i in range(10):
    print(sess.run(q.dequeue()))

  writer = tf.summary.FileWriter('./graphs/g2', sess.graph)
  writer.close()
# Tensorflow的计算主要在使用CPU/GPU和内存，
# 而数据读取涉及磁盘操作，速度远低于前者操作。
# 因此通常会使用多个线程读取数据，然后使用一个线程消费数据。
# QueueRunner就是来管理这些读写队列的线程的。

# ref:
# https://blog.csdn.net/menghaocheng/article/details/79621482
# https://www.jianshu.com/p/d063804fb272