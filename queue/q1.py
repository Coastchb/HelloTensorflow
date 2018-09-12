# -*- coding:utf-8 -*-
# @Time		:2018/8/17 11:39 AM
# @Author	:Coast Cao

import tensorflow as tf

def print_q(sess,q):
  qlen = sess.run(q.size())
  for i in range(qlen):
    print(sess.run(q.dequeue()))

with tf.Session() as sess:
  q = tf.FIFOQueue(4, 'float')
  init = q.enqueue_many(([1.0,2.0,3.0],))
      # Error: init = q.enqueue_many([1.0,2.0,3.0],)
      # Error: init = q.enqueue_many([1.0,2.0,3.0])

  init3 = q.enqueue([4.0])   # OK: init3 = q.enqueue(4.0)
  init2 = q.dequeue()

  sess.run(init)
  sess.run(init2)
  sess.run(init3)

  #print_q(sess,q);


  init4 = q.enqueue(5.0)
  init5 = q.enqueue(6.0)
  init6 = q.enqueue(7.0)
  init7 = q.dequeue();
  init8 = q.dequeue()
  init9 = q.dequeue()

  sess.run(init4)

  writer = tf.summary.FileWriter('./graphs/g1', sess.graph)

  # will be blocked if run init5 or init6
  # 如果一次性入列超过Queue Size的数据，
  # enqueue操作会卡住，直到有数据（被其他线程）从队列取出。
  # 对一个已经取空的队列使用dequeue操作也会卡住，
  # 直到有新的数据（从其他线程）写入。
  #sess.run(init5)
  #sess.run(init6)
  #sess.run(init7)
 # sess.run(init8)
 # sess.run(init9)

  print("After more operation:")
  print_q(sess, q)
  writer.close()
# ref:
# https://blog.csdn.net/menghaocheng/article/details/79621482
# https://www.jianshu.com/p/d063804fb272