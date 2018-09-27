# -*- coding:utf-8 -*- 
# @Time		:2018/9/27 4:24 PM
# @Author	:Coast Cao

import tensorflow as tf
import numpy as np

tf_a = tf.Variable(tf.constant([1,2]))
tf_c = tf.constant([5,6])

np_a = np.array([[3,4,5],[3,5,6]])

print(np_a.shape)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(np.shape(sess.run(tf_a)))

class v_shape_test(tf.test.TestCase):
  def variable_shape_test(self):
    with self.test_session():
      print(np_a.shape)
      self.assertShapeEqual(np_b, tf_a)

class c_shape_test(tf.test.TestCase):
  def constant_shape_test(self):
    with self.test_session():
      self.assertShapeEqual(np_a,tf_c)

class SquareTest(tf.test.TestCase):
  def testSquare(self):
    with self.test_session():
      x = tf.square([2, 3, 4])
      self.assertAllEqual(x.eval(), [4, 9, 16])

tf.test.main()
