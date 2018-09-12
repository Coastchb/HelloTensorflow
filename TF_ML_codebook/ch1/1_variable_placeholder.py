# -*- coding:utf-8 -*- 
# @Time		:2018/9/12 8:14 PM
# @Author	:Coast Cao

import tensorflow as tf

# just the declaration of a tf variable with initial value
var_a = tf.Variable(tf.random_normal((2,3)));
pla_b = tf.placeholder(tf.float32, (1,2));
pla_c = tf.placeholder(tf.float32, (1,2));
op = pla_b + pla_c;

var_d = tf.Variable(0, trainable=False);

# this will fail if initialize with global_varialbles_initializer
# we have to firstly initialize var_a and then initialize var_e
#var_e = tf.Variable(tf.zeros_like(var_a));

with tf.Session() as sess:
  # initialize the variable(s): put the variables and relevant
  # methods on the computational graph)
  sess.run(tf.global_variables_initializer());
  print("var_a:");
  print(sess.run(var_a));
  print("op:");
  print(sess.run(op, feed_dict={pla_b: [[3,4]], pla_c: [[5,6]]}));
  while(sess.run(var_d < 3)):
    print("iteration controlled by var_d; var_d=%d" % sess.run(var_d));
    var_d += 1;

