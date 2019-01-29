# -*- coding:utf-8 -*- 
# @Time		:2018/12/20 8:51 PM
# @Author	:Coast Cao
import tensorflow as tf

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope('a', reuse=tf.AUTO_REUSE) as s:
        with tf.variable_scope(scope_name) as scope:        # "reuse=tf.AUTO_REUSE" can also set here
       # try:
            v = tf.get_variable(var, shape)
        #except ValueError:
        #    scope.reuse_variables()
        #    v = tf.get_variable(var)
    return v

v1 = get_scope_variable('foo', 'v', 1)
v2 = get_scope_variable('foo', 'v', 1)
assert v1 == v2