# -*- coding:utf-8 -*-
# @Time		:2018/12/20 8:51 PM
# @Author	:Coast Cao
import tensorflow as tf
# import numerous


class DenseModule:
    def __init__(self, input_, scope_name_, reuse_=tf.compat.v1.AUTO_REUSE):
        self.input = input_
        self.scope_name = scope_name_
        self.reuse = reuse_

    def __call__(self, *args, **kwargs):
        with tf.compat.v1.variable_scope(self.scope_name, reuse=self.reuse):
            # a = tf.constant([[1., 2., 3.]])
            ret = tf.compat.v1.layers.dense(self.input, units=2, use_bias=False)

        return ret


def build_graph():
    reuse = tf.compat.v1.AUTO_REUSE
    input_a = tf.constant([[1., 2., 3.]])
    ret_a = DenseModule(input_a, 'scopeA', reuse_=reuse)()
    input_b = tf.constant([[11., 22., 33.]])
    ret_b = DenseModule(input_b, 'scopeA', reuse_=reuse)()
    input_c = tf.constant([[111., 222., 333.]])
    ret_c = DenseModule(input_c, 'scopeC', reuse_=reuse)()

    print('tf.compat.v1.global_variables() 0: {0}'.format(tf.compat.v1.global_variables()))

    loss = tf.reduce_mean(ret_b - ret_c)
    return loss


with tf.compat.v1.Session() as sess:
    build_graph()
    sess.run(tf.compat.v1.global_variables_initializer())
