import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

batch_size = 2
neg_num = 3
emb_dim = 4

reps = tf.reshape(tf.range(batch_size * neg_num * emb_dim), (batch_size, neg_num, emb_dim))
random_ind = [[0, 1]] * batch_size


#indx_inc = tf.range(tf.shape(reps)[0]) * tf.shape(reps)[0]

indx_inc = tf.tile(tf.expand_dims(tf.range(tf.shape(reps)[0]) * tf.shape(reps)[1], -1), [1, 2])
#random_neg = [tf.gather(reps[i], random_ind[i]) for i in range(reps.get_shape().as_list()[0])]
#random_neg = tf.gather(reps, random_ind)
random_ind = random_ind + indx_inc
reps = tf.reshape(reps, shape=[batch_size * neg_num, emb_dim])
random_neg = tf.gather(reps, random_ind)

print('reps.get_shape():{0}'.format(reps.get_shape()))
reps = tf.print('reps:', reps)
print('random_ind:{0}'.format(random_ind))
random_neg = tf.print('random_neg:', random_neg)
indx_inc = tf.print('indx_inc:', indx_inc)
random_ind = tf.print('random_ind:', random_ind)

with tf.compat.v1.Session() as sess:
    sess.run([reps, random_neg, indx_inc, random_ind])
