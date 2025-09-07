import tensorflow as tf
import numpy as np

sess = tf.Session()

a1 = tf.constant([1,2])
b1 = tf.constant(3)
print(sess.run(a1 * b1))

a2 = tf.constant([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
b2 = tf.constant([[1,1,1],[2,2,2],[3,3,3]])
#print(sess.run(tf.matmul(a2, b2)))

b3 = tf.constant([[1,1,1],[2,2,2]])
print(sess.run(a2 + b3))
print(sess.run(tf.add(a2, b3)))
print(sess.run(a2 + tf.expand_dims(b3,0)))
print(sess.run(tf.expand_dims(b3,1)))
print(sess.run(a2 + tf.expand_dims(b3,1)))

print(sess.run(tf.nn.tanh([[1.,2.,3.],[4.,5.,6.]])))
print(sess.run(tf.nn.tanh([[11.,22.,33.],[44.,55.,66.]])))
print(sess.run(tf.nn.tanh([[[1.,2.,3.],[4,5,6]],[[11,22,33],[44,55,66]]])))

b4 = tf.constant([1,2,3])

ab4 = tf.squeeze(tf.matmul(tf.expand_dims(b4,0),tf.transpose(a2,(0,2,1))))
print(sess.run(ab4))

ab5 = tf.transpose(tf.nn.softmax(tf.transpose(tf.cast(ab4, tf.float32), (1, 0))), (1, 0))
print(sess.run(ab5))

print(sess.run(a2 + b4))

print(sess.run(a2 * tf.expand_dims(ab4,-1)))
print(sess.run(tf.reduce_sum(a2 * tf.expand_dims(ab4,-1), 0)))

print(sess.run(tf.tensordot(b2, b2, [1,1])))
print(sess.run(tf.reduce_sum(tf.multiply(b2, b2), -1)))


behavior_sequence_len = 20
neg_sampling_num = 8
ns_idx = np.random.choice(np.arange(behavior_sequence_len-1), neg_sampling_num, replace=False)
print(ns_idx)

print(sess.run(tf.reduce_sum(tf.multiply(a2,b3),-1)))
print(sess.run(tf.log(tf.cast(tf.reduce_sum(tf.multiply(a2,b3),-1), tf.float32))))