import tensorflow as tf


a = tf.reshape(tf.range(0, 24), (2, 12))
b = tf.reshape(a, (tf.shape(a)[0], -1, 4))

sess = tf.Session()
aa, bb = sess.run((a, b))
print(aa)
print(bb)
