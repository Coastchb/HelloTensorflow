import tensorflow as tf

a  = tf.reshape(tf.range(10), shape=[2,5])

print(a)
b = tf.slice(a, [0, 1], [-1, -1])
print(b)

print(tf.compat.v1.log(tf.cast(b, tf.float32)))
print(tf.cast(b, tf.float32) * tf.compat.v1.log(tf.cast(b, tf.float32)))
print(tf.matmul(tf.cast(b, tf.float32), tf.transpose(tf.compat.v1.log(tf.cast(b, tf.float32)))))

c = tf.reduce_sum(tf.cast(b, tf.float32) * tf.compat.v1.log(tf.cast(b, tf.float32)), axis=-1)
print(c)