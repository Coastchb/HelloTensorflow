import tensorflow as tf


def my_func(var):
    return var ** 2 + var * 3

x = tf.Variable(initial_value=[1.0], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch([x])
    y = tf.map_fn(my_func, x, dtype=tf.float32)
    grad = tape.gradient(y, x)
    tf.print("grad:", grad)
