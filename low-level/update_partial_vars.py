import tensorflow as tf

x = tf.cast(tf.reshape(tf.range(1,3), [2, 1]),tf.float32)

with tf.variable_scope("first_varible"):
    w1 = tf.Variable([[1.0, 2.0]], name="w1")
with tf.variable_scope("second_varible"):
    w2 = tf.Variable([[2.0, 3.0]], name="w2")
y1 = tf.matmul(w1, x) + tf.matmul(w2, x)
y2 = 3 * tf.matmul(w1, x)
y3 = 2 * tf.matmul(w2, x)

print(tf.all_variables())
print(tf.trainable_variables())
all_vars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0, beta2=0)

y1_gradients = optimizer.compute_gradients(y1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="first_varible"))
y2_gradients = optimizer.compute_gradients(y2, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="first_varible"))
y3_gradients = optimizer.compute_gradients(y3, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="second_varible"))

all_gradients = y1_gradients + y2_gradients + y3_gradients
optimize = optimizer.apply_gradients(all_gradients)

print("graiednets:")
print(y1_gradients)
print(y2_gradients)
print(y3_gradients)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("before updating:")
    print(sess.run(x))
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(y1_gradients))
    print(sess.run(y2_gradients))
    print(sess.run(y3_gradients))
    print(sess.run(all_gradients))

    print("after updating:")
    sess.run(optimize)
    print(sess.run(w1))
    print(sess.run(w2))