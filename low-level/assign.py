import tensorflow as tf

A = tf.constant([1,2,3])

B = tf.zeros([3*2],tf.float32)


sess = tf.Session()
print(sess.run(B))

aa=tf.Variable(tf.zeros(3, tf.int32))
sess.run(tf.global_variables_initializer())
aa=aa[2].assign(A[1])
print(sess.run(aa))