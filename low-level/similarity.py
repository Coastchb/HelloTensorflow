import tensorflow as tf

a = tf.constant([[1,2],[4,5]],tf.float32)
b = tf.constant([4,5],tf.float32)

sess = tf.Session()

m = tf.reduce_mean(a, axis=0)
a = a - m

print(sess.run(m))
print(sess.run(a))


user_tower = b
#user_tower = tf.reshape(user_tower, (-1, 2))
print(sess.run(user_tower))
print(sess.run(tf.reduce_mean(user_tower, axis=0)))
#user_tower = user_tower - tf.reduce_mean(user_tower, axis=0)
print(sess.run(user_tower))
user_tower = user_tower / tf.norm(user_tower, 2, axis=-1, keepdims=True)
print(sess.run(user_tower))
print(sess.run(tf.nn.l2_normalize(b, axis=-1)))
print(sess.run(b))