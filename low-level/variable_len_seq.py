import tensorflow as tf

x1 = [[1,2,3,4,5],[11,22,33]]
max_len = 10

xp = tf.pad(x1, paddings=0)
sess = tf.Session()
sess.run(xp)
