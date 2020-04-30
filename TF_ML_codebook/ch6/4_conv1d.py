import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # batch_size=1; length=5; feature_dim=2
    inputs = tf.constant(np.arange(0, 10, dtype=np.float32), shape=[1, 5, 2])
    w = np.array([[1,1], [2,2]], dtype=np.float32).reshape([2, 2, 1])
    # filter width, filter channels and out channels(number of kernels, output dim)
    cov1 = tf.nn.conv1d(inputs, w, stride=1, padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(cov1)
        print(out)

    # batch_size=1; length=5; feature_dim=2
    inputs = tf.constant(np.arange(0, 10, dtype=np.float32), shape=[1, 5, 2])
    # output_dim = 10
    cov1 = tf.layers.conv1d(inputs, filters=10, kernel_size=2, strides=1, padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(cov1)
        print(out)
