import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


N = 5
adj_dense = [[1,1,0,0,1],
             [1,0,1,0,0],
             [0,0,0,1,1],
             [1,1,0,0,1],
             [1,0,0,0,1]]
adj_sp = sparse_to_tuple(sp.csr_matrix(adj_dense))


adj_ph = tf.sparse_placeholder(dtype=tf.float32)

#b = tf.sparse_transpose(adj_ph)
ret = tf.sparse_add(adj_ph, adj_ph)

sess = tf.Session()

ret_v = sess.run([ret], feed_dict={adj_ph: adj_sp})
print(ret_v)