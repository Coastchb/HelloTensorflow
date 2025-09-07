import scipy.sparse as sp
import numpy as np
import tensorflow as tf


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

N = 5  # node num
CM = [1, 2, 3, 4, 5]  # np.expand_dims(np.arange(1, N+1), axis=0)  # center node projection
NM = [11, 12, 13, 14, 15]  # np.expand_dims(np.arange(11, N+11), axis=0)  # neighbor node projection
adj_dense = [[1,1,0,0,1],
             [1,0,1,0,0],
             [0,0,0,1,1],
             [1,1,0,0,1],
             [1,0,0,0,1]]
adj_sp = sparse_to_tuple(sp.csr_matrix(adj_dense))

cm_ph = tf.placeholder(tf.int32, (N,))
nm_ph = tf.placeholder(tf.int32, (N,))
adj_ph = tf.sparse_placeholder(tf.int32)  # shape=(N, N) must not set,
# or ValueError: Tensor Tensor("Const:0", shape=(2,), dtype=int64) may not be fed.

adj_sp_t = tf.sparse_transpose(adj_ph)
c = adj_ph.__mul__(cm_ph)
n = adj_sp_t.__mul__(nm_ph)
#ret = c + tf.sparse_transpose(n)
logits = tf.sparse_add(c, tf.sparse_transpose(n))

sess = tf.Session()

#cm_v, nm_v, adj_sp_v, adj_sp_t_v, c_v, n_v, ret_v = sess.run([cm_ph, nm_ph, adj_sp, adj_sp_t, c, n, ret])
ret_v, adj_tensor_value = ret_v = sess.run([logits,adj_ph], feed_dict={cm_ph: CM, nm_ph: NM, adj_ph: adj_sp})

'''
print("cm_v:{0}".format(cm_v))
print("nm_v:{0}".format(nm_v))
print("adj_sp_v:{0}".format(adj_sp_v))
print("adj_sp_t_v:{0}".format(adj_sp_t_v))
print("c_v:{0}".format(c_v))
print("n_v:{0}".format(n_v))
'''
print("ret_v:{0}".format(ret_v))
print("adj_tensor_value:{0}".format(adj_tensor_value))
print(adj_tensor_value.values)
print(adj_sp[0])
print(adj_tensor_value.indices)
assert np.array(adj_sp[0]) == np.array(adj_tensor_value.indices)