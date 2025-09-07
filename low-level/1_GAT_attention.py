import tensorflow as tf
import scipy.sparse as sp
import numpy as np

conv1d = tf.layers.conv1d


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


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    print("pre_out:{0}".format(pre_out))
    print("keep_prob:{0}".format(keep_prob))
    return pre_out * (1./keep_prob)

def attn_head_v2(seq, out_sz, adj, bias_nonzero_shape, activation, in_drop=0.0,
                 coef_drop=0.0, residual=False, name=""):
    """
    compared to attn_head(), attn_head_v2() uses sparse tensor to compute node2node attention
    :param seq: (1, N, d)
    :param out_sz: scalar
    :param adj: sparse tensor, shape=(N,N)
    :param bias_nonzero_shape: (1,)
    :param activation: activation function
    :param in_drop: dropout coefficient used on feature matrix
    :param coef_drop: dropout coefficient used on attention weight
    :param residual:
    :param name:
    :return:
    processed feature matrix
    """
    with tf.name_scope(name):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1, activation=tf.nn.leaky_relu)  # (1, N, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1, activation=tf.nn.leaky_relu)  # (1, N, 1)



        adj_t = tf.sparse_transpose(adj)
        c = adj.__mul__(tf.squeeze(f_1))
        n = adj_t.__mul__(tf.squeeze(f_2))

        #return n

        output = tf.sparse_add(c, tf.sparse_transpose(n))  # (N, N)
        return output
        '''
        coefs = tf.sparse_softmax(output)

        if coef_drop != 0.0:
            # dense
            # coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

            # sparse
            print("bias_nonzero_shape:{0}".format(bias_nonzero_shape))
            coefs = sparse_dropout(coefs, 1.0 - coef_drop, bias_nonzero_shape)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # dense
        # vals = tf.matmul(coefs, seq_fts)

        # sparse
        vals = tf.expand_dims(tf.sparse_tensor_dense_matmul(coefs, tf.squeeze(seq_fts, axis=0)), axis=0)

        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation
        '''

N = 5
d = 10
feat = np.arange(50).reshape((N, d)).astype(np.float32)
feat = feat[np.newaxis]

adj_dense = [[1,1,0,0,1],
             [1,0,1,0,0],
             [0,0,0,1,1],
             [1,1,0,0,1],
             [1,0,0,0,1]]
adj_sp = sparse_to_tuple(sp.csr_matrix(adj_dense))


feat_ph = tf.placeholder(dtype=tf.float32, shape=(1, N, d))
adj_ph = tf.sparse_placeholder(dtype=tf.float32, shape=(N, N))
bias_nonzero_shape_ph = tf.placeholder(tf.int32)

ret = attn_head_v2(feat_ph, 8, adj_ph, bias_nonzero_shape_ph, tf.nn.leaky_relu)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ret_v = sess.run([ret], feed_dict={feat_ph: feat, adj_ph: adj_sp, bias_nonzero_shape_ph: adj_sp[1].shape})
print(ret_v)