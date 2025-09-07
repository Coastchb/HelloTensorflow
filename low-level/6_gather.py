import numpy as np
import tensorflow as tf

a = tf.reshape(tf.range(4 * 13 * 8), shape=(4, 13, 8))
print("a:{0}".format(a))

b = np.array([[0, 1, 2, 3, 5, 6, 7, 9],
              [0, 1, 2, 3, 4, 6, 7, 8],
              [0, 1, 2, 3, 4, 5, 7, 8],
              [0, 1, 2, 3, 4, 5, 6, 8]])
bb = np.expand_dims(b, axis=1)

print('bb:{0}'.format(bb))

c = tf.gather(a, b, axis=1)
print('c:{0}, c.shape:{1}'.format(c, tf.shape(c)))

cc = [tf.gather(a[i], b[i]) for i in range(4)]
print('cc:{0}, cc.shape:{1}'.format(cc, np.shape(np.array(cc))))

num_negatives = 8
batch_size = 4
print('num_negatives:{0}, batch_size:{1}'.format(num_negatives, batch_size))
num_negatives_safe = tf.cond(
    num_negatives > 2 * (batch_size - 1),
    lambda: 2 * (batch_size - 1),
    lambda: num_negatives)


seq_len = tf.constant([1, 3, 2, 4], dtype=tf.int32)
seq_len = tf.compat.v1.to_int32(tf.squeeze(seq_len)-1)
seq = tf.convert_to_tensor([tf.gather(a[i], seq_len[i]) for i in range(tf.shape(a)[0])])
print('seq:{0}'.format(seq))


seq_len = tf.constant([1, 3, 2, 4], dtype=tf.int32)
aa = tf.reshape(a, shape=[4*13, 8])
to_gather_indx = seq_len + tf.range(tf.shape(a)[0]) * tf.shape(a)[1]
print('to_gather_indx:{0}'.format(to_gather_indx))
seq = tf.gather(aa, to_gather_indx - 1 , axis=0)
print('seq:{0}'.format(seq))


def fetch_last_emb(seq_emb, seq_lens):
    #print('in fetch_last_emb, seq_emb:{0}, seq_lens:{1}'.format(seq_emb, seq_lens))
    batch_size = tf.shape(seq_emb)[0]

    to_gather_indx = tf.compat.v1.to_int32(tf.squeeze(seq_lens)) + tf.range(tf.shape(seq_lens)[0]) * \
                     tf.shape(seq_emb)[1]
    seq_emb = tf.reshape(seq_emb, shape=[-1, tf.shape(seq_emb)[-1]])
    seq = tf.gather(seq_emb, to_gather_indx - 1, axis=0)
    seq = tf.reshape(seq, shape=[batch_size, -1])
    """
    seq = seq_emb[:, -1, :]
    """

    print('in fetch_last_emb, seq:{0}'.format(seq))

    return seq


ret = fetch_last_emb(a, seq_len)
print('ret:{0}'.format(ret))
