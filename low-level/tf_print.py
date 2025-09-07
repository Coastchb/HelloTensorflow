import tensorflow as tf
import numpy as np

emb_dim = 2
batch_size = 3
feat_num = 4
max_len = 5
seq_lens = [3, 2, 4]

seq_emb_list = []
start = 0
for feat_idx in range(feat_num):
    tmp = np.zeros([batch_size, max_len, emb_dim])
    for idx, seq_len in enumerate(seq_lens):
        tmp[idx, :seq_len, :] = np.arange(start, start + seq_len * emb_dim).reshape([seq_len, emb_dim])
        start += seq_len * emb_dim
    seq_emb_list.append(tmp)

seq_emb_listt = tf.transpose(seq_emb_list, [1, 0, 2, 3])
tf.print(seq_emb_listt)


def get_len(x):
    a = tf.math.count_nonzero(x, 0)
    print(a)
    a = tf.math.count_nonzero(tf.reduce_sum(tf.math.count_nonzero(x, 0), axis=-1), axis=0)
    print(a)

    return x


tf.map_fn(get_len, seq_emb_listt, dtype=tf.float64)

max_len = 5
repeat_rate = 0.7

print("max_len:{0}, repeat_rate:{1}, round(max_len * (1 + repeat_rate)):{2}".format(max_len, repeat_rate,
                                                                                    round(max_len * (1 + repeat_rate))))
print(
    "max_len * repeat_rate:{0},round(max_len * repeat_rate):{1},max_len:{2}, max_len * repeat_rate + max_len:{3}".format(
        max_len * repeat_rate, round(max_len * repeat_rate), max_len, round(max_len * repeat_rate) + max_len))

x = tf.reshape(tf.range(12), (3, 4))
ret = tf.unstack(seq_emb_list, axis=0)
# tf.print("seq_emb_list:", x, "p:", p, "q:", q, "r:", r, "t:", t)
tf.print("seq_emb_list:", seq_emb_list, "ret:", ret)
print(len(ret))

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_mean(b))
