import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

batch_size = 10
feat_num = 4
num_noisy_feat_safe = 3
uniform_distribution = tf.compat.v1.random_uniform(
    shape=[batch_size, feat_num],
    minval=0,
    maxval=None,
    dtype=tf.float32,
    seed=None,
    name=None
)
_, indices_part = tf.nn.top_k(uniform_distribution, num_noisy_feat_safe)

tf.print(uniform_distribution)
tf.print(indices_part, indices_part.shape)

indices_part = tf.compat.v1.to_int64(tf.sort(indices_part, -1))
tf.print(indices_part)

batch_axis = tf.tile(
    tf.expand_dims(tf.expand_dims(tf.range(batch_size), -1), -1),
    (1, num_noisy_feat_safe, 1)
)
tf.print(batch_axis)
batch_axis = tf.cast(batch_axis, dtype=tf.int64)
print('batch_axis:{0}'.format(batch_axis))
indices = tf.concat([batch_axis, tf.expand_dims(indices_part, 2)], -1)
indices = tf.compat.v1.to_int64(tf.reshape(indices, (-1, 2)))
val = tf.ones_like(indices[:, 0], dtype=tf.float32)

tf.print('indices:', indices, summarize=10)
tf.print('val:', val, summarize=10)

feature_mask = 1 - tf.compat.v1.sparse_to_dense(indices, [batch_size, feat_num], val)
tf.print('feature_mask:', feature_mask)


def generate_seq_feat_noise_mask(batch_size, feat_num, max_len, noise_prob=0.):
    """
    batch_size - Tensor or python int
    returns:
        noise_mask - [None, feat_num, max_len]
    """
    uniform_distribution = tf.compat.v1.random_uniform(
        shape=[batch_size, feat_num, max_len],
        minval=0.,
        maxval=1.,
        dtype=tf.float32,
        seed=None,
        name=None
    )
    return tf.compat.v1.to_float(uniform_distribution > noise_prob)


def generate_seq_noise_mask(seq_mask, max_len, noise_prob=0., min_keep_num=1, has_opposite=False):
    """
    seq_mask - [None, max_len]
    """
    batch_size = tf.shape(seq_mask, out_type=tf.int64)[0]
    # [None, 1, max_len]
    seq_noise_mask = generate_seq_feat_noise_mask(batch_size, 1, max_len, noise_prob=noise_prob)
    seq_noise_mask *= tf.expand_dims(seq_mask, 1)
    noise_seq_len = tf.reduce_sum(seq_noise_mask, [1, 2])
    if min_keep_num <= 0:
        return seq_noise_mask, noise_seq_len

    uniform_distribution = tf.compat.v1.random_uniform(
        shape=[batch_size, max_len],
        minval=0.,
        maxval=1.,
        dtype=tf.float32,
        seed=None,
        name=None
    )
    # seq with real value would be selected first
    uniform_distribution_keep = uniform_distribution + tf.squeeze(seq_noise_mask, 1)
    uniform_distribution_keep *= seq_mask

    min_keep_num_safe = min(max_len, min_keep_num)
    _, indices_part = tf.nn.top_k(uniform_distribution_keep, min_keep_num_safe)
    indices_part = tf.compat.v1.to_int64(tf.sort(indices_part, -1))
    batch_axis = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.range(batch_size), -1), -1),
        (1, min_keep_num_safe, 1)
    )
    indices = tf.concat([batch_axis, tf.expand_dims(indices_part, 2)], -1)
    indices = tf.compat.v1.to_int64(tf.reshape(indices, (-1, 2)))
    val = tf.ones_like(indices[:, 0], dtype=tf.float32)
    # [None, max_len], 1 denotes keeping
    keep_mask = tf.compat.v1.sparse_to_dense(indices, [batch_size, max_len], val)
    # discard those out of len
    keep_mask *= seq_mask
    # discard those already have min_keep_num
    keep_mask_final = keep_mask * tf.expand_dims(
        tf.compat.v1.to_float(noise_seq_len < min_keep_num_safe), -1)
    keep_mask_final = tf.expand_dims(keep_mask_final, 1)
    seq_noise_mask = tf.compat.v1.to_float((seq_noise_mask + keep_mask_final) > 0.)
    noise_seq_len = tf.reduce_sum(seq_noise_mask, [1, 2])
    if has_opposite:
        # seq with mask would be selected first
        uniform_distribution_drop = uniform_distribution - keep_mask
        uniform_distribution_drop += (1 - tf.squeeze(seq_noise_mask, 1))
        uniform_distribution_drop *= seq_mask
        _, indices_part = tf.nn.top_k(uniform_distribution_drop, min_keep_num_safe)
        indices_part = tf.compat.v1.to_int64(tf.sort(indices_part, -1))
        batch_axis = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(batch_size), -1), -1),
            (1, min_keep_num_safe, 1)
        )
        indices = tf.concat([batch_axis, tf.expand_dims(indices_part, 2)], -1)
        indices = tf.compat.v1.to_int64(tf.reshape(indices, (-1, 2)))
        val = tf.ones_like(indices[:, 0], dtype=tf.float32)
        # [None, max_len], 1 denotes dropping
        drop_mask = tf.compat.v1.sparse_to_dense(indices, [batch_size, max_len], val)
        # discard those out of len
        drop_mask *= seq_mask
        # discard those already have min_keep_num
        seq_len = tf.reduce_sum(seq_mask, axis=1)
        drop_mask *= tf.expand_dims(
            tf.compat.v1.to_float((seq_len - noise_seq_len) < min_keep_num_safe), -1)
        drop_mask = tf.expand_dims(drop_mask, 1)
        seq_noise_mask = tf.compat.v1.to_float((seq_noise_mask - drop_mask) > 0.)
        noise_seq_len = tf.reduce_sum(seq_noise_mask, [1, 2])

    return seq_noise_mask, noise_seq_len


seq_len = tf.constant([15, 14, 12])
seq_mask = tf.sequence_mask(seq_len, 20, dtype=tf.float32)
seq_a_noise_mask, noise_seq_a_len = generate_seq_noise_mask(seq_mask, 20, min_keep_num=3, noise_prob=0.5, has_opposite=True)
tf.print('seq_a_noise_mask:', seq_a_noise_mask, summarize=20)

seq_b_noise_mask = (1 - seq_a_noise_mask) * tf.expand_dims(seq_mask, 1)
tf.print('seq_b_noise_mask:', seq_b_noise_mask, summarize=20)
