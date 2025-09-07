import tensorflow as tf

seq_h_lens = tf.constant([[[100.0, 0.0, 0.0], [200.0, 300.0, 0.0]], [[0.0, 12.0, 34.0], [13.0, 26.0, 0.0]]])

seq_h_lens = tf.where(seq_h_lens == 0, tf.ones_like(seq_h_lens), seq_h_lens)

print('seq_h_lens:{0}'.format(seq_h_lens))
