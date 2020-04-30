import tensorflow as tf

inputs_pad = tf.reshape(tf.range(2*4*3), (8,3))
inputs_pad = tf.constant([[1,2,3]] * 9)
def stride_split(x):
    inputs_slices = []
    for i in range(2):
        start_idx = i * 4
        for j in range(4):
            # inputs_slices.append(tf.slice(inputs_pad, [start_idx+j, 0], [self.T, self.output_dim]))
            inputs_slices.append(x[start_idx + j: start_idx + j + 2, :])
    return inputs_slices


sess = tf.Session()
ret = tf.py_func(stride_split, [inputs_pad], [tf.int32]*8)
print(sess.run(ret))
print(sess.run(tf.convert_to_tensor(ret)))


def f4(x):
    out = []
    out.append(x[:1,:2])
    return out

c = tf.constant([ [[1,11],[2,22]] ,[[3,33],[4,44]]])
print(sess.run(tf.py_func(f4, [c], [tf.int32])))
print(inputs_pad)
print(sess.run(tf.py_func(f4, [inputs_pad], [tf.int32])))