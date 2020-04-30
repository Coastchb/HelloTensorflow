# -*- coding:utf-8 -*- 
# @Time		:2018/9/12 4:00 PM
# @Author	:Coast Cao

import tensorflow as tf

tensor_zero = tf.zeros((1));
tensor_one = tf.ones((2, 3));
tensor_fill = tf.fill((3, 4, 5), 2);
tensor_constant = tf.constant([[1,2,3],[4,5,6],[7,8,9]]);
tensor_constant = tensor_constant + 1; # modifying the constant is allowed
sim_zeros = tf.zeros_like(tensor_constant);
sim_ones = tf.ones_like(tensor_fill);
tensor_linear = tf.linspace(start=0.0, stop=1.0, num=3);
tensor_limit = tf.range(start=0.0, limit=1.0, delta=0.3);
tensor_rand_u = tf.random_uniform((3,4),minval=0, maxval=3);
tensor_rand_n = tf.random_normal((2,3), mean=0.0, stddev=1.0);
tensor_rand_nt = tf.truncated_normal((2,3), mean=0.0, stddev=1.0);
tensor_rand_sf = tf.random_shuffle(tensor_rand_n);
tensor_rand_c = tf.random_crop(tensor_rand_n, (1,2));

a = 1;
tensor_a = tf.constant(a);
tensor_A = tf.convert_to_tensor(a);

tensor_var = tf.constant([[[0,1,2],
  [3,4,5],
  [0,0,0],
  [0,0,0]],
 [[6,7,8],
  [9,10,11],
  [12,13,14],
  [0,0,0]],
 [[15,16,17],
  [18,19,20],
  [21,22,23],
  [24,25,26]]])

def get_target_in_axis1(input, dims):
  output = []
  for (x,d) in zip(input,dims):
    output.append(x[d])
  return output

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer());
  print("tensor_zero:");
  print(sess.run(tensor_zero));
  print("tensor_one:");
  print(sess.run(tensor_one));
  print("tensor_fill:");
  print(sess.run(tensor_fill));
  print("tensor_constant:");
  print(sess.run(tensor_constant));
  print("compare constant:");
  print(sess.run(tf.constant(1) < tf.constant(2)));
  print(sess.run(tf.constant(1) < 2));
  print("sim_zeros:");
  print(sess.run(sim_zeros));
  print("sim_ones:");
  print(sess.run(sim_ones));
  print("tensor_linear:");
  print(sess.run(tensor_linear));
  print("tensor_limit:");
  print(sess.run(tensor_limit));
  print("tensor_rand_u:");
  print(sess.run(tensor_rand_u));
  print("tensor_rand_n, tensor_rand_sf, tensor_rand_c:");
  print(sess.run([tensor_rand_n, tensor_rand_sf, tensor_rand_c]));
  print("tensor_rand_nt:");
  print(sess.run(tensor_rand_nt));
  print("tensor_a:");
  print(sess.run(tensor_a));
  print("tensor_A:");
  print(sess.run(tensor_A));
  print((tensor_a*2))
  print((tensor_a*2).shape)
  print(sess.run(tensor_a*2))

  print(sess.run(tensor_fill[:,-1,:]))

  print(sess.run(tensor_var))
  output_types = [tf.int32] * 3
  output = tf.py_func(get_target_in_axis1,[tensor_var,[1,2,3]],output_types)
  print(sess.run(output))
  print(sess.run(output * 3))

  batch_size = 3
  max_len = 4
  output1 = tf.tile(tf.reshape(tf.convert_to_tensor([output]),(batch_size,1,3)),(1, max_len, 1))
  print(sess.run(output1))

  concat_tensor = tf.concat((tensor_var, output1), axis=-1)
  print(len(concat_tensor.shape))
  print(concat_tensor.shape)
  assert concat_tensor.shape == (3,4,6)
