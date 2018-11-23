import tensorflow as tf
import numpy as np

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer());
w = v + 1;
print(v);
print(w);

def conv_relu(input, kernel_shape, bias_shape):

    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer());
    bias = tf.get_variable("bias", bias_shape,
                           initializer=tf.constant_initializer(0.0));
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding="SAME");
    return tf.nn.relu(conv + bias);

input_val1 = np.reshape(np.random.normal(0, 1, 3200),(1,10,10,32))
#input_val2 = np.random_normal([1,20,20,32]);

input_data = tf.placeholder(shape=([1,10,10,32]),dtype=tf.float32)
#x = conv_relu(input_data, kernel_shape=[5, 5, 32, 32], bias_shape=[32]);

def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32]);
    with tf.variable_scope("conv2"):
       return conv_relu(relu1, [5, 5, 32, 32], [32]);

output = my_image_filter(input_data)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  output_val = sess.run(output, feed_dict={input_data: input_val1})
  print(output_val)

#my_image_filter(input1)
#my_image_filter(input2)