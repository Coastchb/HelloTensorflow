# -*- coding:utf-8 -*- 
# @Time		:2019/1/15 3:55 PM
# @Author	:Coast Cao
import tensorflow as tf

dense_layer = tf.layers.Dense(8)

#input1 = tf.constant([[1,2,3,4,5],[11,22,33,44,55]])
#output1= dense_layer(input1)
#print(output1.shape)

input2 = tf.constant([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,22]]])
output2= dense_layer(input2)
print(output2.shape)