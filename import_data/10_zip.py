# -*- coding:utf-8 -*- 
# @Time		:2018/11/27 8:47 AM
# @Author	:Coast Cao

import tensorflow as tf

# mark 1: pay attention to what the map() function passes to lambda or remove_empty function  (two tensors instead of a tuple of two tensors)

sess = tf.Session()

print("##### #####")
a = tf.data.Dataset.from_tensors(({'a':1, 'b':2},{'aa':11,'bb':22}))
print(a)
b = tf.data.Dataset.from_tensors(({'aaa':111, 'bbb':222},{'aaaa':1111,'bbbb':2222}))
print(b)

print(tf.data.Dataset.zip((a,b)))                           # just concatenate the two dataset

print("##### d1 #####")
d1 = tf.data.Dataset.from_tensor_slices([[1,2],[3,4]])      # each element of the list will be regarded as an example
d1 = d1.batch(1)
it1 = d1.make_one_shot_iterator()
n1 = it1.get_next()
print(n1)
print(sess.run(n1[0]))
print(sess.run(n1))
#print(sess.run(n1))

print("##### d2 #####")
d2 = tf.data.Dataset.from_tensors([[1,2],[3,4]])            # the whole list will be regarded as an example
print(d2)
d2 = d2.batch(1)
it2 = d2.make_one_shot_iterator()
n2 = it2.get_next()

print(sess.run(n2))

d3 = tf.data.Dataset.from_tensor_slices([[1,2],[3,4]])
d4 = tf.data.Dataset.from_tensors([[1,2],[3,4]])
d5=tf.data.Dataset.zip((d3,d4))
print(d5)
print(d5.make_one_shot_iterator().get_next())

print("##### dataset.map() #####")
def f(*a):
    b = []
    for i in a:
        print(i)
    return 0
# mark 1
d5 = d5.map(f)
print(d5)
print(sess.run(d5.make_one_shot_iterator().get_next()))
