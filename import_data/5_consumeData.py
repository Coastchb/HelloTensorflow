# -*- coding:utf-8 -*- 
# @Time		:2018/11/12 8:12 PM
# @Author	:Coast Cao

import tensorflow as tf

### mark 1: dict() is optional;  (dict(features7,labels5) can not be: [dict(features7,labels5]

dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()

sess.run(iterator.initializer)

result = tf.add(next_element, next_element)

while True:
    try:
        print(sess.run(result))
    except tf.errors.OutOfRangeError:
        print("reach the end of the dataset")
        break

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4,100])))
dataset3 = tf.data.Dataset.zip((dataset1,dataset2))

iterator3 = dataset3.make_initializable_iterator()

x, (y, z) = iterator3.get_next()

sess.run(iterator3.initializer)

while True:
    try:
        xx, yy, zz = sess.run((x,y,z))
        print(xx)
        print(yy)
        print(zz)
    except tf.errors.OutOfRangeError:
        print("reach the end of dataset")
        break

print("########################\n\n\n")

print("\n### image with grey value ###")
features4 = [[[1,2,3],[1,2,3]],[[2,3,4],[2,3,4]],[[3,4,5],[3,4,5]]]
dataset4 = tf.data.Dataset.from_tensor_slices(features4)
print(dataset4)
print(sess.run(dataset4.make_one_shot_iterator().get_next()))

print("\n### image with grey value and label ###")
labels5 = [0,1,1]
dataset5 = tf.data.Dataset.from_tensor_slices((features4, labels5))
print(dataset5)
print(sess.run(dataset5.make_one_shot_iterator().get_next()))

print("\n### image with RGB value and label ###")
features6 = [[[[1,1,1],[2,2,2],[3,3,3]],[[1,1,1],[2,2,2],[3,3,3]]],[[[2,2,2],[3,3,3],[4,4,4]],[[2,2,2],[3,3,3],[4,4,4]]],[[[3,3,3],[4,4,4],[5,5,5]],[[3,3,3],[4,4,4],[5,5,5]]]]
dataset6 = tf.data.Dataset.from_tensor_slices((features6,labels5))
print(dataset6)
print(sess.run(dataset6.make_one_shot_iterator().get_next()))

print("\n### image with RGB value and label ###")
features7 = {"R": [[[1,2,3],[1,2,3]],[[2,3,4],[2,3,4]],[[3,4,5],[3,4,5]]],
             "G": [[[11,22,33],[11,22,33]],[[22,33,44],[22,33,44]],[[33,44,55],[33,44,55]]],
             "B": [[[111,222,333],[111,222,333]],[[222,333,444],[222,333,444]],[[333,444,555],[333,444,555]]]}

# mark 1
dataset7 = tf.data.Dataset.from_tensor_slices((dict(features7),labels5))
print(dataset7)
print(sess.run(dataset7.make_one_shot_iterator().get_next()))

print("\n### shuffle ###")
dataset_0 = dataset.shuffle(5)
iter_0= dataset_0.make_one_shot_iterator()
next_element = iter_0.get_next()
for i in range(5):
    print(sess.run(next_element))

print("\n### repeat ###")
dataset_1 = dataset.shuffle(5).repeat()
iter_1= dataset_1.make_one_shot_iterator()
next_element = iter_1.get_next()
for i in range(11):
    print(sess.run(next_element))

print("\n### batch ###")
dataset_2 = dataset.shuffle(5).repeat().batch(2)
iter_2= dataset_2.make_one_shot_iterator()
next_element = iter_2.get_next()
for i in range(11):
    print(sess.run(next_element))
print(dataset_2)