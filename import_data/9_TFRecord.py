# -*- coding:utf-8 -*-
# @Time		:2018/11/21 4:16 PM
# @Author	:Coast Cao


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from functools import partial

### the example features are sequences

# mark 1: pay attention to the return of parse_single_sequence_example()
#         pay attention to what the map() function passes to lambda or remove_empty function  (two tensors instead of a tuple of two tensors)

utterance1 = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
utterance2 = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
utterance3 = np.array([[1.0,2.0,3.0]])
utterance4 = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
utterance5 = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])

def create_sequence_example(features):
    example = tf.train.SequenceExample(
        feature_lists = tf.train.FeatureLists(
            feature_list = {
                "feat": tf.train.FeatureList(
                    feature = [tf.train.Feature(
                        float_list = tf.train.FloatList(value=value)
                    ) for value in features]
                )
            }
        )
    )
    return example

utt_tfr = "tfrecords/utt.tfr"
with tf.io.TFRecordWriter(utt_tfr) as f:
    for utt in [utterance1,utterance2,utterance3,utterance4,utterance5]:
        f.write(create_sequence_example(utt).SerializeToString())


### Read TFRecord
def parse_example(example_proto):
    #print(type(example_proto))
    example = {
        "feat": tf.FixedLenSequenceFeature(shape=[3], dtype=tf.float32),
    }
    d = tf.parse_single_sequence_example(example_proto, sequence_features=example)
    # mark 1
    print("####")
    print(d)
    print(tf.shape(d[1]['feat'])[0])
    print("####")
    return d

dataset = tf.data.TFRecordDataset(utt_tfr)
dataset = dataset.map(parse_example)
#dataset = tf.data.Dataset.zip(dataset)
iter = dataset.make_one_shot_iterator()
next_element = iter.get_next()
sess = tf.Session()

print("\ndataset:")
print(dataset)
print("\nnext_element:")
print(next_element)
print("\nnext_element[1]:")
print(next_element[1])
print("\nnext_element[1]['feat']:")
print(next_element[1]['feat'])
print("\nsess.run(next_element):")
print(sess.run(next_element))

W = tf.ones(shape=[3,1])
output = tf.matmul(next_element[1]['feat'],W)
print("\nsess.run(output):")
print(sess.run(output))



# batch the dataset
print("\n\n## after shuffle, repeat and batch ##")

'''
def remove_empty(*x):
    return x[1]
'''
# mark 1
d1 = dataset.map(lambda x,y:y) # remove_empty is also OK; x,y can also be *x;
#d1 = d1.shuffle(200, reshuffle_each_iteration=True).repeat(3)
print(d1)
d1 = d1.padded_batch(2,padded_shapes={'feat':(None,3)},drop_remainder=True)  # specify the padding shape for each element(example) in dataset
print(d1)
it1 = d1.make_one_shot_iterator()
next_element = it1.get_next()
print(sess.run(next_element))
print(sess.run(next_element))
#print(sess.run(next_element))


len = tf.shape([1,2,3])
print(len)


print('\n')
print("####################")
print("# more while parse #")
print("####################")
print('\n')

def parse_example1(example_proto):
    #print(type(example_proto))
    example = {
        "feat": tf.FixedLenSequenceFeature(shape=[3], dtype=tf.float32),
    }
    _, d = tf.parse_single_sequence_example(example_proto, sequence_features=example)
    # mark 1
    print("####")
    print(d)
    print(tf.shape(d['feat'])[0])
    print("####")
    data = {}
    data['feat'] = d['feat']
    data['feat_len'] = tf.shape(d['feat'])[0]
    return data

dataset1 = tf.data.TFRecordDataset(utt_tfr)
dataset1 = dataset1.map(parse_example1)

dataset1 = dataset1.padded_batch(2,padded_shapes={'feat':(None,3), 'feat_len':()}, padding_values={'feat':100.0,'feat_len':0}, drop_remainder=True)  # specify the padding shape for each element(example) in dataset
print(dataset1)
print(sess.run(dataset1.make_one_shot_iterator().get_next()))


print('\n')
print("####################")
print("#  from generator  #")
print("####################")
print('\n')

def _generator(seqs):
    for seq in seqs:
        yield {"feat": seq, "feat_len": seq.shape[0]}


dataset2 = tf.data.Dataset.from_generator(
    lambda : _generator([utterance1,utterance2,utterance3,utterance4,utterance5]),
    {
        "feat": tf.float32,
        "feat_len": tf.int32,
    }, {
        "feat": tf.TensorShape([None,3]),
        "feat_len": tf.TensorShape([]),
    }
)
print(dataset2)
dataset2 = dataset2.padded_batch(2,padded_shapes={'feat':(None,3), 'feat_len':()}, padding_values={'feat':100.0,'feat_len':0}, drop_remainder=True)
print(dataset2)
it2 = dataset2.make_one_shot_iterator()
print(sess.run(it2.get_next()))