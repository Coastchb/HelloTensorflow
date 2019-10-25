# -*- coding:utf-8 -*-
# @Time		:2018/9/26 上午8:15
# @Author	:Coast Cao

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import requests
from zipfile import ZipFile
import re
from train_rnn import inference

########################
# set the hyparameters #
########################
epochs = 20
batch_size = 250
max_sequene_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005

#####################################
# load and preprocess the real data #
#####################################
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(os.path.join(data_dir,data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    # Save data to text file
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    # Open data from text file
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
        text_data = text_data[:-1]
text_data = [x.split('\t') for x in text_data if len(x)>=1]
#print(text_data)
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string

text_data_train = [clean_text(x) for x in text_data_train]

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequene_length, min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

ix_cutoff = int(len(y_shuffled)*0.10)
x_test = x_shuffled[:ix_cutoff]
y_test = y_shuffled[:ix_cutoff]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))

##################
# start predict #
##################

try:
    checkpoint_state = tf.train.get_checkpoint_state("log/ckpt")
    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
        print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
        checkpoint_path = checkpoint_state.model_checkpoint_path
    else:
        print('No model to load at {}'.format(save_dir))

except tf.errors.OutOfRangeError as e:
    print('Cannot restore checkpoint: {}'.format(e))

with tf.Session() as sess:
    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
    saver.restore(sess, checkpoint_path)
    get_accuracy = graph.get_tensor_by_name("model/inference/accuracy:0")

    x_data = graph.get_tensor_by_name("x_data:0")
    y_output = graph.get_tensor_by_name("y_output:0")
    dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
    acc = sess.run([get_accuracy], feed_dict={x_data:x_test, y_output:y_test, dropout_keep_prob: 1.0})
    print(acc)
