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

# set the hyparameters
epochs = 20
batch_size = 250
max_sequene_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005

# load the real data
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exits(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(os.path.join(data_dir,data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode.split('\n')
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
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]




# set the variables and placeholders
dropout_keep_prob = tf.placeholder(tf.float32)
