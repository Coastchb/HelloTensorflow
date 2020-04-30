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

ckpt_dir = "log/ckpt"
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

ix_cutoff = int(len(y_shuffled)*0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train),len(y_test)))

######################################
# set the variables and placeholders #
######################################
dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
x_data = tf.placeholder(tf.int32, (None, max_sequene_length),name="x_data")
y_output = tf.placeholder(tf.int32, (None), name="y_output")

embedding_mat = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1.0, 1.0), name="emb")
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

weight = tf.Variable(tf.truncated_normal((rnn_size, 2), stddev=0.1), name="fc_w")
bias = tf.Variable(tf.constant(0.1, shape=[2]),name="fc_b")

####################
# define the model #
####################
def inference():
    with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
        output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
        output = tf.nn.dropout(output, dropout_keep_prob)

        output = tf.transpose(output, (1, 0, 2))
        last = tf.gather(output, int(output.get_shape()[0]) -1 )

        # FC layer
        logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

        # loss and optimizer, etc.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)
        loss = tf.reduce_mean(losses)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out,
                                                             1), tf.cast(y_output, tf.int64)), tf.float32), name="accuracy")
        return logits_out, loss, accuracy

def create_model(global_steps):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        logits, loss, acc = inference()
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_steps)
        return logits, loss, acc, train_op

##################
# start training #
##################

global_steps = tf.train.get_or_create_global_step()
logits, loss, acc, train_op = create_model(global_steps)

saver = tf.train.Saver()
with tf.Session() as sess:
    try:
        checkpoint_state = tf.train.get_checkpoint_state(ckpt_dir)
        if (checkpoint_state and checkpoint_state.model_checkpoint_path):
            print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            checkpoint_path = checkpoint_state.model_checkpoint_path
            #saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
            saver.restore(sess, checkpoint_path)
        else:
            print('No model to load at {}'.format(ckpt_dir))
            sess.run(tf.global_variables_initializer())
    except tf.errors.OutOfRangeError as e:
        print('Cannot restore checkpoint: {}'.format(e))


    # add training loss to summary
    summary_writer = tf.summary.FileWriter('log/train', sess.graph)
    tf.summary.scalar("training_loss", loss)
    train_summary = tf.summary.merge_all()

    training_finished = False
    while(not training_finished):
        shuffled_ix = np.random.permutation(np.arange(len(x_train)))
        x_train = x_train[shuffled_ix]
        y_train = y_train[shuffled_ix]
        num_batches = int(len(x_train)/batch_size) + 1
        for i in range(num_batches):
            step = sess.run(global_steps)
            if (step >= epochs * num_batches):
                print("Training reaches the max training step!")
                training_finished = True
                break

            print("training step: %d" % step)
            min_ix = i * batch_size
            max_ix = np.min((len(x_train), ((i+1) * batch_size)))
            x_train_batch = x_train[min_ix:max_ix]
            y_train_batch = y_train[min_ix:max_ix]

            train_dict = {x_data:x_train_batch, y_output:y_train_batch, dropout_keep_prob:0.5}
            sess.run(train_op, feed_dict=train_dict)

        if(training_finished):
            break

        # run training summary
        summary_writer.add_summary(sess.run(train_summary, feed_dict = train_dict), step)


        test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob: 1.0}
        temp_test_loss, temp_test_acc = sess.run([loss, acc], feed_dict=test_dict)
        # add test loss to summary
        values = [tf.Summary.Value(tag='testing_loss', simple_value=temp_test_loss), ]
        test_summary = tf.Summary(value=values)
        summary_writer.add_summary(test_summary, step)

        # save ckpt
        saver.save(sess, os.path.join(ckpt_dir, "rnn-ckpt"), global_step=step)
        print("checkpoint-%d stored" % step)

summary_writer.close()

